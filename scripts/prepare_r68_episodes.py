#!/usr/bin/env python3
"""Bounded MU episode/provenance inventory; no pixel reads or model fitting."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,re
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

MEMBERSHIP='4df898761ecdb299bc87f99749fbebc8faab294043d454c4055850fcc676e2d6'
CLINICAL='4341badf954511d67fc52b82b3b488ed7323f7e46d249a354ff9b3a62269c79b'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
MRI_COLS=('BQ','BR','BS','BT','BU','BV')
EVENTS={'first_procedure':'AI','initial_chemo_start':'AL','initial_chemo_end':'AM','initial_rt_start':'AO','initial_rt_end':'AP','first_progression':'AS','further_progression':'AT','new_treatment':'AV','additional_start':'AY','additional_end':'AZ','second_additional_start':'BD','second_additional_end':'BE','immunotherapy_start':'BI','immunotherapy_end':'BJ','brachytherapy':'BM','other_start':'BO','other_end':'BP'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def missing(v):return v is None or str(v).strip().upper() in ('','N/A','NA','NAN','NONE','UNKNOWN')
def day(v):
    if missing(v):return None
    try:
        f=float(v)
        return int(f) if f.is_integer() else None
    except (ValueError,TypeError):return None

def clinical_rows(path,allowed):
    # Decode only header cells and allowlisted patient cells. Shared-string
    # elements for other rows are never interpreted as clinical values.
    with zipfile.ZipFile(path) as z:
        strings=list(ET.fromstring(z.read('xl/sharedStrings.xml')))
        def value(c):
            v=c.find('s:v',NS)
            return ''.join(strings[int(v.text)].itertext()) if c.get('t')=='s' else v.text if v is not None else ''.join(c.itertext())
        records={};headers={}
        for event,row in ET.iterparse(z.open('xl/worksheets/sheet2.xml'),events=('end',)):
            if row.tag!='{'+NS['s']+'}row':continue
            cells={re.sub(r'\d+','',c.attrib['r']):c for c in row}
            if row.attrib['r']=='1':headers={k:value(c) for k,c in cells.items()};row.clear();continue
            patient=value(cells['A']) if 'A' in cells else None
            if patient not in allowed:row.clear();continue
            selected={k:value(cells[k]).strip() for k in headers if (k in ('O','P') or (len(k)==2 and k>='AI')) and k in cells}
            if patient in records:raise ValueError('Duplicate allowed clinical row')
            records[patient]={'row':int(row.attrib['r']),'values':selected};row.clear()
        return records,headers

def count(records):return {'patients':len({r['patient'] for r in records}),'episodes':len(records)}

def clinical_completeness(records,rows):
    result={}
    for name,col in {'procedure_day':'AI','initial_chemo_name':'AK','initial_chemo_start':'AL','rt_start':'AO','rt_end':'AP','delivered_dose':'AQ','delivered_fractions':'AR','death_day':'P'}.items():
        result[name]=count([r for r in records if not missing(rows[r['patient']]['values'].get(col))])
    result['verified_cutoff_known_plan']={'patients':0,'episodes':0}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--asset-index',type=Path,required=True);p.add_argument('--prepared-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    membership=a.base/'mu/membership.json';clinical=a.base/'mu/timing-sources/clinical.xlsx'
    assert sha(membership)==MEMBERSHIP and sha(clinical)==CLINICAL
    members=json.loads(membership.read_text());roles={p:g for g in ('fitting','selection') for p in members[g]};allowed=set(roles)
    assert len(allowed)==131 and not allowed.intersection(members['reserved']) and not allowed.intersection(members['pilot_scoring'])
    assets=json.loads(a.asset_index.read_text());assert set(assets['allowed_patients'])==allowed
    source=defaultdict(dict)
    suffixes={'CT1':'brain_t1c','T1':'brain_t1n','T2':'brain_t2w','FLAIR':'brain_t2f','annotation':'tumorMask'}
    for e in assets['entries']:
        assert e['patient'] in allowed
        for key,suffix in suffixes.items():
            if e['path'].endswith('_'+suffix+'.nii.gz'):source[(e['patient'],e['visit'])][key]=dict(e,on_disk=Path(e['path']).is_file())
    prepared={}
    for role in ('R','P','V','A'):
        for r in json.loads((a.prepared_root/'data'/f'{role}.json').read_text()):
            if r.get('dataset')!='MU' or r['patient'] not in allowed:continue
            prepared[(r['patient'],r['visit'])]={k:str((a.prepared_root/'code'/r[k]).resolve()) if not Path(r[k]).is_absolute() else r[k] for k in ('images','mask')}
    rows,headers=clinical_rows(clinical,allowed);assert set(rows)==allowed
    assert all('Timepoint_'+str(i) in headers[c] for i,c in enumerate(MRI_COLS,1))
    knowledge_headers=[v for v in headers.values() if re.search(r'plan|decision|order|known|available|recorded_at|documented_at',v,re.I)]
    assert not knowledge_headers, 'New knowledge-time source fields require explicit interpretation'
    prior=json.loads((a.base/'cavity-transition-review-v1/private-transitions.json').read_text())
    prior={(r['patient'],r['cutoff'],r['target']):{'flags':r['flags'],'review_status':r['review_status']} for r in prior if r['patient'] in allowed}
    old_queue=json.loads((a.base/'cavity-transition-review-v1/private-review-queue.json').read_text())
    old_queue=[{'patient':r['patient'],'cutoff':r['cutoff'],'target':r['target'],'flags':r['flags']} for r in old_queue if r['patient'] in allowed]
    patients=[];episodes=[];issues=[]
    for patient in sorted(allowed):
        record=rows[patient];v=record['values'];group=roles[patient]
        times={f'Timepoint_{i}':day(v.get(c)) for i,c in enumerate(MRI_COLS,1)}
        valid=[t for t in times.values() if t is not None];bad_order=any(y<=x for x,y in zip(valid,valid[1:]))
        if bad_order:issues.append({'patient':patient,'issue':'nonincreasing MRI days; chronology unresolved'})
        for i,c in enumerate(MRI_COLS,1):
            if not missing(v.get(c)) and day(v.get(c)) is None:issues.append({'patient':patient,'visit':f'Timepoint_{i}','issue':'unparseable MRI day'})
        event_days={name:day(v.get(col)) for name,col in EVENTS.items()}
        rt_start,rt_end=event_days['initial_rt_start'],event_days['initial_rt_end']
        rt_valid=rt_start is not None and rt_end is not None and rt_start<=rt_end and str(v.get('AN','')).lower()=='yes'
        death_day=day(v.get('P'));death_documented=v.get('O')=='1'
        patient_scans=[]
        for i,c in enumerate(MRI_COLS,1):
            visit=f'Timepoint_{i}';files=source.get((patient,visit),{})
            if missing(v.get(c)) and not files:continue
            t=times[visit];scan={'visit':visit,'recorded_diagnosis_relative_day':t,'date_source_cell':c+str(record['row']),'date_quality':'unresolved_trajectory' if bad_order else 'missing_or_invalid' if t is None else 'released_integer_day_not_independently_verified','source_assets':files,'prepared_arrays':prepared.get((patient,visit)),'available_sequences':[s for s in suffixes if s!='annotation' and s in files],'annotation_id':files.get('annotation',{}).get('path'),'annotation_review':'release-level neuroradiologist refinement; no case-level longitudinal adjudication receipt'}
            scan['complete_structural_panel']=len(scan['available_sequences'])==4
            patient_scans.append(scan)
        post=[s for s in patient_scans if s['complete_structural_panel'] and s['recorded_diagnosis_relative_day'] is not None and rt_valid and s['recorded_diagnosis_relative_day']>rt_end and not bad_order]
        first_post=min(post,key=lambda s:(s['recorded_diagnosis_relative_day'],s['visit']))['visit'] if post else None
        patients.append({'patient':patient,'split':group,'clinical_row':record['row'],'source_fields':{headers[c]:{'value':val,'cell':c+str(record['row']),'knowledge_time':None} for c,val in v.items()},'event_days':event_days,'scans':patient_scans,'chronology_unresolved':bad_order,'first_released_post_initial_rt_visit':first_post,'death_record':{'documented':death_documented,'day':death_day,'zero_semantics':'alive, lost follow-up or unknown; not a verified alive/censoring label'}})
        for current in patient_scans:
            t=current['recorded_diagnosis_relative_day'];date_ok=t is not None and not bad_order
            past=[s for s in patient_scans if date_ok and s['complete_structural_panel'] and s['recorded_diagnosis_relative_day'] is not None and s['recorded_diagnosis_relative_day']<t]
            future=[s for s in patient_scans if date_ok and s['recorded_diagnosis_relative_day'] is not None and s['recorded_diagnosis_relative_day']>t]
            targets={}
            for horizon in (84,168):
                target=min(future,key=lambda s:(abs(s['recorded_diagnosis_relative_day']-t-horizon),s['recorded_diagnosis_relative_day'],s['visit'])) if future else None
                gap=target['recorded_diagnosis_relative_day']-t if target else None
                status='unresolved_cutoff_date' if not date_ok else 'no_later_recorded_mri' if target is None else 'out_of_window' if abs(gap-horizon)>28 else 'matched'
                usable=status=='matched' and current['complete_structural_panel'] and current['annotation_id'] is not None and target['complete_structural_panel'] and target['annotation_id'] is not None
                conflicts=[]
                if death_documented and death_day is not None and t is not None and death_day<t:conflicts.append('cutoff after recorded death')
                if status=='matched' and death_documented and death_day is not None and target['recorded_diagnosis_relative_day']>death_day:conflicts.append('target after recorded death')
                targets[str(horizon)]={'window_days':[horizon-28,horizon+28],'status':status,'target':target if status=='matched' else None,'nearest_later_recorded_visit':None if target is None else target['visit'],'realized_gap_days':gap,'usable_released_change_labels':usable and not conflicts,'date_conflicts':conflicts,'death_before_window_end':bool(date_ok and death_documented and death_day is not None and t<death_day<=t+horizon+28),'prior_transition_inspection':prior.get((patient,current['visit'],target['visit'])) if target else None,'longitudinal_expert_adjudication':None}
            event_audit={name:{'recorded_event_day':d,'event_not_after_cutoff':None if d is None or t is None else d<=t,'knowledge_time':None,'usable_as_verified_cutoff_context':False,'source_cell':EVENTS[name]+str(record['row'])} for name,d in event_days.items()}
            episodes.append({'episode_id':patient+'_'+current['visit'],'patient':patient,'split':group,'observations':{'current':current,'past':sorted(past,key=lambda s:s['recorded_diagnosis_relative_day']),'requested_horizon_days':84,'cutoff_known_treatment_plan':None,'cutoff_known_plan_dose':None,'cutoff_known_regimen':None,'clinical_context_missing':True},'phase':{'name':'follow-up MRI after recorded initial radiotherapy completion','event_defined_candidate':current['visit']==first_post,'post_initial_rt_event':current in post,'cutoff_knowledge_verified':False,'reason':'No source documentation/availability timestamp for treatment completion or intent'},'clinical_event_audit_not_predictors':event_audit,'evaluation_only':targets,'eligible_for_R68_clinical_comparison':False})
    coverage={}
    for group in ('fitting','selection'):
        rs=[r for r in episodes if r['split']==group];coverage[group]={}
        for phase in ('all_recorded_cutoffs','post_initial_rt_event','first_post_initial_rt_candidate'):
            ps=rs if phase=='all_recorded_cutoffs' else [r for r in rs if r['phase']['post_initial_rt_event' if phase=='post_initial_rt_event' else 'event_defined_candidate']]
            coverage[group][phase]={'cutoffs':count(ps),'with_past_released_scan':count([r for r in ps if r['observations']['past']]),'retrospective_field_completeness_not_input_validity':clinical_completeness(ps,rows),'horizons':{}}
            for h in ('84','168'):
                matched=[r for r in ps if r['evaluation_only'][h]['status']=='matched'];usable=[r for r in ps if r['evaluation_only'][h]['usable_released_change_labels']]
                coverage[group][phase]['horizons'][h]={'status_counts':dict(Counter(r['evaluation_only'][h]['status'] for r in ps)),'matched':count(matched),'usable_released_change_labels':count(usable),'matched_with_history':count([r for r in usable if r['observations']['past']]),'death_before_window_end':count([r for r in ps if r['evaluation_only'][h]['death_before_window_end']]),'matched_with_unusable_assets_or_date_conflict':count([r for r in matched if not r['evaluation_only'][h]['usable_released_change_labels']]),'strict_clinical_context_and_adjudication_eligible':{'patients':0,'episodes':0}}
                coverage[group][phase]['horizons'][h]['usable_pair_retrospective_field_completeness_not_input_validity']=clinical_completeness(usable,rows)
    completeness={g:{headers[c]:sum(not missing(rows[p]['values'].get(c)) for p in members[g]) for c in ('AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','O','P',*MRI_COLS)} for g in ('fitting','selection')}
    proposed=[r for r in episodes if r['phase']['post_initial_rt_event'] and r['evaluation_only']['84']['usable_released_change_labels']]
    review=[]
    for group in ('fitting','selection'):
        ordered=sorted([r for r in proposed if r['split']==group],key=lambda r:hashlib.sha256(('r68-review|'+r['episode_id']).encode()).hexdigest())
        selected=[];seen=set()
        for r in ordered:
            if r['patient'] not in seen:selected.append(r);seen.add(r['patient'])
            if len(selected)==3:break
        review.extend({'episode_id':r['episode_id'],'patient':r['patient'],'split':group,'current_annotation':r['observations']['current']['annotation_id'],'target_annotation':r['evaluation_only']['84']['target']['annotation_id'],'current_assets':r['observations']['current']['source_assets'],'target_assets':r['evaluation_only']['84']['target']['source_assets'],'review_status':'qualified longitudinal review required; no adjudication supplied','questions':['Are FLAIR-abnormality boundaries comparable across visits?','Is apparent change attributable to anatomy, postoperative cavity, acquisition/alignment or annotation?','Label ambiguity explicitly; preserve stable and changing cases.'],'prior_review':r['evaluation_only']['84']['prior_transition_inspection']} for r in selected)
    provenance={'membership_sha256':MEMBERSHIP,'clinical_sha256':CLINICAL,'asset_index_sha256':sha(a.asset_index),'source_url':'https://www.cancerimagingarchive.net/wp-content/uploads/MU-Glioma-Post_ClinicalData-July2025.xlsx','script_sha256':sha(__file__),'scopes':{'fitting':99,'selection':32,'reserved_requested':0,'pilot_scoring_requested':0},'MRI_pixels_read':0,'labels_recomputed_or_changed':0,'new_model_fits':0,'date_semantics':'Released diagnosis-relative integer days; no independent acquisition-date verification','annotation_convention':'MU raw labels: 1 core,2 FLAIR abnormality,3 enhancing,4 cavity; model-native map [0,2,3,1,4]; retain separate cavity'}
    manifest={'identity':'r68-preparation-v1','provenance':provenance,'clinical_headers':headers,'patients':patients,'episodes':episodes,'review_packet':review,'existing_review_queue':old_queue,'existing_review_assets':str(a.base/'cavity-transition-review-v1/diagnostic-slices.png'),'issues':issues,'clinical_inputs_gate':'Closed: no documented information-availability timestamps or treatment plans','matching_rule':'Nearest later recorded MRI day within nominal 12/24 weeks +/-28 days; earliest-day/visit tie; assets and death conflicts reported after date matching; no outcome-change, treatment-completion-after-cutoff, or survival selection'}
    # Illustrations are selected by field semantics, never by prediction errors.
    examples=[]
    descriptions={
        'ever_chemo_before_actual_start':'An ever-treated flag can be present at a cutoff preceding the recorded treatment start; it is not a cutoff-known plan.',
        'dose_without_plan_timestamp':'A recorded course dose has no plan/order or availability timestamp; do not use it as intended dose at the cutoff.',
        'completed_rt_without_knowledge_timestamp':'Recorded RT completion before MRI defines a retrospective event phase, but does not verify documentation availability at that MRI.',
        'death_zero_not_confirmed_survival':'Death code zero includes alive, lost follow-up and unknown; it cannot establish survival through the forecast window.'}
    for kind in descriptions:
        for r in episodes:
            v=rows[r['patient']]['values'];t=r['observations']['current']['recorded_diagnosis_relative_day'];start=day(v.get('AL'));end=day(v.get('AP'))
            matches={'ever_chemo_before_actual_start':str(v.get('AJ','')).lower()=='yes' and t is not None and start is not None and start>t,'dose_without_plan_timestamp':not missing(v.get('AQ')),'completed_rt_without_knowledge_timestamp':r['phase']['event_defined_candidate'] and end is not None and t is not None and end<t,'death_zero_not_confirmed_survival':v.get('O')=='0'}
            if matches[kind]:
                cols={'ever_chemo_before_actual_start':('AJ','AK','AL'),'dose_without_plan_timestamp':('AN','AO','AP','AQ','AR'),'completed_rt_without_knowledge_timestamp':('AN','AO','AP'),'death_zero_not_confirmed_survival':('O','P')}[kind]
                examples.append({'kind':kind,'episode_id':r['episode_id'],'split':r['split'],'cutoff_day':t,'source_cells':{c+str(rows[r['patient']]['row']):v.get(c) for c in cols},'information_available_at':None,'interpretation':descriptions[kind]});break
    manifest['private_field_examples']=examples
    write(a.out/'episodes-private.json',manifest)
    aggregate={'identity':'r68-preparation-v1','status':'preparation_complete_clinical_context_and_longitudinal_adjudication_blocked','provenance':provenance,'coverage':coverage,'clinical_field_nonmissing_patients':completeness,'source_knowledge_time_columns':knowledge_headers,'strictly_usable_R68_patients':0,'issue_counts':dict(Counter(i['issue'] for i in issues)),'recommendation':{'phase':'follow-up MRI after recorded initial radiotherapy completion, provisional until documentation timing is verified','horizon_days':84,'tolerance_days':28,'target':'native FLAIR-abnormality change (MU source label 2 / model label 3)','not_a_training_authorization':True,'rationale':'Recorded initial RT completion provides a common event anchor. Use all later cutoffs to retain history support; first-post-RT-only is inventoried but not recommended. The 84-day window has more patient and pair support than 168 days. No prediction performance or outcome-change values inform this choice.'},'annotation_review':{'released_masks':'automatic segmentation manually refined/validated by neuroradiologists at release; case-level independent longitudinal-change adjudication not supplied','existing_agent_inspections':len(prior),'expert_adjudications_available':0,'new_bounded_review_packet':len(review),'targets_relabelled':0},'specific_external_needs':['Dated treatment plans/orders and their documentation/availability timestamps; intended regimen/dose, revisions and historical delivery records linked to allowed cutoffs','Source clarification for nonincreasing MRI-day trajectory, scan/death contradictions and workbook Timepoint_6 dictionary typo','Qualified reviewer for bounded FLAIR longitudinal pairs; existing agent review is not expert adjudication'],'missingness_policy':'NA is unknown, not no treatment. Death=0 is alive/lost-follow-up/unknown and is never treated as confirmed survival. Future outcomes remain evaluation-only.'}
    aggregate['field_examples']=[{'kind':e['kind'],'interpretation':e['interpretation'],'patient_values':'private manifest only'} for e in examples]
    aggregate['data_accounting']={}
    for group in ('fitting','selection'):
        rs=[r for r in episodes if r['split']==group]
        checks={
            'recorded_or_released_visits':lambda r:True,
            'released_structural_panel':lambda r:r['observations']['current']['complete_structural_panel'],
            'clinical_date_without_released_panel':lambda r:r['observations']['current']['recorded_diagnosis_relative_day'] is not None and not r['observations']['current']['complete_structural_panel'],
            'released_panel_missing_date':lambda r:r['observations']['current']['complete_structural_panel'] and r['observations']['current']['recorded_diagnosis_relative_day'] is None,
            'released_panel_missing_annotation':lambda r:r['observations']['current']['complete_structural_panel'] and r['observations']['current']['annotation_id'] is None,
            'cutoff_after_recorded_death':lambda r:'cutoff after recorded death' in r['evaluation_only']['84']['date_conflicts'],
            'source_panel_or_annotation_not_on_disk':lambda r:any(not f['on_disk'] for f in r['observations']['current']['source_assets'].values()),
            'existing_prepared_arrays':lambda r:bool(r['observations']['current']['prepared_arrays']) and all(Path(p).is_file() for p in r['observations']['current']['prepared_arrays'].values())}
        aggregate['data_accounting'][group]={name:count([r for r in rs if test(r)]) for name,test in checks.items()}
        aggregate['data_accounting'][group]['date_conflict_pairs_by_horizon']={h:count([r for r in rs if r['evaluation_only'][h]['status']=='matched' and r['evaluation_only'][h]['date_conflicts']]) for h in ('84','168')}
    aggregate['recommendation']['history_support']={g:coverage[g]['post_initial_rt_event']['horizons']['84']['matched_with_history'] for g in ('fitting','selection')}
    aggregate['recommendation']['history_limitation']='The first-post-RT-only restriction would leave just 5 fitting and 3 selection pairs with history; all post-RT cutoffs retain more history, but clinical information-availability and adjudication gates remain closed.'
    aggregate['scope_limit']='Zero fully qualified R68 patients established in checked assets; not evidence that source institutions lack the required records. Released masks are not unreviewed, but per-scan refinement does not establish longitudinal biological-change truth.'
    write(a.out/'aggregate.json',aggregate);print(json.dumps({'coverage':coverage,'issues':aggregate['issue_counts'],'strictly_usable':0,'review_cases':len(review)},indent=2))


if __name__=='__main__':main()
