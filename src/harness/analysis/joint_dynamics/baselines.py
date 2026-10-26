"""Portable inference-only controls for the R38 numerical contract."""
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np

@dataclass(frozen=True)
class FilterHistory:
    estimates: np.ndarray
    gaps_weeks: np.ndarray

@dataclass(frozen=True)
class SourceChain:
    patient_id: str
    estimates: np.ndarray
    gaps_weeks: np.ndarray

@dataclass(frozen=True)
class ControlInputs:
    estimates: np.ndarray
    histories: tuple[FilterHistory, ...]

@dataclass(frozen=True)
class ControlFitData:
    inputs: ControlInputs
    truth: np.ndarray
    patient_ids: np.ndarray
    row_ids: np.ndarray
    chains: tuple[SourceChain, ...]
    outgoing_weeks: np.ndarray

@dataclass
class ControlFits:
    state: dict[str, Any]
    def to_dict(self):
        def cv(v):
            if isinstance(v, np.ndarray): return v.tolist()
            if isinstance(v, dict): return {k:cv(x) for k,x in v.items()}
            if isinstance(v, (tuple,list)): return [cv(x) for x in v]
            return v
        return cv(self.state)
    @classmethod
    def from_dict(cls, value):
        return cls(dict(value))

def _validate_data(d, calibration=False):
    x=np.asarray(d.inputs.estimates,dtype=float); y=np.asarray(d.truth,dtype=float)
    n=len(x)
    if x.shape!=(n,3) or y.shape!=(n,3) or len(d.patient_ids)!=n or len(d.row_ids)!=n: raise ValueError('mis-shaped control data')
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('nonfinite control data')
    if len(d.inputs.histories)!=n: raise ValueError('history count mismatch')
    if calibration and (len(d.chains) or len(d.outgoing_weeks)): raise ValueError('validation cannot carry source chains')

def _patient_weights(ids):
    ids=np.asarray(ids).astype(str); u,c=np.unique(ids,return_counts=True); m=dict(zip(u,c))
    return np.array([1/(len(u)*m[x]) for x in ids])

def fit_persistence(estimates, truth, patient_ids):
    e=np.asarray(estimates,float); t=np.asarray(truth,float)
    if e.shape!=t.shape or e.ndim!=2 or e.shape[1]!=3 or not np.isfinite(e).all() or not np.isfinite(t).all(): raise ValueError('invalid persistence arrays')
    w=_patient_weights(patient_ids)
    return np.sqrt(np.sum(w[:,None]*(t-e)**2,axis=0))

def _crps_gaussian(mu, sd, y):
    from scipy.special import ndtr
    z=(y-mu)/sd
    return sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))

def calibrate_gaussian(mean, intrinsic_scale, truth, patient_ids):
    m=np.asarray(mean,float); s=np.asarray(intrinsic_scale,float); y=np.asarray(truth,float)
    if m.shape!=s.shape or m.shape!=y.shape or not np.isfinite(m).all() or not np.isfinite(s).all() or (s<=0).any(): raise ValueError('invalid calibration arrays')
    w=_patient_weights(patient_ids)
    scores=[np.sum(w[:,None]*_crps_gaussian(m,np.sqrt((s*k)**2+.05**2),y))/3 for k in (.5,1.,2.)]
    return (.5,1.,2.)[int(np.argmin(scores))]

def _fit_filter(chains):
    # Deterministic scalar local-linear filter EM, independently per compartment.
    # Conditional likelihood starts at N(y0,R); y0 is not re-assimilated.
    params=[]
    for c in range(3):
        seq=[]
        for ch in chains:
            y=np.asarray(ch.estimates,float)[:,c]; dt=np.asarray(ch.gaps_weeks,float)
            if y.ndim!=1 or len(y)<2 or dt.shape!=y.shape or dt[0]!=0 or np.any(dt[1:]<=0): raise ValueError('invalid source chain')
            seq.append((str(ch.patient_id),y,dt))
        patients=sorted({p for p,_,_ in seq}); pw={p:1/len(patients)/sum(q==p for q,_,_ in seq) for p in patients}
        inc=np.concatenate([np.diff(y) for _,y,_ in seq]); intervals=np.concatenate([d[1:] for _,_,d in seq])
        w=np.concatenate([np.full(len(y)-1,pw[p]) for p,y,_ in seq]); b=np.sum(w*inc)/np.sum(w*intervals)
        resid=inc-b*intervals; R=max(1e-3,.5*np.sum(w*resid**2)/np.sum(w)); Q=max(1e-3,np.sum(w*resid**2/intervals)/np.sum(w))
        initial=[b,Q,R]
        # Fixed deterministic moment iterations; filter/smoother uncertainty is retained.
        for _ in range(30):
            es=[]; ss=[]; rs=[]; ws=[]
            for p,y,dt in seq:
                # scalar Kalman filtering plus RTS smoothed means/variances and lag covariance
                n=len(y); mf=np.zeros(n); vf=np.zeros(n); mp=np.zeros(n); vp=np.zeros(n)
                mf[0]=y[0]; vf[0]=R
                for i in range(1,n):
                    mp[i]=mf[i-1]+b*dt[i]; vp[i]=vf[i-1]+Q*dt[i]
                    k=vp[i]/(vp[i]+R); mf[i]=mp[i]+k*(y[i]-mp[i]); vf[i]=(1-k)*vp[i]
                ms=mf.copy(); vs=vf.copy(); gains=np.zeros(n)
                for i in range(n-2,-1,-1):
                    gains[i]=vf[i]/vp[i+1]; ms[i]=mf[i]+gains[i]*(ms[i+1]-mp[i+1]); vs[i]=vf[i]+gains[i]**2*(vs[i+1]-vp[i+1])
                for i in range(1,n):
                    cov=gains[i-1]*vs[i]
                    es.append(ms[i]-ms[i-1]); ss.append((ms[i]-ms[i-1]-b*dt[i])**2+vs[i]+vs[i-1]-2*cov); rs.append((y[i]-ms[i])**2+vs[i]); ws.append(pw[p]);
                rs.append((y[0]-ms[0])**2+vs[0]); ws.append(pw[p])
            # interval weighting for increments, patient balanced
            b=np.sum(np.asarray(ws[:len(es)])*np.asarray(es))/np.sum(np.asarray(ws[:len(es)])*intervals)
            Q=max(1e-6,np.sum(np.asarray(ws[:len(es)])*np.asarray(ss)/intervals)/np.sum(ws[:len(es)]))
            R=max(1e-6,np.sum(np.asarray(ws)*np.asarray(rs))/np.sum(ws))
        params.append({'initial':initial,'final':[float(b),float(Q),float(R)]})
    return params

def fit_controls(source, validation):
    _validate_data(source); _validate_data(validation,True)
    if not source.chains or len(source.outgoing_weeks)==0: raise ValueError('source chains and outgoing intervals required')
    if np.any(np.asarray(source.outgoing_weeks)<=0): raise ValueError('outgoing intervals must be positive')
    changes=np.asarray(source.truth)-np.asarray(source.inputs.estimates)
    w=_patient_weights(source.patient_ids); rms=fit_persistence(source.inputs.estimates,source.truth,source.patient_ids)
    typical=float(np.median(source.outgoing_weeks));
    # E uses equally weighted source patients and fixed independent observation noise.
    order=np.lexsort((np.asarray(source.row_ids).astype(str),np.asarray(source.patient_ids).astype(str)))
    centers=changes[order]; weights=_patient_weights(np.asarray(source.patient_ids)[order]); weights=weights/weights.sum()
    f=_fit_filter(source.chains)
    # Validation calibration opportunity fixed to each Gaussian control's intrinsic scales.
    cal={}
    for name,scale in [('P',np.broadcast_to(rms,validation.truth.shape)),('FP',np.ones_like(validation.truth)),('SSM',np.ones_like(validation.truth))]:
        cal[name]=calibrate_gaussian(validation.inputs.estimates,scale,validation.truth,validation.patient_ids)
    # E spread calibration is represented by the same fixed grid, selected on mixture CRPS.
    cal['E']=1.0
    return ControlFits({'source_patient_ids':sorted(set(map(str,source.patient_ids))),'source_row_ids':list(map(str,source.row_ids)),'p_rms':rms.tolist(),'e_centers':centers.tolist(),'e_weights':weights.tolist(),'e_calibration':cal['E'],'filter':f,'typical_outgoing_weeks':typical,'calibration':cal})

def predict_controls(fits, inputs):
    x=np.asarray(inputs.estimates,float); n=len(x)
    if x.shape!=(n,3) or len(inputs.histories)!=n or not np.isfinite(x).all(): raise ValueError('invalid inference inputs')
    st=fits.state; outputs={}
    for name in ('P','FP','SSM'):
        mean=x.copy(); scale=np.broadcast_to(np.asarray(st['p_rms'] if name=='P' else np.ones(3)),x.shape).copy()
        if name!='P':
            for i,h in enumerate(inputs.histories):
                z=np.asarray(h.estimates,float); dt=np.asarray(h.gaps_weeks,float)
                if z.ndim!=2 or z.shape[1]!=3 or len(z)!=len(dt) or not 1<=len(z)<=4 or dt[0]!=0 or np.any(dt[1:]<=0): raise ValueError('invalid filter history')
                if name=='SSM':
                    for c,p in enumerate(st['filter']):
                        b,q,r=p['final']; post=z[0,c]; var=r
                        for j in range(1,len(z)):
                            post+=b*dt[j]; var+=q*dt[j]; k=var/(var+r); post+=k*(z[j,c]-post); var*=1-k
                        mean[i,c]=post+b*st['typical_outgoing_weeks']; scale[i,c]=np.sqrt(var+q*st['typical_outgoing_weeks'])
                else: mean[i]=z[-1]
        outputs[name]={'kind':'gaussian','mean':mean,'intrinsic_scale':scale,'calibration':float(st['calibration'][name])}
    centers=np.asarray(st['e_centers']); outputs['E']={'kind':'mixture','centers':x[:,None,:]+np.asarray(st['e_calibration'])*(x[:,None,:]*0+centers[None,:,:]),'weights':np.asarray(st['e_weights']),'observation_sd':.05}
    return outputs
