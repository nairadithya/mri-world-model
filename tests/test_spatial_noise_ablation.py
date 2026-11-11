import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/spatial/run_spatial_noise_ablation.py"
spec = importlib.util.spec_from_file_location("spatial_noise_ablation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SpatialNoiseAblationTests(unittest.TestCase):
    def test_week_bin_timing_uses_suffix_only_for_order(self):
        self.assertEqual(module.gap_days("week-003-1", "week-004-9"), 7)
        self.assertEqual(module.gap_days("week-003-1", "week-003-2"), 0)
        with self.assertRaises(ValueError): module.week_order("week-3.5")

    def test_history_never_bridges_unusable_intermediate(self):
        visits = [module.ChronologyVisit("a", "p", True, None, (True,)*4, {"id":"a"}),
                  module.ChronologyVisit("b", "p", False, 7, (False,)*4, None),
                  module.ChronologyVisit("c", "p", True, 7, (True,)*4, {"id":"c"}),
                  module.ChronologyVisit("d", "p", True, 7, (True,)*4, {"id":"d"})]
        rows = module.build_history_rows(module.FullChronology(tuple(visits), True))
        self.assertEqual([(r.cutoff_visit_id,r.target_visit_id) for r in rows], [("c","d")])

    def test_build_rows_keeps_same_week_target_and_compartment_order(self):
        m=[]; inv=[]; prep=[]; pairs=[]
        for i,(visit,day,vals) in enumerate([("week-000",0,[1.,2.,3.]),("week-001",7,[4.,5.,6.]),("week-001-1",8,[7.,8.,9.]),("week-002",14,[10.,11.,12.])]):
            vid=f"v{i}"; row={"patient_id":"p","visit":visit,"visit_id":vid,"day":day}
            m.append(row); record={**row,"values_mm3":dict(zip(module.COMPARTMENTS,vals)),"mask_relative":"mask"}; inv.append(record)
            prep.append({**record,"pregrid":"same"})
        pairs=[{"patient_id":"p","source_visit":"week-000","target_visit":"week-001","usable":True,"exclusions":[]},
               {"patient_id":"p","source_visit":"week-001","target_visit":"week-001-1","usable":True,"exclusions":[]},
               {"patient_id":"p","source_visit":"week-001-1","target_visit":"week-002","usable":True,"exclusions":[]}]
        rows,excluded=module.build_rows(m,inv,prep,pairs)
        self.assertEqual(len(rows),2); self.assertEqual(excluded,{"invalid_historical_gap":1})
        self.assertEqual(rows[0]["target"]["values_mm3"]["necrotic_non_enhancing"],4.)
        self.assertEqual(rows[0]["target"]["values_mm3"]["edema_flair"],6.)
        self.assertTrue(rows[1]["primary_history"])

    def test_load_inputs_rejects_cache_source_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); base=root/"bundle"; state=base/"results/comparison/seed42/state"; state.mkdir(parents=True)
            (base/"results/comparison").mkdir(exist_ok=True)
            (state/"best.pt").write_bytes(b"checkpoint"); (state/"features.pt").write_bytes(b"features")
            (state/"direct.predictions.npz").write_bytes(b"direct")
            anatomy=root/"outputs/anatomy_harmonization.json"; anatomy.parent.mkdir(parents=True)
            anatomy.write_text(json.dumps({"visits":[]}))
            inventory={"source_hashes":{"manifest":module.sha256(anatomy),"geometry":"g","protocol":"p"},"visits":[]}
            inv=root/"outputs/representation-comparison/inventory.json"; inv.parent.mkdir(parents=True); inv.write_text(json.dumps(inventory))
            prepared={"inventory_sha256":module.json_hash(inventory),"source_hashes":inventory["source_hashes"],"visits":[],"pairs":[]}
            prep=base/"training_prepared.json"; prep.write_text(json.dumps(prepared))
            run={"prepared_sha256":module.sha256(prep),"normalization":{"mean":[0,0,0],"scale":[1,1,1]},"training_patients":[],"validation_patients":[],"evaluation_patients":[]}
            (base/"results/comparison/run.json").write_text(json.dumps(run))
            chronology=root/"info/lumiere_task_manifest.json"; chronology.parent.mkdir(); chronology.write_text(json.dumps({"tasks":{"forecast":[]}}))
            (root/"data/lumiere_preprocessed").mkdir(parents=True)
            expected=module.sha256(state/"best.pt")
            provenance={"prepared_sha256":run["prepared_sha256"],"source_data_hashes":{"manifest":"WRONG"},"source_hashes":{}}
            features={"checkpoint":expected,"provenance":provenance,"normalization":run["normalization"],"contract":"mri-only-state-observed-change-v1","features":{}}
            checkpoint={"provenance":provenance,"normalization":run["normalization"],"contract":"mri-only-state-observed-change-v1","state_dict":{}}
            with mock.patch.object(module,"ROOT",root), mock.patch.object(module,"BASE",base), mock.patch.object(module,"EXPECTED_CHECKPOINT",expected), mock.patch.object(module.torch,"load",side_effect=lambda path,**kw:features if Path(path).name=="features.pt" else checkpoint):
                with self.assertRaisesRegex(ValueError,"source-data provenance disagrees"):
                    module.load_inputs()

    def test_provenance_inventory_hash_accepts_raw_or_canonical_only(self):
        obj={"x":1}; raw=json.dumps(obj,indent=2).encode()
        self.assertTrue(module._inventory_digest_matches(hashlib.sha256(raw).hexdigest(),obj,raw))
        self.assertTrue(module._inventory_digest_matches(module.json_hash(obj),obj,raw))
        self.assertFalse(module._inventory_digest_matches("0"*64,obj,raw))

    def test_budget_guard_rejects_expired_deadlines(self):
        with self.assertRaises(TimeoutError): module._budget_guard(0, 1e8)
        with self.assertRaises(TimeoutError): module._budget_guard(1e8, 0)

    def test_preflight_ledger_transitions_to_run_and_stale_digest_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(module,"OUT",Path(td)):
                ledger={"version":3,"status":"preflight_complete","mode":"preflight","started_wall_unix":module.time.time(),"preflight_cpu_seconds":1.,"run_cpu_seconds":0.}
                (Path(td)/"budget_ledger.json").write_text(json.dumps(ledger))
                path,updated=module._ledger_start("run")
                self.assertEqual(updated["mode"],"run")
                self.assertEqual(updated["status"],"running")
                with self.assertRaises(FileExistsError): module._ledger_start("run")
                path.write_text(json.dumps(ledger))
                data={"files":{},"objects":{"run":{}},"rows":[],"predicted":{}}
                with mock.patch.object(module,"load_inputs",return_value=data), mock.patch.object(module,"make_contract",return_value={"sentinel":1}):
                    with self.assertRaisesRegex(ValueError,"contract digest mismatch"):
                        module.run("stale")
                saved=json.loads(path.read_text())
                self.assertEqual(saved["status"],"run_failed")

    def test_cpu_signal_records_failure_and_prevents_restart(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            ledger={"status":"preflight_complete","mode":"preflight","started_wall_unix":module.time.time(),"preflight_cpu_seconds":1.,"run_cpu_seconds":0.}
            (out/"budget_ledger.json").write_text(json.dumps(ledger))
            def exhaust_cpu():
                module.signal.raise_signal(module.signal.SIGXCPU)
            # Control elapsed CPU time rather than depend on the host clock's granularity.
            ticks = iter(1000. + .01 * i for i in range(100))
            with mock.patch.object(module,"OUT",out), mock.patch.object(module,"load_inputs",side_effect=exhaust_cpu), mock.patch.object(module.time,"process_time",side_effect=ticks):
                with self.assertRaisesRegex(TimeoutError,"CPU budget exhausted"):
                    module.run("not_reached")
                saved=json.loads((out/"budget_ledger.json").read_text())
                self.assertEqual(saved["status"],"run_failed")
                self.assertGreater(saved["run_cpu_seconds"],0.)
                with self.assertRaises(FileExistsError): module._ledger_start("run")

    def test_mocked_run_saves_replayable_arrays_and_both_paired_contrasts(self):
        rows=[]; predicted={}
        groups={"source52":"s","validation13":"v","unseen26":"q"}
        for group,patient in groups.items():
            for n in range(2):
                cutoff=f"{patient}{n}c"; target=f"{patient}{n}t"; rowid=f"{patient}{n}"
                vals={"necrotic_non_enhancing":float(n+1),"enhancing":float(n+3),"edema_flair":float(n+7)}
                c={"visit_id":cutoff}; t={"visit_id":target,"values_mm3":vals}
                rows.append({"row_id":rowid,"patient_id":patient,"cutoff_visit_id":cutoff,"target_visit_id":target,"history_visit_ids":[cutoff],"primary_history":n==1,"cutoff":c,"target":t})
                predicted[cutoff]=np.array([n+.5,n+2.5,n+6.5])
        splits={k:[v] for k,v in groups.items()}; manifest=[{"row_id":r["row_id"],"patient_id":r["patient_id"],"split":k} for k,v in splits.items() for r in rows if r["patient_id"] in v]
        contract={"splits":splits,"eligible_row_manifest":manifest,"eligible_row_manifest_sha256":module.json_hash(manifest),"exclusion_counts":{},"input_hashes":{}}
        data={"rows":rows,"predicted":predicted,"files":{},"objects":{"run":{}},"row_exclusions":{}}
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(module,"OUT",Path(td)), mock.patch.object(module,"load_inputs",return_value=data), mock.patch.object(module,"make_contract",return_value=contract), mock.patch.object(module,"_ledger_start",side_effect=lambda mode:(Path(td)/"budget.json",{"status":"started","mode":mode,"preflight_cpu_seconds":0,"run_cpu_seconds":0,"started_wall_unix":module.time.time()})):
                (Path(td)/"budget.json").write_text("{}")
                result=module.run(module.json_hash(contract))
                self.assertIn("paired_common_k1",result); self.assertIn("paired_validation_selected",result)
                with np.load(Path(td)/"run/private_predictions_scores.npz",allow_pickle=False) as arrays:
                    np.testing.assert_allclose(arrays["assessment_truth_log1p"],np.log1p([[2.,4.,8.]]))
                    np.testing.assert_allclose(arrays["source_changes"],[[np.log1p(1)-.5,np.log1p(3)-2.5,np.log1p(7)-6.5],[np.log1p(2)-1.5,np.log1p(4)-3.5,np.log1p(8)-7.5]])
                    self.assertEqual(len(arrays["validation_grid_discrete"]),3)
                    np.testing.assert_allclose(arrays["source_patient_weights"],[.5,.5])
                    self.assertEqual(arrays["single_visit_assessment_ids"].tolist(),["q0"])
                    self.assertEqual(arrays["single_visit_assessment_truth_log1p"].shape,(1,3))
                saved=json.loads((Path(td)/"run/aggregate.json").read_text())
                self.assertFalse(saved["assessment_winner_selection"])


if __name__ == "__main__": unittest.main()
