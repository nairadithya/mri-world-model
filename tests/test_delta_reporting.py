"""Reject mismatched nuisance receipts and classify fixed R36 tradeoffs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.representation import digest, json_hash, save_json
from src.harness.analysis.delta_evaluation import aggregate_matrix


def fixture(root, *, nuisance_delta=-0.1, change_delta=0.0, state_delta=0.0):
    patients = ["p", "q"]
    contract = {"seeds": [42, 43, 44], "evaluation_patients": patients,
                "prepared_sha256": "prepared", "nuisance_training": ["identity", "noise", "intensity", "bias"],
                "held_out_nuisance": "resolution",
                "margins": {"change_noninferiority": 0.05, "state_noninferiority": 0.05},
                "pair_rows": [{"row_id": p, "patient_id": p} for p in patients]}
    manifest = {"prepared_sha256": "prepared", "tensors": {"p.npz": "p-bytes", "q.npz": "q-bytes"}}
    for arm in ("A", "B", "C", "D"):
        for seed in contract["seeds"]:
            directory = root / f"seed{seed}" / arm
            (directory / "difference").mkdir(parents=True)
            (directory / "final.pt").write_bytes(f"{seed}-{arm}-checkpoint".encode())
            (directory / "difference/probe.pt").write_bytes(f"{seed}-{arm}-probe".encode())
            provenance = {"contract_sha256": json_hash(contract), "prepared_sha256": "prepared",
                          "checkpoint_sha256": digest(directory / "final.pt"),
                          "probe_sha256": digest(directory / "difference/probe.pt"),
                          "tensor_hashes": manifest["tensors"]}
            changes = {p: 0.4 + (change_delta if arm == "D" else 0) for p in patients}
            states = {p: 0.9 + (state_delta if arm == "D" else 0) for p in patients}
            evaluation = {"status": "complete", "arm": arm, "seed": seed, "cohort": "LUMIERE",
                          "provenance": provenance, "rows": contract["pair_rows"],
                          "states": [{"patient_id": p, "visit_id": f"{p}-first"} for p in patients],
                          "state": {"per_patient_mae": states},
                          "local": {"frozen_probe": {"per_patient": changes},
                                    "target_only": {"per_patient": changes},
                                    "independent_state": {"per_patient": changes}}}
            save_json(directory / "lumiere.json", evaluation)
            rows = []
            for p in patients:
                for kind in ("identity", "noise", "intensity", "bias", "resolution"):
                    excess = 0.0 if kind == "identity" else 0.2 + (nuisance_delta if arm == "D" else 0)
                    rows.append({"patient_id": p, "visit_id": f"{p}-first", "kind": kind,
                                 "false_change": 0.1 + excess, "state_drift": 0.0, "excess_false_change": excess})
            nuisance = {"status": "complete", "arm": arm, "seed": seed, "rows": rows,
                        "provenance": {**provenance, "tensor_manifest_sha256": json_hash(manifest)},
                        "conditions": {kind: {key: sum(r[key] for r in rows if r["kind"] == kind) / 2
                                              for key in ("false_change", "state_drift", "excess_false_change")}
                                       for kind in ("identity", "noise", "intensity", "bias", "resolution")}}
            save_json(directory / "nuisance.json", nuisance)
    return contract


class ReportingTests(unittest.TestCase):
    def verdict(self, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = fixture(root, **kwargs)
            return aggregate_matrix(root, contract)["hypotheses"]["D_nuisance_tradeoff"]

    def test_verified_tradeoff(self):
        self.assertEqual(self.verdict(), "VERIFIED")

    def test_demonstrated_nuisance_harm_is_negative(self):
        self.assertEqual(self.verdict(nuisance_delta=0.1), "NOT VERIFIED")

    def test_demonstrated_change_and_state_harm_are_negative(self):
        for field in ("change_delta", "state_delta"):
            with self.subTest(field=field):
                self.assertEqual(self.verdict(**{field: 0.1}), "NOT VERIFIED")

    def test_interval_overlap_is_inconclusive(self):
        self.assertEqual(self.verdict(nuisance_delta=0.0), "INCONCLUSIVE")

    def test_missing_nuisance_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = fixture(root)
            (root / "seed43/D/nuisance.json").unlink()
            self.assertEqual(aggregate_matrix(root, contract)["hypotheses"]["D_nuisance_tradeoff"], "INCONCLUSIVE")

    def test_identity_coverage_provenance_and_metrics_rejected(self):
        cases = {
            "status": lambda r: r.update(status="running"),
            "arm": lambda r: r.update(arm="B"),
            "seed": lambda r: r.update(seed=44),
            "checkpoint": lambda r: r["provenance"].update(checkpoint_sha256="other"),
            "probe": lambda r: r["provenance"].update(probe_sha256="other"),
            "contract": lambda r: r["provenance"].update(contract_sha256="other"),
            "prepared": lambda r: r["provenance"].update(prepared_sha256="other"),
            "tensor": lambda r: r["provenance"].update(tensor_manifest_sha256="nonempty-but-wrong"),
            "missing": lambda r: r["rows"].pop(),
            "duplicate": lambda r: r["rows"].append(copy.deepcopy(r["rows"][0])),
            "wrong_visit": lambda r: r["rows"][0].update(visit_id="p-second"),
            "unexpected_patient": lambda r: r["rows"][0].update(patient_id="outsider"),
            "unexpected_kind": lambda r: r["rows"][0].update(kind="blur"),
            "false_change": lambda r: r["rows"][0].update(false_change=float("nan")),
            "drift": lambda r: r["rows"][0].update(state_drift=float("inf")),
            "excess": lambda r: r["rows"][0].update(excess_false_change=float("nan")),
            "inconsistent_excess": lambda r: r["rows"][0].update(excess_false_change=0.1),
            "summary": lambda r: r["conditions"]["noise"].update(false_change=10.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = fixture(root)
            path = root / "seed42/C/nuisance.json"
            original = json.loads(path.read_text())
            for name, mutate in cases.items():
                with self.subTest(case=name):
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        aggregate_matrix(root, contract)
            save_json(path, original)
            self.assertEqual(aggregate_matrix(root, contract)["hypotheses"]["D_nuisance_tradeoff"], "VERIFIED")

    def test_checkpoint_and_probe_bytes_are_bound(self):
        for filename in ("final.pt", "difference/probe.pt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                contract = fixture(root)
                (root / "seed42/C" / filename).write_bytes(b"replaced")
                with self.assertRaises(ValueError):
                    aggregate_matrix(root, contract)


if __name__ == "__main__":
    unittest.main()
