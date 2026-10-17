"""R36 fixed splits, locked artifacts, and real CLI discovery."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.data.representation import CONTRACT as PREPARED_CONTRACT, COMPARTMENTS, digest, json_hash, save_json
from src.harness.analysis.delta_evaluation import paired_patients, seed_mean, companion_tables, scalar_readouts
from src.harness.analysis.delta_protocol import (ARMS, NUISANCES, load_contract, lock_contract,
                                                validate_metadata, verify_tensor_manifest)


def fixture():
    train = [f"train{i}" for i in range(52)]
    valid = [f"valid{i}" for i in range(13)]
    query = [f"query{i}" for i in range(26)]
    visits, pairs = [], []
    for patients, total, pair_total in ((train + valid, 450, 379), (query, 149, 117)):
        for i, patient in enumerate(patients):
            count = total // len(patients) + (i < total % len(patients))
            records = [{"visit_id": f"{patient}-{j}", "patient_id": patient, "pregrid": {"shape": [96]*3},
                        "tensor_file": f"{patient}-{j}.npz"} for j in range(count)]
            visits.extend(records)
            for a, b in zip(records, records[1:]):
                if patients is query and i >= 23:
                    continue
                pairs.append({"row_id": f"{a['visit_id']}->{b['visit_id']}", "patient_id": patient,
                              "source_visit_id": a["visit_id"], "target_visit_id": b["visit_id"],
                              "usable": True, "ordering_verified": True})
        current = [p for p in pairs if p["patient_id"] in patients]
        for row in current[pair_total:]:
            pairs.remove(row)
        current = current[:pair_total]
        while len(current) < pair_total:
            patient = patients[len(current) % (23 if patients is query else len(patients))]
            records = [v for v in visits if v["patient_id"] == patient]
            row = {"row_id": f"extra-{patient}-{len(current)}", "patient_id": patient,
                   "source_visit_id": records[0]["visit_id"], "target_visit_id": records[-1]["visit_id"],
                   "usable": True, "ordering_verified": True}
            current.append(row); pairs.append(row)
    metadata = {"contract": PREPARED_CONTRACT, "cohort": "LUMIERE", "limit": None,
                "geometry_contract": "affine-aligned-lesion-v1", "visits": visits, "pairs": pairs,
                "protocol": {"encoder_train": train + valid, "encoder_unseen": query}}
    run = {"training_patients": train, "validation_patients": valid, "evaluation_patients": query}
    return metadata, run


class ProtocolTests(unittest.TestCase):
    def test_fixed_coverage_and_disjoint_patients(self):
        metadata, run = fixture()
        validate_metadata(metadata, run)
        bad = copy.deepcopy(run)
        bad["training_patients"][0] = bad["evaluation_patients"][0]
        with self.assertRaises(ValueError):
            validate_metadata(metadata, bad)
        bad = copy.deepcopy(metadata)
        bad["pairs"].pop()
        with self.assertRaises(ValueError):
            validate_metadata(bad, run)

    def test_cross_patient_and_bad_geometry_fail(self):
        metadata, run = fixture()
        metadata["pairs"][0]["target_visit_id"] = "query0-0"
        with self.assertRaises(ValueError):
            validate_metadata(metadata, run)

    def test_immutable_contract_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r36_contract.json"
            contract = {"contract": "r36-signed-feature-difference-v1", "epochs": 20}
            lock_contract(path, contract)
            self.assertEqual(load_contract(path), contract)
            lock_contract(path, contract)
            with self.assertRaises(ValueError):
                lock_contract(path, {**contract, "epochs": 10})
            save_json(path, {**contract, "epochs": 10})
            with self.assertRaises(ValueError):
                load_contract(path)

    def test_tensor_bytes_not_filename_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "visit.npz").write_bytes(b"MRI bytes")
            save_json(root / "prepared.json", {"visits": [{"tensor_file": "visit.npz"}]})
            manifest = {"prepared_sha256": digest(root / "prepared.json"), "tensors": {"visit.npz": digest(root / "visit.npz")}}
            verify_tensor_manifest(root, manifest)
            (root / "visit.npz").write_bytes(b"new bytes")
            with self.assertRaises(ValueError):
                verify_tensor_manifest(root, manifest)

    def test_matrix_changes_only_objectives(self):
        self.assertEqual(ARMS["B"], {"change": "grid", "nuisance": False})
        self.assertEqual(ARMS["C"], {"change": "balanced", "nuisance": False})
        self.assertEqual(ARMS["D"], {"change": "balanced", "nuisance": True})
        self.assertEqual(NUISANCES, ("identity", "noise", "intensity", "bias"))

    def test_seed_error_mean_and_patient_bootstrap(self):
        values = [{"p": 0.1, "q": 0.3}, {"p": 0.2, "q": 0.4}, {"p": 0.3, "q": 0.5}]
        result = seed_mean(values)
        self.assertAlmostEqual(result["p"], 0.2)
        self.assertAlmostEqual(result["q"], 0.4)
        interval = paired_patients({"p": 0.2, "q": 0.4}, {"p": 0.3, "q": 0.5})
        self.assertAlmostEqual(interval["difference"], -0.1)
        np.testing.assert_allclose(interval["family_adjusted_ci"], [-0.1, -0.1])
        with self.assertRaises(ValueError):
            seed_mean(values[:2])
        with self.assertRaises(ValueError):
            paired_patients({"p": 1}, {"q": 1})

    def test_empty_compartment_errors_are_not_hidden(self):
        states = [{"patient_id": "p", "truth": [0, 2, 3], "prediction": [7, 2, 3]},
                  {"patient_id": "p", "truth": [1, 2, 3], "prediction": [1, 2, 3]}]
        method = {"balanced": [0.5, 0.5, 0.5], "changed_count": [1, 1, 1]}
        row = {"patient_id": "p", "source_empty": [True, False, False], "target_empty": [False]*3,
               "volume_truth": [1, 0, 0], "volume_independent": [1, 0, 0],
               **{key: method for key in ("frozen_probe", "target_only", "independent_state", "zero")}}
        tables = companion_tables([row], states, [[1, 2, 3]]*3)
        self.assertEqual(tables["state_empty_nonempty"]["necrotic_non_enhancing"]["empty"]["patient_uniform_mae"], 7)
        self.assertEqual(tables["local_strata"]["necrotic_non_enhancing"]["appearance"]["frozen_probe"]["n_rows"], 1)
        self.assertIsNone(tables["state_empty_nonempty"]["enhancing"]["empty"]["patient_uniform_mae"])

    def test_scalar_readout_never_fits_query_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_patients = [f"train{i}" for i in range(52)]
            contract = {"contract": "r36-signed-feature-difference-v1", "training_patients": train_patients,
                        "evaluation_patients": ["query"], "prepared_sha256": "pending"}
            for name, patients, values in (("source", train_patients, [2., 3., 4.]), ("query", ["query"], [5., 6., 7.])):
                folder = root / name; folder.mkdir()
                visits, pairs, features, hashes = [], [], {}, {}
                for patient in patients:
                    for visit in range(2):
                        identifier = f"{patient}-{visit}"
                        filename = identifier + ".npz"
                        (folder / filename).write_bytes(identifier.encode())
                        hashes[filename] = digest(folder / filename)
                        visits.append({"visit_id": identifier, "patient_id": patient, "tensor_file": filename,
                                       "values_mm3": {c: float(np.expm1(v)) for c, v in zip(COMPARTMENTS, values)}})
                        features[identifier] = np.zeros(128)
                    pairs.append({"row_id": patient, "patient_id": patient, "source_visit_id": f"{patient}-0", "target_visit_id": f"{patient}-1"})
                save_json(folder / "prepared.json", {"cohort": "LUMIERE", "visits": visits, "pairs": pairs})
                if name == "source":
                    contract["prepared_sha256"] = digest(folder / "prepared.json")
                torch.save({"contract": contract["contract"], "features": features,
                            "provenance": {"checkpoint_sha256": "same encoder", "tensor_hashes": hashes,
                                           "prepared_sha256": digest(folder / "prepared.json")}}, root / f"{name}.pt")
            lock_contract(root / "r36_contract.json", contract)
            for name in ("source", "query"):
                artifact = torch.load(root / f"{name}.pt", weights_only=False)
                artifact["provenance"]["contract_sha256"] = json_hash(contract)
                torch.save(artifact, root / f"{name}.pt")
                (root / f"{name}.sha256").write_text(digest(root / f"{name}.pt"))
            result = scalar_readouts(str(root / "source.pt"), str(root / "query.pt"), str(root / "source"),
                                     str(root / "query"), str(root / "r36_contract.json"), str(root / "first.json"))
            self.assertAlmostEqual(result["scores"]["state"]["patient_uniform_mae"], 3.)
            with np.load(root / "first.predictions.npz") as data:
                np.testing.assert_allclose(data["state_prediction"], [[2, 3, 4], [2, 3, 4]])
                np.testing.assert_allclose(data["delta_prediction"], [[0, 0, 0]])
            metadata = json.loads((root / "query/prepared.json").read_text())
            for visit in metadata["visits"]:
                visit["values_mm3"] = {c: float(np.expm1(v)) for c, v in zip(COMPARTMENTS, [15, 16, 17])}
            save_json(root / "query/prepared.json", metadata)
            artifact = torch.load(root / "query.pt", weights_only=False)
            artifact["provenance"]["prepared_sha256"] = digest(root / "query/prepared.json")
            torch.save(artifact, root / "query.pt")
            (root / "query.sha256").write_text(digest(root / "query.pt"))
            second = scalar_readouts(str(root / "source.pt"), str(root / "query.pt"), str(root / "source"),
                                     str(root / "query"), str(root / "r36_contract.json"), str(root / "second.json"))
            self.assertAlmostEqual(second["scores"]["state"]["patient_uniform_mae"], 13.)
            with np.load(root / "second.predictions.npz") as data:
                np.testing.assert_allclose(data["state_prediction"], [[2, 3, 4], [2, 3, 4]])

    def test_real_cli_exposes_commands_and_rejects_forecast(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "scripts/harness.py", "delta-representation", "--help"],
                                cwd=root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("audit", result.stdout)
        self.assertIn("transfer", result.stdout)
        bad = subprocess.run([sys.executable, "scripts/harness.py", "delta-representation", "forecast"],
                             cwd=root, text=True, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("invalid choice", bad.stderr)


if __name__ == "__main__":
    unittest.main()
