import dataclasses
import unittest

from src.harness.analysis.spatial_dynamics_protocol import (
    ChronologyVisit, FullChronology, build_history_rows,
)


def visit(name, *, usable=True, gap=10.0, target=None, patient="p"):
    return ChronologyVisit(name, patient, usable, gap, (True, False, True, False), target)


class SpatialDynamicsProtocolTests(unittest.TestCase):
    def test_adjacency_and_unusable_visit_are_not_bridged(self):
        rows = build_history_rows(FullChronology((visit("a", target="A"),
            visit("b", usable=False), visit("c", target="C")), True))
        self.assertEqual(rows, ())

    def test_each_row_targets_immediate_next_visit(self):
        rows = build_history_rows(FullChronology((visit("a", target="A"), visit("b", target="B"),
                                                   visit("c", target="C")), True))
        self.assertEqual([(r.cutoff_visit_id, r.target_visit_id) for r in rows], [("a", "b"), ("b", "c")])
        self.assertFalse(rows[0].primary_history)
        self.assertTrue(rows[1].primary_history)

    def test_missing_or_nonpositive_historical_gap_invalidates_row(self):
        for bad in (None, 0, -1, float("nan"), float("inf")):
            rows = build_history_rows(FullChronology((visit("a"), visit("b", gap=bad), visit("c", target="C")), True))
            self.assertEqual(len(rows), 0)

    def test_window_truncation_and_first_gap_reset(self):
        rows = build_history_rows(FullChronology(tuple(visit(str(i), target=i, gap=7) for i in range(6)), True))
        last = rows[-1]
        self.assertEqual(len(last.history.visits), 4)
        self.assertEqual(last.history.history_mask, (True, True, True, True))
        self.assertEqual(last.history.visits[0].gap_log1p_days, 0.0)
        self.assertAlmostEqual(last.history.visits[1].gap_log1p_days, __import__("math").log1p(7))

    def test_padding_and_single_visit_classification(self):
        rows = build_history_rows(FullChronology((visit("a"), visit("b", target="B")), True))
        self.assertEqual(rows[0].history.history_mask, (False, False, False, True))
        self.assertFalse(rows[0].primary_history)

    def test_future_gap_and_availability_do_not_change_fixed_cutoff_history(self):
        visits = (visit("a"), visit("b", target="B"), visit("c", target="C"))
        original = build_history_rows(FullChronology(visits, True))[0]
        altered = (visits[0], dataclasses.replace(visits[1], gap_from_previous_days=999,
                    availability=(False, True, False, True)), visits[2])
        changed = build_history_rows(FullChronology(altered, True))[0]
        self.assertEqual(original.history, changed.history)

    def test_history_restarts_after_unusable_and_patient_boundary(self):
        rows = build_history_rows(FullChronology((
            visit("a", target="A"), visit("b", usable=False), visit("c", target="C"),
            visit("d", target="D"), visit("e", target="E", patient="q"),
            visit("f", target="F", patient="q")), True))
        self.assertEqual([(r.cutoff_visit_id, r.target_visit_id) for r in rows],
                         [("c", "d"), ("e", "f")])
        self.assertEqual(rows[0].history_visit_ids, ("c",))
        self.assertEqual(rows[1].history_visit_ids, ("e",))

    def test_target_changes_do_not_change_predictor_input(self):
        chronology = FullChronology((visit("a"), visit("b", target={"volume": 1})), True)
        row = build_history_rows(chronology)[0]
        changed = build_history_rows(dataclasses.replace(chronology, visits=(
            chronology.visits[0], dataclasses.replace(chronology.visits[1], target={"volume": 99}))))[0]
        self.assertEqual(row.history, changed.history)

    def test_incomplete_full_chronology_evidence_rejected(self):
        with self.assertRaisesRegex(ValueError, "full original"):
            build_history_rows(FullChronology((visit("a"), visit("b", target="B")), False))

    def test_history_audit_ids_are_exact_ordered_suffix(self):
        rows = build_history_rows(FullChronology(tuple(visit(str(i), target=i) for i in range(4)), True))
        self.assertEqual(rows[-1].history_visit_ids, ("0", "1", "2"))

    def test_interleaved_patient_blocks_and_invalid_locked_window_rejected(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            build_history_rows(FullChronology((visit("a", patient="p"), visit("b", patient="q"),
                                                visit("c", patient="p")), True))
        for value in (0, 5, True):
            with self.assertRaisesRegex(ValueError, "1 through 4"):
                build_history_rows(FullChronology((), True), max_visits=value)

    def test_ids_and_target_fields_are_not_history_tensor_fields(self):
        row = build_history_rows(FullChronology((visit("a"), visit("b", target="B")), True))[0]
        self.assertEqual(set(row.history.visits[0].__dataclass_fields__), {"availability", "gap_log1p_days"})
        self.assertNotIn("patient_id", row.history.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
