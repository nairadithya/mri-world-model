"""Small exact checks for retrospective interpretation, run on mosaic-gpu."""
import unittest
import numpy as np
from scripts.interpret_structured_forecasts import attribution,dice_counts,interval


class InterpretationTests(unittest.TestCase):
    def test_one_patient_does_not_get_a_degenerate_confidence_interval(self):
        value=interval([{'patient':'a','gain':.1},{'patient':'a','gain':.2}],'gain')
        self.assertIsNone(value['patient_bootstrap95'])
        self.assertAlmostEqual(value['mean'],.15)

    def test_stable_error_correction_is_not_assigned_to_change(self):
        before=np.array([[0,10,0],[0,0,0],[0,0,0],[0,0,0]])
        after=np.zeros((4,3),dtype=int)
        a=attribution(before,after)
        self.assertEqual(a['unchanged_reference'],1)
        self.assertEqual(a['changed_reference'],0)

    def test_reference_change_gets_its_own_gain(self):
        before=np.array([[5,0,0],[0,0,5],[0,0,0],[0,0,0]])
        after=np.array([[5,0,0],[5,0,0],[0,0,0],[0,0,0]])
        a=attribution(before,after)
        self.assertAlmostEqual(a['changed_reference'],1/3)
        self.assertEqual(a['unchanged_reference'],0)

    def test_signed_attributions_sum_with_interacting_denominators(self):
        before=np.array([[6,10,3],[2,4,7],[1,20,0],[0,1,1]])
        after=np.array([[8,2,1],[0,5,9],[1,0,0],[1,0,0]])
        a=attribution(before,after)
        self.assertAlmostEqual(sum(a.values()),dice_counts(after.sum(0))-dice_counts(before.sum(0)))
        self.assertLess(a['changed_reference'],0)


if __name__=='__main__':unittest.main()
