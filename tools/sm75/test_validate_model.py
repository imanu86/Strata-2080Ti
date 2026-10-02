"""Fail-closed checks for the paired model validation harness; CPU only."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import validate_model as V


class ValidationTests(unittest.TestCase):
    def test_dump_rejects_truncation_and_extra_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'logits.bin'
            header = np.array([2, 3], dtype='<i4').tobytes()
            p.write_bytes(header + np.zeros((2,2), dtype='<f4').tobytes())
            with self.assertRaises(RuntimeError): V.check_dump(p,3,2)
            p.write_bytes(header + np.zeros((4,2), dtype='<f4').tobytes())
            with self.assertRaises(RuntimeError): V.check_dump(p,3,2)
            p.write_bytes(header + np.zeros((3,2), dtype='<f4').tobytes())
            self.assertEqual(V.check_dump(p,3,2).shape, (3,2))

    def test_teacher_targets_and_kl_against_analytic_result(self):
        a = np.log(np.array([[.8,.2],[.6,.4],[.5,.5]]))
        b = np.log(np.array([[.4,.6],[.3,.7],[.5,.5]]))
        stats = V.paired_stats(a,b,[0,1,0])
        self.assertEqual(stats['positions'],2)
        expected_kl = (.8*np.log(2)+.2*np.log(1/3)+.6*np.log(2)+.4*np.log(4/7))/2
        self.assertAlmostEqual(stats['kl_mean'],expected_kl)
        self.assertAlmostEqual(stats['ppl_reference'],1/np.sqrt(.2*.6))
        self.assertAlmostEqual(stats['nll_delta_mean'],(-np.log(.6/.2)-np.log(.3/.6))/2)

    def test_infinite_and_unpaired_data_are_rejected(self):
        a = np.zeros((3,2)); b = a.copy(); b[1,0] = np.nan
        with self.assertRaises(RuntimeError): V.paired_stats(a,b,[0,1,0])
        with self.assertRaises(RuntimeError): V.paired_stats(a,a[:2],[0,1,0])

    def test_complete_answer_requires_stop_and_oracle(self):
        self.assertTrue(V.evaluate_answer('{"result":[7,2,1,9]}','stop',{'result':[7,2,1,9]}))
        self.assertFalse(V.evaluate_answer('{"result":[7,2,1,9]}','length',{'result':[7,2,1,9]}))
        self.assertFalse(V.evaluate_answer('{"result":[1,2,7,9]}','stop',{'result':[7,2,1,9]}))
        self.assertFalse(V.evaluate_answer('<think>partial','stop',{'result':[7,2,1,9]}))

    def test_gate_rejects_degradation_even_with_matching_top1(self):
        baseline = {'nll_deltas':[0.0]*128,'kl_median':0.,'kl_p99':0.,'top1_agreement_pct':100.}
        candidate = {**baseline,'nll_deltas':[.1]*128}
        self.assertFalse(V.regression_gate(candidate,baseline)['pass'])
        self.assertTrue(V.regression_gate(baseline,baseline)['pass'])
        candidate = {**baseline,'kl_p99':.2}
        self.assertFalse(V.regression_gate(candidate,baseline)['pass'])


if __name__ == '__main__':
    unittest.main()
