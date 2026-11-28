import unittest

import pandas as pd

from src.anomaly_comparison import add_anomaly_detectors, add_mean_flags, make_vote_flags
from src.forecast import DETECTOR_FEATURES, detect_anomalies


class VotingTests(unittest.TestCase):
    def test_all_thresholds_for_pairs_and_triples(self):
        flags = pd.DataFrame({'a': [1, 1, 0, 0], 'b': [1, 0, 0, 0], 'c': [1, 0, 1, 0]}, dtype=bool)
        rules = make_vote_flags(flags, {'A': 'a', 'B': 'b', 'C': 'c'})
        self.assertEqual(rules.shape[1], 9)
        self.assertEqual(rules['1 из 2 · A + B'].tolist(), [True, True, False, False])
        self.assertEqual(rules['2 из 2 · A + B'].tolist(), [True, False, False, False])
        self.assertEqual(rules['2 из 3 · A + B + C'].tolist(), [True, False, False, False])


class DetectorTests(unittest.TestCase):
    def test_group_means_use_training_and_fall_back_to_global(self):
        monday, tuesday, later_monday, wednesday = pd.to_datetime(
            ['2017-01-02', '2017-01-03', '2017-01-09', '2017-01-11']
        )
        data = pd.DataFrame({
            'arrival': [monday] * 3 + [tuesday] * 3 + [later_monday, wednesday],
            'LeadTime': [9, 10, 11, 29, 30, 31, 15, 15],
            'active_count': 8, 'over_median_lead': 0,
        })
        scored = add_mean_flags(data, [monday, tuesday], min_group_size=3)
        self.assertEqual(scored['mean_global'].iloc[-2:].tolist(), [False, False])
        self.assertEqual(scored['mean_weekday'].iloc[-2:].tolist(), [True, False])

    def test_future_features_do_not_define_dbscan_core(self):
        train, holdout = pd.to_datetime(['2017-01-01', '2017-01-05'])
        data = pd.DataFrame({
            'arrival': [train] * 48 + [holdout, holdout],
            'LeadTime': [10] * 24 + [20] * 24 + [10, 1000],
            'active_count': 8, 'over_median_lead': 0,
        })
        scored, _, _ = detect_anomalies(data, [train])
        first = add_anomaly_detectors(scored, [train])
        changed = scored.copy()
        changed.loc[changed['arrival'].eq(holdout), DETECTOR_FEATURES] = 10_000
        second = add_anomaly_detectors(changed, [train])
        self.assertEqual(first['dbscan'].iloc[-2:].tolist(), [False, True])
        self.assertEqual(second['dbscan'].iloc[-2:].tolist(), [True, True])
        pd.testing.assert_series_equal(first['dbscan'].iloc[:-2], second['dbscan'].iloc[:-2])
