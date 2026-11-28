import unittest

import pandas as pd

from src.forecast import detect_anomalies, forecast_baseline, forecast_by_flag, make_snapshot, temporal_split


class SnapshotTests(unittest.TestCase):
    def test_cutoff_boundaries_for_all_horizons(self):
        arrival = pd.Timestamp('2017-05-20')
        dates = pd.date_range(arrival, periods=2, name='arrival')
        names = ['created_on_cutoff', 'canceled_on_cutoff', 'canceled_later',
                 'no_show', 'created_later']

        for horizon in (14, 7, 3):
            with self.subTest(horizon=horizon):
                day = pd.Timedelta(days=1)
                cutoff = arrival - horizon * day
                bookings = pd.DataFrame({
                    'arrival': arrival,
                    'created': [cutoff, cutoff - day, cutoff - day, cutoff - 2 * day,
                                cutoff + day],
                    'canceled': [False, True, True, False, False],
                    'status_date': [arrival + day, cutoff, cutoff + day, arrival, arrival],
                    'show': [1, 0, 0, 0, 1],
                }, index=names)
                bookings['ready'] = bookings[['arrival', 'status_date']].max(axis=1)
                bookings['LeadTime'] = (bookings['arrival'] - bookings['created']).dt.days

                active, daily, boundary_changes = make_snapshot(bookings, dates, horizon)

                self.assertEqual(active.index.tolist(), ['created_on_cutoff', 'canceled_later', 'no_show'])
                self.assertEqual(active['active_count'].tolist(), [3, 3, 3])
                self.assertEqual(active['over_median_lead'].tolist(), [-1, 0, 1])
                self.assertEqual(daily['n'].tolist(), [3, 0])
                self.assertEqual(daily['actual'].tolist(), [1, 0])
                self.assertEqual(daily.loc[arrival, 'cutoff'], cutoff)
                self.assertEqual(daily.loc[arrival, 'ready'], arrival + day)
                self.assertEqual(daily.loc[arrival + day, 'ready'], arrival + day)
                self.assertEqual(boundary_changes, 2)


class TemporalSplitTests(unittest.TestCase):
    def test_only_labels_ready_before_next_cutoff_are_eligible(self):
        dates = pd.to_datetime(['2017-01-01', '2017-01-02', '2017-01-03',
                                '2017-01-10', '2017-01-11', '2017-01-20'])
        windows = {
            'train': ('2017-01-01', '2017-01-03'),
            'validation': ('2017-01-10', '2017-01-11'),
            'test': ('2017-01-20', '2017-01-20'),
        }
        daily = pd.DataFrame({
            'ready': pd.to_datetime(['2017-01-06', '2017-01-07', '2017-01-08',
                                     '2017-01-16', '2017-01-17', '2017-01-21']),
            'cutoff': dates - pd.Timedelta(days=3),
        }, index=dates)

        split = temporal_split(daily, horizon=3, windows=windows)

        self.assertEqual({name: index.tolist() for name, index in split.items()}, {
            'train': [dates[0]], 'validation': [dates[3]], 'test': [dates[5]],
        })


class BaselineTests(unittest.TestCase):
    def setUp(self):
        train_day = pd.Timestamp('2017-01-01')
        forecast_days = pd.to_datetime(['2017-05-01', '2017-05-02'])
        self.train_dates = [train_day]
        self.active_counts = pd.Series([8, 0], index=forecast_days)
        self.scored = pd.DataFrame({
            'arrival': [train_day] * 4 + [forecast_days[0]],
            'show': [1, 1, 1, 0, 0],
            'iqr': [True, False, False, False, False],
            'forest': [False, False, False, True, False],
        })

    def test_future_outcomes_do_not_affect_forecasts(self):
        changed = self.scored.copy()
        changed.loc[4, 'show'] = 1

        for scored in (self.scored, changed):
            rate, forecast, excluded = forecast_baseline(scored, self.train_dates, self.active_counts)
            self.assertEqual(rate, 0.75)
            self.assertEqual(excluded, 0)
            pd.testing.assert_series_equal(forecast, self.active_counts.astype(float) * 0.75)
            self.assertEqual(forecast.tolist(), [6, 0])

    def test_anomaly_filters_change_only_the_history(self):
        for flag, expected_rate in (('iqr', 2 / 3), ('forest', 1.0)):
            with self.subTest(flag=flag):
                rate, forecast, excluded = forecast_baseline(
                    self.scored, self.train_dates, self.active_counts, excluded_flag=flag
                )
                self.assertEqual(rate, expected_rate)
                self.assertEqual(excluded, 1)
                pd.testing.assert_series_equal(forecast, self.active_counts * expected_rate)

    def test_flag_rates_keep_flagged_bookings_and_handle_unseen_groups(self):
        dates = self.active_counts.index
        forecast = forecast_by_flag(self.scored, self.train_dates, dates, 'iqr')
        self.assertEqual(forecast.tolist(), [2 / 3, 0])
        unseen = self.scored.assign(forest=[False, False, False, False, True])
        forecast = forecast_by_flag(unseen, self.train_dates, dates, 'forest')
        self.assertEqual(forecast.tolist(), [0.75, 0])


class AnomalyTests(unittest.TestCase):
    def test_holdout_features_do_not_change_training_fit(self):
        train_day, test_day = pd.to_datetime(['2017-01-01', '2017-05-01'])
        training_lead = [10, 11, 12, 13, 14, 15, 16, 17, 100]
        active = pd.DataFrame({
            'arrival': [train_day] * len(training_lead) + [test_day, test_day],
            'LeadTime': training_lead + [14, 14],
            'active_count': [9] * len(training_lead) + [2, 2],
            'over_median_lead': [lead - 14 for lead in training_lead] + [0, 0],
        })
        train_mask = active['arrival'].eq(train_day)
        changed = active.copy()
        changed.loc[~train_mask, ['LeadTime', 'active_count', 'over_median_lead']] = 10_000

        scored, limits, threshold = detect_anomalies(active, [train_day])
        rescored, new_limits, new_threshold = detect_anomalies(changed, [train_day])

        self.assertEqual(limits.loc['LeadTime'].tolist(), [6, 22])
        self.assertEqual(scored.loc[train_mask, 'iqr'].tolist(), [False] * 8 + [True])
        self.assertEqual(rescored.loc[~train_mask, 'iqr'].tolist(), [True, True])
        pd.testing.assert_frame_equal(limits, new_limits)
        self.assertAlmostEqual(threshold, new_threshold)
        pd.testing.assert_frame_equal(scored.loc[train_mask], rescored.loc[train_mask])
