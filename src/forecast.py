import pandas as pd
from sklearn.ensemble import IsolationForest


DETECTOR_FEATURES = ['LeadTime', 'active_count', 'over_median_lead']


def make_snapshot(bookings, dates, horizon):
    """Build the end-of-day booking cohort and its daily totals."""
    cutoff = bookings['arrival'] - pd.Timedelta(days=horizon)
    opened = bookings['created'].le(cutoff)
    ended = bookings['canceled'] & bookings['status_date'].le(cutoff)
    active = bookings.loc[opened & ~ended].copy()
    by_day = active.groupby('arrival')
    daily = by_day.agg(n=('show', 'size'), actual=('show', 'sum'),
                       ready=('ready', 'max')).reindex(dates)
    daily[['n', 'actual']] = daily[['n', 'actual']].fillna(0).astype(int)
    daily['ready'] = daily['ready'].fillna(pd.Series(dates, index=dates))
    daily['cutoff'] = dates - pd.Timedelta(days=horizon)

    active['active_count'] = active['arrival'].map(daily['n'])
    active['over_median_lead'] = active['LeadTime'] - by_day['LeadTime'].transform('median')

    assert active['LeadTime'].ge(horizon).all()
    assert daily['actual'].le(daily['n']).all()
    active_start = bookings['created'].lt(cutoff) & ~(
        bookings['canceled'] & bookings['status_date'].lt(cutoff)
    )
    return active, daily, int((active_start != (opened & ~ended)).sum())


def temporal_split(daily, horizon, windows):
    """Select dates whose labels were ready before the next period's first forecast."""
    split = {}
    periods = list(windows)
    for i, (name, (start, end)) in enumerate(windows.items()):
        mask = (daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))
        if i + 1 < len(periods):
            next_start = pd.Timestamp(windows[periods[i + 1]][0])
            first_cutoff = next_start - pd.Timedelta(days=horizon)
            mask &= daily['ready'].lt(first_cutoff)
        split[name] = daily.index[mask]
        assert len(split[name]) > 0
    for left, right in zip(periods, periods[1:]):
        assert daily.loc[split[left], 'ready'].max() < daily.loc[split[right], 'cutoff'].min()
    return split


def forecast_baseline(scored, train_dates, active_counts, excluded_flag=None):
    """Estimate the show rate on historical bookings and apply it to full daily counts."""
    history = scored.loc[scored['arrival'].isin(train_dates)]
    sample = history if excluded_flag is None else history.loc[~history[excluded_flag]]
    rate = sample['show'].mean()
    forecast = active_counts * rate
    assert forecast.ge(0).all() and forecast.le(active_counts).all()
    return rate, forecast, len(history) - len(sample)


def detect_anomalies(active, train_dates, seed=42):
    """Fit the IQR limits and Isolation Forest threshold on historical bookings."""
    train_mask = active['arrival'].isin(train_dates)
    all_x = active[DETECTOR_FEATURES]
    training_x = all_x.loc[train_mask]
    q1, q3 = training_x.quantile([.25, .75]).to_numpy()
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flags = ((all_x < low) | (all_x > high)).any(axis=1)
    forest = IsolationForest(n_estimators=160, max_samples=min(512, len(training_x)),
                             random_state=seed, n_jobs=1)
    forest.fit(training_x)
    scores = pd.Series(-forest.score_samples(all_x), index=active.index)
    threshold = scores.loc[train_mask].quantile(1 - flags.loc[train_mask].mean())
    scored = active.assign(iqr=flags, forest=scores.gt(threshold), forest_score=scores)
    limits = pd.DataFrame({'Нижняя граница': low, 'Верхняя граница': high}, index=DETECTOR_FEATURES)
    return scored, limits, threshold
