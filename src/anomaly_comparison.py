from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity, LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from src.forecast import DETECTOR_FEATURES


def add_anomaly_detectors(scored, train_dates, seed=42):
    """Add novelty scores and IQR-calibrated flags using historical bookings only."""
    train_mask = scored['arrival'].isin(train_dates).to_numpy()
    train_size = int(train_mask.sum())
    if train_size < 2:
        raise ValueError('At least two historical bookings are required')

    features = scored[DETECTOR_FEATURES]
    scaler = RobustScaler().fit(features.loc[train_mask])
    scaled = scaler.transform(features)
    training = scaled[train_mask]
    holdout = scaled[~train_mask]
    neighbors = min(20, train_size - 1)
    flag_rate = scored.loc[train_mask, 'iqr'].mean()

    knn = NearestNeighbors(n_neighbors=neighbors, algorithm='kd_tree').fit(training)
    lof = LocalOutlierFactor(n_neighbors=neighbors, algorithm='kd_tree', novelty=True).fit(training)
    knn_scores = np.empty(len(scored))
    lof_scores = np.empty(len(scored))
    knn_scores[train_mask] = knn.kneighbors()[0].mean(axis=1)
    lof_scores[train_mask] = -lof.negative_outlier_factor_
    if len(holdout):
        knn_scores[~train_mask] = knn.kneighbors(holdout)[0].mean(axis=1)
        lof_scores[~train_mask] = -lof.score_samples(holdout)

    rng = np.random.default_rng(seed)
    reference_indices = rng.choice(train_size, size=min(train_size, 8_000), replace=False)
    reference = training[reference_indices]
    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=max(1 / train_size, flag_rate))
    svm.fit(reference)
    scores = {'knn': knn_scores, 'lof': lof_scores, 'ocsvm': -svm.score_samples(scaled)}

    # Map scores to percentiles of the training distribution before averaging.
    ensemble_inputs = [scored['forest_score'].to_numpy(), *scores.values()]
    scores['ensemble'] = np.mean([
        np.searchsorted(np.sort(values[train_mask]), values, side='right') / train_size
        for values in ensemble_inputs
    ], axis=0)

    additional_columns = {}
    for name, values in scores.items():
        threshold = np.quantile(values[train_mask], 1 - flag_rate)
        additional_columns[f'{name}_score'] = values
        additional_columns[name] = values > threshold
    # DBSCAN counts the observation itself in min_samples.
    dbscan_neighbors = min(neighbors + 1, len(reference))
    radius_model = NearestNeighbors(n_neighbors=dbscan_neighbors, algorithm='kd_tree').fit(reference)
    distances = radius_model.kneighbors(reference)[0][:, -1]
    eps = max(1e-8, np.quantile(distances, 1 - flag_rate))
    dbscan = DBSCAN(eps=eps, min_samples=dbscan_neighbors, algorithm='kd_tree').fit(reference)
    core = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(dbscan.components_)
    additional_columns['dbscan'] = core.kneighbors(scaled)[0][:, 0] > eps
    return scored.assign(**additional_columns)


def add_combined_flags(scored, train_dates, limits, seed=42):
    """Compare Boolean combinations with a joint statistical density estimate."""
    train_mask = scored['arrival'].isin(train_dates)
    features = scored[DETECTOR_FEATURES]
    outside = features.lt(limits['Нижняя граница']) | features.gt(limits['Верхняя граница'])
    votes = scored[['forest', 'knn', 'lof', 'ocsvm']].sum(axis=1)

    scaler = RobustScaler().fit(features.loc[train_mask])
    scaled = scaler.transform(features)
    training = scaled[train_mask.to_numpy()]
    rng = np.random.default_rng(seed)
    reference = rng.choice(len(training), size=min(4_000, len(training)), replace=False)
    kde = KernelDensity(bandwidth=0.25, rtol=1e-4).fit(training[reference])
    kde_score = -kde.score_samples(scaled)
    threshold = np.quantile(kde_score[train_mask], 1 - scored.loc[train_mask, 'iqr'].mean())

    return scored.assign(
        iqr_two=outside.sum(axis=1).ge(2),
        iqr_or_forest=scored['iqr'] | scored['forest'],
        iqr_and_forest=scored['iqr'] & scored['forest'],
        ml_majority=votes.ge(3),
        kde_score=kde_score,
        kde=kde_score > threshold,
    )


def add_mean_flags(scored, train_dates, min_group_size=30):
    """Flag deviations from historical means overall and by arrival weekday."""
    train = scored['arrival'].isin(train_dates)
    features = scored[DETECTOR_FEATURES]
    history = features.loc[train]
    mean, std = history.mean(), history.std(ddof=0)
    weekday = scored['arrival'].dt.day_of_week
    grouped = history.groupby(weekday.loc[train])
    enough_rows = grouped.size().ge(min_group_size)
    group_mean = grouped.mean().loc[enough_rows]
    group_std = grouped.std(ddof=0).loc[enough_rows]
    usable = group_std.gt(0)
    centers = group_mean.where(usable).reindex(weekday).set_axis(scored.index).fillna(mean)
    scales = group_std.where(usable).reindex(weekday).set_axis(scored.index).fillna(std)

    return scored.assign(
        mean_global=features.sub(mean).abs().gt(3 * std).any(axis=1),
        mean_weekday=(features - centers).abs().gt(3 * scales).any(axis=1),
    )


def make_vote_flags(scored, methods):
    """Build every pair and triple of independent detector flags."""
    rules = {}
    for size in (2, 3):
        for names in combinations(methods, size):
            votes = scored[[methods[name] for name in names]].sum(axis=1)
            description = ' + '.join(names)
            for minimum in range(1, size + 1):
                rules[f'{minimum} из {size} · {description}'] = votes.ge(minimum)
    return pd.DataFrame(rules, index=scored.index)
