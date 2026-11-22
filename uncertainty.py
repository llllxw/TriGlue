"""Across-model score dispersion."""
import numpy as np
from scipy.stats import rankdata

def ensemble_summary(values):
    """Across-model dispersion; one model cannot estimate ensemble uncertainty."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError('expected finite scores shaped (triplets, models)')
    if ((values < 0) | (values > 1)).any():
        raise ValueError('scores must lie in [0, 1]')
    n = values.shape[1]
    empty = np.full(values.shape[0], np.nan)
    lo = np.quantile(values, 0.025, axis=1) if n > 1 else empty.copy()
    hi = np.quantile(values, 0.975, axis=1) if n > 1 else empty.copy()
    mean = values.mean(axis=1)
    return dict(inducibility_score_mean=mean,
                predicted_class=(mean >= 0.5).astype(int),
                prediction_confidence=np.maximum(mean, 1 - mean),
                rank=rankdata(-mean, method='min').astype(int),
                ensemble_sd=values.std(axis=1, ddof=1) if n > 1 else empty.copy(),
                ensemble_q025=lo, ensemble_q975=hi, ensemble_interval_width=hi-lo,
                n_models=n, uncertainty_status='ensemble_dispersion' if n>1 else 'single_model_not_available')
