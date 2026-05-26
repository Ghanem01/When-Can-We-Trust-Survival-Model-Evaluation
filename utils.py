# utils.py (version robuste)
import numpy as np
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

def compute_ibs(surv_df, T_train, E_train, T_test, E_test, time_grid, verbose=False):
    """
    Compute the Integrated Brier Score (IBS) for survival models.
    """
    try:
        T_train = np.asarray(T_train, dtype=float)
        E_train = np.asarray(E_train, dtype=bool)
        T_test = np.asarray(T_test, dtype=float)
        E_test = np.asarray(E_test, dtype=bool)

        min_train, max_train = T_train.min(), T_train.max()
        test_mask = (T_test > min_train) & (T_test < max_train)
        if np.sum(test_mask) == 0:
            if verbose:
                print(" No test patients within train time range — IBS skipped.")
            return np.nan

        min_test, max_test = T_test[test_mask].min(), T_test[test_mask].max()
        valid_times = time_grid[(time_grid > min_test) & (time_grid < max_test)]
        if len(valid_times) == 0:
            if verbose:
                print(" No overlapping times between train/test — IBS skipped.")
            return np.nan

        surv_probs_filtered = surv_df.loc[valid_times].values.T[test_mask, :]
        surv_train = Surv.from_arrays(event=E_train, time=T_train)
        surv_test = Surv.from_arrays(event=E_test[test_mask], time=T_test[test_mask])

        ibs = integrated_brier_score(surv_train, surv_test, surv_probs_filtered, valid_times)
        return float(ibs)
    except Exception as e:
        if verbose:
            print(f" IBS computation failed: {e}")
        return np.nan


def concordance_at_time(t_ref, times, events, surv_probs_at_t):
    """
    Compute concordance index at a fixed time t_ref.
    """
    n_concordant = n_discordant = n_tied = 0
    n = len(times)
    for i in range(n):
        for j in range(i+1, n):
            if events[i] and times[i] <= t_ref < times[j]:
                n_concordant += surv_probs_at_t[i] < surv_probs_at_t[j]
                n_discordant += surv_probs_at_t[i] > surv_probs_at_t[j]
                n_tied += surv_probs_at_t[i] == surv_probs_at_t[j]
            elif events[j] and times[j] <= t_ref < times[i]:
                n_concordant += surv_probs_at_t[j] < surv_probs_at_t[i]
                n_discordant += surv_probs_at_t[j] > surv_probs_at_t[i]
                n_tied += surv_probs_at_t[j] == surv_probs_at_t[i]
    total = n_concordant + n_discordant + n_tied
    return np.nan if total == 0 else (n_concordant + 0.5 * n_tied) / total


def get_columns_to_normalize(df, exclude_cols=['time', 'event', 'true_time']):
    """
    Return numeric columns that are not already normalized in [0,1].
    """
    candidate_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_cols = df[candidate_cols].select_dtypes(include=['number']).columns.tolist()
    cols_to_norm = []
    for col in numeric_cols:
        min_val, max_val = df[col].min(), df[col].max()
        if not (0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0):
            cols_to_norm.append(col)
    return cols_to_norm
