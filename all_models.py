# all_models.py
# ============================================================

# ============================================================

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceError

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

import optuna
import torchtuples as tt

from pycox.evaluation import EvalSurv
from pycox.models import MTLR, DeepHitSingle

from SurvivalEVAL import PycoxEvaluator
from utils import concordance_at_time, get_columns_to_normalize, compute_ibs

warnings.filterwarnings("ignore", category=UserWarning)


def _ensure_results_dir(dataset_name: str) -> Path:
    out_dir = Path(f"./results/res_{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _write_results_csv(dataset_name: str, model_name: str, df: pd.DataFrame):
    out_dir = _ensure_results_dir(dataset_name)
    out_path = out_dir / f"{dataset_name}_{model_name}.csv"
    
    if out_path.exists():
        df_existing = pd.read_csv(out_path)
        df_all = pd.concat([df_existing, df], ignore_index=True)
        df_all.to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

def _annotate_metrics_df(df: pd.DataFrame,
                         eval_type: str,
                         censorship: int | None,
                         replication: int | None,
                         dataset_name: str,
                         optimized_on: str = "c_index") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    df["evaluation_type"] = eval_type
    df["optimized_on"] = optimized_on
    if censorship is not None:
        df["censorship_rate"] = censorship
    if replication is not None:
        df["replication"] = replication
   
    lead_cols = ["censorship_rate", "fold", "replication",
                 "evaluation_type", "optimized_on"]
    ordered = [c for c in lead_cols if c in df.columns] + [c for c in df.columns if c not in lead_cols]
    return df[ordered]

def _generate_outer_folds_with_high_censoring(df: pd.DataFrame,
                                              n_splits: int = 5,
                                              test_censoring_target: float = 0.9,
                                              seed: int = 42):
    """Version générique pour forcer une forte censure en test."""
    censored = df[df['event'] == 0].copy()
    events = df[df['event'] == 1].copy()

    test_size = len(df) // n_splits
    n_cens_per_test = int(test_censoring_target * test_size)
    n_event_per_test = test_size - n_cens_per_test

    censored = censored.sample(frac=1, random_state=seed).reset_index(drop=True)
    events = events.sample(frac=1, random_state=seed).reset_index(drop=True)

    folds = []
    for i in range(n_splits):
        test_cens = censored.iloc[i * n_cens_per_test:(i + 1) * n_cens_per_test]
        test_evnt = events.iloc[i * n_event_per_test:(i + 1) * n_event_per_test]
        test_df = pd.concat([test_cens, test_evnt])
        train_idx = df.drop(test_df.index).index.to_numpy()
        test_idx = test_df.index.to_numpy()
        folds.append((train_idx, test_idx))
    return folds

def _save_pred_surv_and_info(surv_df: pd.DataFrame,
                             df_test: pd.DataFrame,
                             dataset_name: str,
                             model_name: str,
                             censorship: int | None,
                             fold_idx: int,
                             replication: int | None,
                             mode: str):
    """
    Optional: saves the predicted survival functions and corresponding
    test information (indices, times, events) — exactly as in the original
    model scripts — to ensure full reproducibility and traceability.
"""

    base_dir = Path(f"./predictions/pred_{dataset_name}_{censorship if censorship is not None else 'NA'}")
    base_dir.mkdir(parents=True, exist_ok=True)

    surv_path = base_dir / f"surv_{model_name}_fold{fold_idx+1}_rep{replication if replication is not None else 'NA'}_{mode}.csv"
    info_path = base_dir / f"info_{model_name}_fold{fold_idx+1}_rep{replication if replication is not None else 'NA'}_{mode}.csv"

    try:
        surv_df.to_csv(surv_path)
        if mode == "standard":
            y_time = df_test["time"].values
            y_event = df_test["event"].values.astype(bool)
        else:
            y_time = df_test["true_time"].values
            y_event = np.ones_like(df_test["event"].values, dtype=bool)
        pd.DataFrame({"idx": df_test.index, "time": y_time, "event": y_event}).to_csv(info_path, index=False)
    except Exception as e:
        print(f"[Warn] Saving predictions failed: {e}")

# ============================================================
# CoxPH
# ============================================================

def nested_cv_coxph(df: pd.DataFrame,
                    features_col: list[str],
                    n_splits_outer: int = 5,
                    dataset_name: str = "",
                    model_name: str = "coxph",
                    censorship: int | None = None,
                    replication: int | None = None,
                    force_test_censoring: bool = False,
                    test_censoring_target: float = 0.9):
    y_event = df['event'].astype(bool).values

    if force_test_censoring:
        outer_folds = _generate_outer_folds_with_high_censoring(df, n_splits_outer, test_censoring_target)
    else:
        skf_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        outer_folds = list(skf_outer.split(df, y_event))

    metrics_std, metrics_true = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
        print(f"\n[CoxPH] Outer Fold {fold_idx+1}")
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()

        # Normalisation
        cols_to_norm = get_columns_to_normalize(df_train)
        scaler = MinMaxScaler()
        df_train[cols_to_norm] = scaler.fit_transform(df_train[cols_to_norm])
        df_test[cols_to_norm] = scaler.transform(df_test[cols_to_norm])

        # Fit
        try:
            model = CoxPHFitter(penalizer=0.1)
            model.fit(df_train.drop(columns=['true_time'], errors='ignore'),
                      duration_col="time", event_col="event")
        except Exception as e:
            print(f"[CoxPH] skipping fold due to error: {e}")
            continue

        # Eval sur 2 modes
        time_grid = np.sort(df['time'].unique())
        for mode in ["standard", "true_time"]:
            T_test = df_test['time'].values if mode == "standard" else df_test['true_time'].values
            E_test = df_test['event'].astype(bool).values if mode == "standard" else np.ones_like(df_test['event'].values, dtype=bool)
            T_train = df_train['time'].values
            E_train = df_train['event'].astype(bool).values

            try:
                surv = model.predict_survival_function(df_test.drop(columns=['true_time'], errors='ignore'), times=time_grid)
                risk_scores = model.predict_partial_hazard(df_test.drop(columns=['true_time'], errors='ignore')).values
                _save_pred_surv_and_info(surv, df_test, dataset_name, model_name, censorship, fold_idx, replication, mode)
            except Exception as e:
                print(f"[CoxPH] prediction failed: {e}")
                continue

            ev = EvalSurv(surv, T_test, E_test, censor_surv="km")
            c_index = ev.concordance_td('antolini')
            c_index_lifelines = concordance_index(T_test, -risk_scores, E_test)
            ibs = compute_ibs(surv_df=surv, T_train=T_train, E_train=E_train,
                              T_test=T_test, E_test=E_test, time_grid=time_grid)

            evaluator = PycoxEvaluator(surv, T_test, E_test, T_train, E_train, predict_time_method="Median")
            try:
                _, bin_stats = evaluator.d_calibration()
                bin_stats /= bin_stats.sum()
                d_calib_score = np.sum(np.abs(bin_stats - 0.1))
            except Exception:
                d_calib_score = np.nan
            try:
                mae_score = evaluator.mae(method="Pseudo_obs")
            except Exception:
                mae_score = np.nan

            one_calib_scores, c_index_at_t_scores = [], []
            for perc in [25, 50, 75]:
                t_ref = round(np.percentile(T_test, perc))
                try:
                    _, ob, exp = evaluator.one_calibration(target_time=t_ref, method="DN")
                    one_calib_scores.append(np.sum(np.abs(np.array(ob) - np.array(exp))))
                except Exception:
                    one_calib_scores.append(np.nan)

                # c-index@t
                try:
                    if t_ref in surv.index:
                        surv_probs_at_t = surv.loc[t_ref].values
                    else:
                        t_closest = surv.index[np.argmin(np.abs(surv.index.values - t_ref))]
                        surv_probs_at_t = surv.loc[t_closest].values
                    cidx_t = concordance_at_time(t_ref, T_test, E_test, surv_probs_at_t)
                except Exception:
                    cidx_t = np.nan
                c_index_at_t_scores.append(cidx_t)

            metrics = {
                'fold': fold_idx + 1,
                'c_index': c_index,
                'c_index_lifelines': c_index_lifelines,
                'ibs': ibs,
                'd_calib': d_calib_score,
                'mae': mae_score,
                'one_calib@25': one_calib_scores[0],
                'one_calib@50': one_calib_scores[1],
                'one_calib@75': one_calib_scores[2],
                'c_index@25': c_index_at_t_scores[0],
                'c_index@50': c_index_at_t_scores[1],
                'c_index@75': c_index_at_t_scores[2],
            }
            (metrics_std if mode == "standard" else metrics_true).append(metrics)

    df_std = pd.DataFrame(metrics_std)
    df_true = pd.DataFrame(metrics_true)

    # Annotate + Save
    df_std_ann = _annotate_metrics_df(df_std, "standard", censorship, replication, dataset_name)
    df_true_ann = _annotate_metrics_df(df_true, "true_time", censorship, replication, dataset_name)
    _write_results_csv(dataset_name, model_name, pd.concat([df_std_ann, df_true_ann], ignore_index=True))

    return df_std_ann, df_true_ann


# ============================================================
# Weibull-AFT
# ============================================================

def nested_cv_weibull_aft(df: pd.DataFrame,
                          features_col: list[str],
                          n_splits_outer: int = 5,
                          dataset_name: str = "",
                          model_name: str = "weibull",
                          censorship: int | None = None,
                          replication: int | None = None,
                          force_test_censoring: bool = False,
                          test_censoring_target: float = 0.9):
    y_event = df['event'].astype(bool).values

    if force_test_censoring:
        outer_folds = _generate_outer_folds_with_high_censoring(df, n_splits_outer, test_censoring_target)
    else:
        skf_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        outer_folds = list(skf_outer.split(df, y_event))

    metrics_std, metrics_true = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
        print(f"\n[Weibull-AFT] Outer Fold {fold_idx+1}")
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()

        cols_to_norm = get_columns_to_normalize(df_train)
        scaler = MinMaxScaler()
        df_train[cols_to_norm] = scaler.fit_transform(df_train[cols_to_norm])
        df_test[cols_to_norm] = scaler.transform(df_test[cols_to_norm])

        try:
            model = WeibullAFTFitter(penalizer=0.1)
            model._scipy_fit_method = "SLSQP"
            model.fit(df_train.drop(columns=['true_time'], errors='ignore'),
                      duration_col="time", event_col="event")
        except ConvergenceError as ce:
            print(f"[Weibull-AFT] skipping fold due to convergence error: {ce}")
            continue
        except Exception as e:
            print(f"[Weibull-AFT] skipping fold due to error: {e}")
            continue

        time_grid = np.sort(df['time'].unique())
        for mode in ["standard", "true_time"]:
            T_test = df_test['time'].values if mode == "standard" else df_test['true_time'].values
            E_test = df_test['event'].astype(bool).values if mode == "standard" else np.ones_like(df_test['event'].values, dtype=bool)
            T_train = df_train['time'].values
            E_train = df_train['event'].astype(bool).values

            try:
                surv = model.predict_survival_function(df_test.drop(columns=['true_time'], errors='ignore'), times=time_grid)
                pred_median = model.predict_median(df_test.drop(columns=['true_time'], errors='ignore'))
                _save_pred_surv_and_info(surv, df_test, dataset_name, model_name, censorship, fold_idx, replication, mode)
            except Exception as e:
                print(f"[Weibull-AFT] prediction failed: {e}")
                continue

            ev = EvalSurv(surv, T_test, E_test, censor_surv="km")
            c_index = ev.concordance_td('antolini')
            c_index_lifelines = concordance_index(T_test, pred_median, E_test)
            ibs = compute_ibs(surv_df=surv, T_train=T_train, E_train=E_train,
                              T_test=T_test, E_test=E_test, time_grid=time_grid)

            evaluator = PycoxEvaluator(surv, T_test, E_test, T_train, E_train, predict_time_method="Median")
            try:
                _, bin_stats = evaluator.d_calibration()
                bin_stats /= bin_stats.sum()
                d_calib_score = np.sum(np.abs(bin_stats - 0.1))
            except Exception:
                d_calib_score = np.nan
            try:
                mae_score = evaluator.mae(method="Pseudo_obs")
            except Exception:
                mae_score = np.nan

            one_calib_scores, c_index_at_t_scores = [], []
            for perc in [25, 50, 75]:
                t_ref = round(np.percentile(T_test, perc))
                try:
                    _, ob, exp = evaluator.one_calibration(target_time=t_ref, method="DN")
                    one_calib_scores.append(np.sum(np.abs(np.array(ob) - np.array(exp))))
                except Exception:
                    one_calib_scores.append(np.nan)
                try:
                    if t_ref in surv.index:
                        surv_probs_at_t = surv.loc[t_ref].values
                    else:
                        t_closest = surv.index[np.argmin(np.abs(surv.index.values - t_ref))]
                        surv_probs_at_t = surv.loc[t_closest].values
                    cidx_t = concordance_at_time(t_ref, T_test, E_test, surv_probs_at_t)
                except Exception:
                    cidx_t = np.nan
                c_index_at_t_scores.append(cidx_t)

            metrics = {
                'fold': fold_idx + 1,
                'c_index': c_index,
                'c_index_lifelines': c_index_lifelines,
                'ibs': ibs,
                'd_calib': d_calib_score,
                'mae': mae_score,
                'one_calib@25': one_calib_scores[0],
                'one_calib@50': one_calib_scores[1],
                'one_calib@75': one_calib_scores[2],
                'c_index@25': c_index_at_t_scores[0],
                'c_index@50': c_index_at_t_scores[1],
                'c_index@75': c_index_at_t_scores[2],
            }
            (metrics_std if mode == "standard" else metrics_true).append(metrics)

    df_std = pd.DataFrame(metrics_std)
    df_true = pd.DataFrame(metrics_true)

    df_std_ann = _annotate_metrics_df(df_std, "standard", censorship, replication, dataset_name)
    df_true_ann = _annotate_metrics_df(df_true, "true_time", censorship, replication, dataset_name)
    _write_results_csv(dataset_name, model_name, pd.concat([df_std_ann, df_true_ann], ignore_index=True))

    return df_std_ann, df_true_ann


# ============================================================
# RSF
# ============================================================

def nested_cv_rsf(df: pd.DataFrame,
                  features_col: list[str],
                  n_splits_outer: int = 5,
                  n_splits_inner: int = 3,
                  n_trials: int = 20,
                  dataset_name: str = "",
                  model_name: str = "rsf",
                  censorship: int | None = None,
                  replication: int | None = None,
                  force_test_censoring: bool = False,
                  test_censoring_target: float = 0.9):
    y_event = df['event'].astype(bool).values

    if force_test_censoring:
        outer_folds = _generate_outer_folds_with_high_censoring(df, n_splits_outer, test_censoring_target)
    else:
        skf_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        outer_folds = list(skf_outer.split(df, y_event))

    metrics_std, metrics_true = [], []

    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_folds):
        print(f"\n[RSF] Outer Fold {fold_idx+1}")
        df_train_val = df.iloc[train_val_idx].copy()
        df_test = df.iloc[test_idx].copy()

        cols_to_norm = get_columns_to_normalize(df_train_val)
        scaler = MinMaxScaler()
        df_train_val[cols_to_norm] = scaler.fit_transform(df_train_val[cols_to_norm])
        df_test[cols_to_norm] = scaler.transform(df_test[cols_to_norm])

        X_train_val = df_train_val[features_col].to_numpy()
        y_time_train_val = df_train_val['time'].values
        y_event_train_val = df_train_val['event'].astype(bool).values

        def _objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
            rsf = RandomSurvivalForest(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                n_jobs=-1,
                random_state=42
            )

            skf_inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
            scores = []
            for inner_train_idx, inner_val_idx in skf_inner.split(X_train_val, y_event_train_val):
                X_train = X_train_val[inner_train_idx]
                X_val = X_train_val[inner_val_idx]

                y_train_struct = Surv.from_arrays(event=y_event_train_val[inner_train_idx],
                                                  time=y_time_train_val[inner_train_idx])
                y_val_event = y_event_train_val[inner_val_idx]
                y_val_time = y_time_train_val[inner_val_idx]

                rsf.fit(X_train, y_train_struct)
                surv_preds = rsf.predict_survival_function(X_val)
                surv_times = surv_preds[0].x
                surv_matrix = np.asarray([[fn(t) for t in surv_times] for fn in surv_preds])
                surv_df = pd.DataFrame(surv_matrix.T, index=surv_times)
                ev = EvalSurv(surv_df, y_val_time, y_val_event, censor_surv='km')
                scores.append(ev.concordance_td("antolini"))

            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(_objective, n_trials=n_trials)
        best_params = study.best_params

        rsf = RandomSurvivalForest(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params["max_features"],
            n_jobs=-1,
            random_state=42
        )
        y_train_val_structured = Surv.from_arrays(event=y_event_train_val, time=y_time_train_val)
        rsf.fit(X_train_val, y_train_val_structured)

        for mode in ["standard", "true_time"]:
            T_test = df_test['time'].values if mode == "standard" else df_test['true_time'].values
            E_test = df_test['event'].astype(bool).values if mode == "standard" else np.ones_like(df_test['event'].values, dtype=bool)
            T_train = y_time_train_val
            E_train = y_event_train_val

            surv_preds = rsf.predict_survival_function(df_test[features_col].to_numpy())
            surv_times = surv_preds[0].x
            surv_matrix = np.asarray([[fn(t) for t in surv_times] for fn in surv_preds])
            surv_df = pd.DataFrame(surv_matrix.T, index=surv_times)

            _save_pred_surv_and_info(surv_df, df_test, dataset_name, model_name, censorship, fold_idx, replication, mode)

            ev = EvalSurv(surv_df, T_test, E_test, censor_surv="km")
            c_index = ev.concordance_td("antolini")
            try:
                risk_scores = rsf.predict(df_test[features_col])
                c_index_lifelines = concordance_index(T_test, -risk_scores, E_test)
            except Exception:
                c_index_lifelines = np.nan

            ibs = compute_ibs(surv_df=surv_df, T_train=T_train, E_train=E_train,
                              T_test=T_test, E_test=E_test, time_grid=surv_times)

            evaluator = PycoxEvaluator(surv_df, T_test, E_test, T_train, E_train, predict_time_method="Median")
            try:
                _, bin_stats = evaluator.d_calibration()
                bin_stats /= bin_stats.sum()
                d_calib_score = np.sum(np.abs(bin_stats - 0.1))
            except Exception:
                d_calib_score = np.nan
            try:
                mae_score = evaluator.mae(method="Pseudo_obs")
            except Exception:
                mae_score = np.nan

            one_calib_scores, c_index_at_t_scores = [], []
            for perc in [25, 50, 75]:
                t_ref = round(np.percentile(T_test, perc))
                try:
                    _, ob, exp = evaluator.one_calibration(target_time=t_ref, method="DN")
                    one_calib_scores.append(np.sum(np.abs(np.array(ob) - np.array(exp))))
                except Exception:
                    one_calib_scores.append(np.nan)
                try:
                    if t_ref in surv_df.index:
                        surv_probs_at_t = surv_df.loc[t_ref].values
                    else:
                        t_closest = surv_df.index[np.argmin(np.abs(surv_df.index.values - t_ref))]
                        surv_probs_at_t = surv_df.loc[t_closest].values
                    cidx_t = concordance_at_time(t_ref, T_test, E_test, surv_probs_at_t)
                except Exception:
                    cidx_t = np.nan
                c_index_at_t_scores.append(cidx_t)

            metrics = {
                'fold': fold_idx + 1,
                'c_index': c_index,
                'c_index_lifelines': c_index_lifelines,
                'ibs': ibs,
                'd_calib': d_calib_score,
                'mae': mae_score,
                'one_calib@25': one_calib_scores[0],
                'one_calib@50': one_calib_scores[1],
                'one_calib@75': one_calib_scores[2],
                'c_index@25': c_index_at_t_scores[0],
                'c_index@50': c_index_at_t_scores[1],
                'c_index@75': c_index_at_t_scores[2],
            }
            (metrics_std if mode == "standard" else metrics_true).append(metrics)

    df_std = pd.DataFrame(metrics_std)
    df_true = pd.DataFrame(metrics_true)

    df_std_ann = _annotate_metrics_df(df_std, "standard", censorship, replication, dataset_name)
    df_true_ann = _annotate_metrics_df(df_true, "true_time", censorship, replication, dataset_name)
    _write_results_csv(dataset_name, model_name, pd.concat([df_std_ann, df_true_ann], ignore_index=True))

    return df_std_ann, df_true_ann


# ============================================================
# NMTLR
# ============================================================

def _create_mtlr_model(in_features, out_features, dropout, lr, labtrans):
    net = tt.practical.MLPVanilla(in_features, [32, 32], out_features, batch_norm=False, dropout=dropout)
    model = MTLR(net, tt.optim.Adam(lr), duration_index=labtrans.cuts)
    return model

def nested_cv_mtlr(df: pd.DataFrame,
                   features_col: list[str],
                   n_splits_outer: int = 5,
                   n_splits_inner: int = 3,
                   n_trials: int = 20,
                   dataset_name: str = "",
                   model_name: str = "nmtlr",
                   censorship: int | None = None,
                   replication: int | None = None,
                   force_test_censoring: bool = False,
                   test_censoring_target: float = 0.9):
    y_time = df['time'].astype(float)
    y_event = df['event'].astype(bool)

    # découpages réguliers en 100 coupes
    labtrans = MTLR.label_transform(np.linspace(y_time.min(), y_time.max(), 100))

    if force_test_censoring:
        outer_folds = _generate_outer_folds_with_high_censoring(df, n_splits_outer, test_censoring_target)
    else:
        skf_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        outer_folds = list(skf_outer.split(df, y_event.values))

    metrics_std, metrics_true = [], []

    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_folds):
        print(f"\n[NMTLR] Outer Fold {fold_idx+1}")
        df_train_val = df.iloc[train_val_idx].copy()
        df_test = df.iloc[test_idx].copy()

        cols_to_norm = get_columns_to_normalize(df_train_val)
        scaler = MinMaxScaler()
        df_train_val[cols_to_norm] = scaler.fit_transform(df_train_val[cols_to_norm])
        df_test[cols_to_norm] = scaler.transform(df_test[cols_to_norm])

        X_train_val = df_train_val[features_col].to_numpy(dtype=np.float32)
        X_test = df_test[features_col].to_numpy(dtype=np.float32)
        y_time_train_val = df_train_val['time'].values.astype(float)
        y_event_train_val = df_train_val['event'].values.astype(bool)
        y_time_test = df_test['time'].values.astype(float)
        y_event_test = df_test['event'].values.astype(bool)

        best_score, best_params = -np.inf, None

        def _objective(trial):
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            lr = trial.suggest_categorical("lr", [1e-1, 1e-2, 1e-3])
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

            model = _create_mtlr_model(X_train_val.shape[1], labtrans.out_features, dropout, lr, labtrans)

            skf_inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
            scores = []
            for inner_train_idx, inner_val_idx in skf_inner.split(X_train_val, y_event_train_val):
                X_train = X_train_val[inner_train_idx]
                X_val = X_train_val[inner_val_idx]

                y_time_train = y_time_train_val[inner_train_idx]
                y_time_val = y_time_train_val[inner_val_idx]
                y_event_train = y_event_train_val[inner_train_idx]
                y_event_val = y_event_train_val[inner_val_idx]

                y_train_trans = labtrans.transform(y_time_train, y_event_train)
                y_val_trans = labtrans.transform(y_time_val, y_event_val)

                model.fit(X_train, y_train_trans, batch_size=batch_size, epochs=100,
                          val_data=(X_val, y_val_trans), verbose=False)
                surv = model.predict_surv_df(X_val)
                ev = EvalSurv(surv, y_time_val, y_event_val, censor_surv='km')
                scores.append(ev.concordance_td('antolini'))

            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(_objective, n_trials=n_trials)
        best_params = study.best_params

        model = _create_mtlr_model(X_train_val.shape[1], labtrans.out_features,
                                   best_params["dropout"], best_params["lr"], labtrans)
        y_train_val_transformed = labtrans.transform(y_time_train_val, y_event_train_val)
        model.fit(X_train_val, y_train_val_transformed, batch_size=best_params["batch_size"],
                  epochs=100, verbose=False)

        for mode in ["standard", "true_time"]:
            if mode == "standard":
                y_time_eval = y_time_test
                y_event_eval = y_event_test
            else:
                y_time_eval = df_test['true_time'].values
                y_event_eval = np.ones_like(y_event_test, dtype=bool)

            surv = model.predict_surv_df(X_test)

            _save_pred_surv_and_info(surv, df_test, dataset_name, model_name, censorship, fold_idx, replication, mode)

            ev = EvalSurv(surv, y_time_eval, y_event_eval, censor_surv='km')
            try:
                c_index = ev.concordance_td('antolini')
            except Exception:
                c_index = np.nan

            try:
                pred_median = surv.apply(lambda s: s[s < 0.5].index.min() if any(s < 0.5) else s.index[-1])
                c_index_lifelines = concordance_index(y_time_eval, pred_median.values.astype(float), y_event_eval)
            except Exception:
                c_index_lifelines = np.nan

            try:
                ibs = compute_ibs(surv_df=surv, T_train=y_time_train_val, E_train=y_event_train_val,
                                  T_test=y_time_eval, E_test=y_event_eval, time_grid=surv.index.values.astype(float))
            except Exception:
                ibs = np.nan

            evaluator = PycoxEvaluator(surv, y_time_eval, y_event_eval, y_time_train_val, y_event_train_val, predict_time_method="Median")
            try:
                _, bin_stats = evaluator.d_calibration()
                bin_stats /= bin_stats.sum()
                d_calib_score = np.sum(np.abs(bin_stats - 0.1))
            except Exception:
                d_calib_score = np.nan
            try:
                mae_score = evaluator.mae(method="Pseudo_obs")
            except Exception:
                mae_score = np.nan

            one_calib_scores, c_index_at_t_scores = [], []
            for perc in [25, 50, 75]:
                t_ref = round(np.percentile(y_time_eval, perc))
                try:
                    _, ob, exp = evaluator.one_calibration(target_time=t_ref, method="DN")
                    one_calib_scores.append(np.sum(np.abs(np.array(ob) - np.array(exp))))
                except Exception:
                    one_calib_scores.append(np.nan)
                try:
                    if t_ref in surv.index:
                        surv_probs_at_t = surv.loc[t_ref].values
                    else:
                        t_closest = surv.index[np.argmin(np.abs(surv.index.values - t_ref))]
                        surv_probs_at_t = surv.loc[t_closest].values
                    cidx_t = concordance_at_time(t_ref, y_time_eval, y_event_eval, surv_probs_at_t)
                except Exception:
                    cidx_t = np.nan
                c_index_at_t_scores.append(cidx_t)

            metrics = {
                'fold': fold_idx + 1,
                'c_index': c_index,
                'c_index_lifelines': c_index_lifelines,
                'ibs': ibs,
                'd_calib': d_calib_score,
                'mae': mae_score,
                'one_calib@25': one_calib_scores[0],
                'one_calib@50': one_calib_scores[1],
                'one_calib@75': one_calib_scores[2],
                'c_index@25': c_index_at_t_scores[0],
                'c_index@50': c_index_at_t_scores[1],
                'c_index@75': c_index_at_t_scores[2],
            }
            (metrics_std if mode == "standard" else metrics_true).append(metrics)

    df_std = pd.DataFrame(metrics_std)
    df_true = pd.DataFrame(metrics_true)

    df_std_ann = _annotate_metrics_df(df_std, "standard", censorship, replication, dataset_name)
    df_true_ann = _annotate_metrics_df(df_true, "true_time", censorship, replication, dataset_name)
    _write_results_csv(dataset_name, model_name, pd.concat([df_std_ann, df_true_ann], ignore_index=True))

    return df_std_ann, df_true_ann


# ============================================================
# DeepHit
# ============================================================

def _create_deephit_model(in_features, out_features, dropout, alpha, sigma, lr, labtrans):
    net = tt.practical.MLPVanilla(in_features, [32, 32], out_features, batch_norm=False, dropout=dropout)
    model = DeepHitSingle(net, tt.optim.Adam, alpha=alpha, sigma=sigma, duration_index=labtrans.cuts)
    model.optimizer.set_lr(lr)
    return model

def nested_cv_deephit(df: pd.DataFrame,
                      features_col: list[str],
                      n_splits_outer: int = 5,
                      n_splits_inner: int = 3,
                      n_trials: int = 20,
                      dataset_name: str = "",
                      model_name: str = "deephit",
                      censorship: int | None = None,
                      replication: int | None = None,
                      force_test_censoring: bool = False,
                      test_censoring_target: float = 0.9):
    y_time = df['time'].astype(float).values
    y_event = df['event'].astype(bool).values

    # cuts via quantiles (100)
    labtrans = DeepHitSingle.label_transform(np.unique(np.quantile(y_time, np.linspace(0, 1, 100))))
    X_all = df[features_col]

    if force_test_censoring:
        outer_folds = _generate_outer_folds_with_high_censoring(df, n_splits_outer, test_censoring_target)
    else:
        skf_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
        outer_folds = list(skf_outer.split(X_all, y_event))

    metrics_std, metrics_true = [], []

    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_folds):
        print(f"\n[DeepHit] Outer Fold {fold_idx+1}")
        df_train_val = df.iloc[train_val_idx].copy()
        df_test = df.iloc[test_idx].copy()

        cols_to_norm = get_columns_to_normalize(df_train_val)
        scaler = StandardScaler()
        df_train_val[cols_to_norm] = scaler.fit_transform(df_train_val[cols_to_norm])
        df_test[cols_to_norm] = scaler.transform(df_test[cols_to_norm])

        X_train_val = df_train_val[features_col].to_numpy(dtype=np.float32)
        X_test = df_test[features_col].to_numpy(dtype=np.float32)
        y_time_train_val = df_train_val['time'].values.astype(float)
        y_event_train_val = df_train_val['event'].values.astype(bool)
        y_time_test = df_test['time'].values.astype(float)
        y_event_test = df_test['event'].values.astype(bool)

        best_params = None
        best_score = -np.inf

        def _objective(trial):
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            alpha = trial.suggest_float("alpha", 0.1, 1.0, step=0.1)
            sigma = trial.suggest_float("sigma", 0.01, 0.2, step=0.01)
            lr = trial.suggest_categorical("lr", [1e-1, 1e-2, 1e-3])
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            model = _create_deephit_model(X_train_val.shape[1], labtrans.out_features, dropout, alpha, sigma, lr, labtrans)

            skf_inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
            scores = []
            for inner_train_idx, inner_val_idx in skf_inner.split(X_train_val, y_event_train_val):
                X_train = X_train_val[inner_train_idx]
                X_val = X_train_val[inner_val_idx]

                y_time_train = y_time_train_val[inner_train_idx]
                y_time_val = y_time_train_val[inner_val_idx]
                y_event_train = y_event_train_val[inner_train_idx]
                y_event_val = y_event_train_val[inner_val_idx]

                y_train_trans = labtrans.transform(y_time_train, y_event_train)
                y_val_trans = labtrans.transform(y_time_val, y_event_val)

                model.fit(X_train, y_train_trans, batch_size=batch_size, epochs=50,
                          val_data=(X_val, y_val_trans), verbose=False)
                surv = model.predict_surv_df(X_val)
                ev = EvalSurv(surv, y_time_val, y_event_val, censor_surv='km')
                scores.append(ev.concordance_td('antolini'))

            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        # Warm start possible
        study.enqueue_trial({"alpha": 0.5, "sigma": 0.1, "dropout": 0.2, "lr": 1e-3, "batch_size": 64})
        study.optimize(_objective, n_trials=n_trials)
        best_params = study.best_params

        model = _create_deephit_model(X_train_val.shape[1], labtrans.out_features,
                                      best_params["dropout"], best_params["alpha"], best_params["sigma"],
                                      best_params["lr"], labtrans)
        y_train_val_transformed = labtrans.transform(y_time_train_val, y_event_train_val)
        model.fit(X_train_val, y_train_val_transformed, batch_size=best_params["batch_size"],
                  epochs=50, verbose=False)

        for mode in ["standard", "true_time"]:
            if mode == "standard":
                y_time_eval = y_time_test
                y_event_eval = y_event_test
            else:
                y_time_eval = df_test['true_time'].values
                y_event_eval = np.ones_like(y_event_test, dtype=bool)

            surv = model.predict_surv_df(X_test)

            _save_pred_surv_and_info(surv, df_test, dataset_name, model_name, censorship, fold_idx, replication, mode)

            ev = EvalSurv(surv, y_time_eval, y_event_eval, censor_surv='km')
            try:
                c_index = ev.concordance_td('antolini')
            except Exception:
                c_index = np.nan

            try:
                pred_median = surv.apply(lambda s: s[s < 0.5].index.min() if any(s < 0.5) else s.index[-1])
                c_index_lifelines = concordance_index(y_time_eval, pred_median.values.astype(float), y_event_eval)
            except Exception:
                c_index_lifelines = np.nan

            try:
                ibs = compute_ibs(surv_df=surv, T_train=y_time_train_val, E_train=y_event_train_val,
                                  T_test=y_time_eval, E_test=y_event_eval, time_grid=surv.index.values.astype(float))
            except Exception:
                ibs = np.nan

            evaluator = PycoxEvaluator(surv, y_time_eval, y_event_eval, y_time_train_val, y_event_train_val, predict_time_method="Median")
            try:
                _, bin_stats = evaluator.d_calibration()
                bin_stats /= bin_stats.sum()
                d_calib_score = np.sum(np.abs(bin_stats - 0.1))
            except Exception:
                d_calib_score = np.nan
            try:
                mae_score = evaluator.mae(method="Pseudo_obs")
            except Exception:
                mae_score = np.nan

            one_calib_scores, c_index_at_t_scores = [], []
            for perc in [25, 50, 75]:
                t_ref = round(np.percentile(y_time_eval, perc))
                try:
                    _, ob, exp = evaluator.one_calibration(target_time=t_ref, method="DN")
                    one_calib_scores.append(np.sum(np.abs(np.array(ob) - np.array(exp))))
                except Exception:
                    one_calib_scores.append(np.nan)
                try:
                    if t_ref in surv.index:
                        surv_probs_at_t = surv.loc[t_ref].values
                    else:
                        t_closest = surv.index[np.argmin(np.abs(surv.index.values - t_ref))]
                        surv_probs_at_t = surv.loc[t_closest].values
                    cidx_t = concordance_at_time(t_ref, y_time_eval, y_event_eval, surv_probs_at_t)
                except Exception:
                    cidx_t = np.nan
                c_index_at_t_scores.append(cidx_t)

            metrics = {
                'fold': fold_idx + 1,
                'c_index': c_index,
                'c_index_lifelines': c_index_lifelines,
                'ibs': ibs,
                'd_calib': d_calib_score,
                'mae': mae_score,
                'one_calib@25': one_calib_scores[0],
                'one_calib@50': one_calib_scores[1],
                'one_calib@75': one_calib_scores[2],
                'c_index@25': c_index_at_t_scores[0],
                'c_index@50': c_index_at_t_scores[1],
                'c_index@75': c_index_at_t_scores[2],
            }
            (metrics_std if mode == "standard" else metrics_true).append(metrics)

    df_std = pd.DataFrame(metrics_std)
    df_true = pd.DataFrame(metrics_true)

    df_std_ann = _annotate_metrics_df(df_std, "standard", censorship, replication, dataset_name)
    df_true_ann = _annotate_metrics_df(df_true, "true_time", censorship, replication, dataset_name)
    _write_results_csv(dataset_name, model_name, pd.concat([df_std_ann, df_true_ann], ignore_index=True))

    return df_std_ann, df_true_ann


# ============================================================
# Dispatcher
# ============================================================

def run_model(model_name: str,
              df: pd.DataFrame,
              features_col: list[str],
              **kwargs):
    """
    Unified call:
    run_model("coxph", df, features_col, dataset_name=..., censorship=..., replication=...)
    """
    model_name = model_name.lower()
    mapping = {
        "coxph": nested_cv_coxph,
        "weibull": nested_cv_weibull_aft,
        "weibull_aft": nested_cv_weibull_aft,
        "aft": nested_cv_weibull_aft,
        "rsf": nested_cv_rsf,
        "nmtlr": nested_cv_mtlr,
        "deephit": nested_cv_deephit,
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         f"Available: {', '.join(mapping.keys())}")
    # injecter le nom du modèle s'il n'est pas donné
    kwargs.setdefault("model_name", model_name)
    return mapping[model_name](df, features_col, **kwargs)
