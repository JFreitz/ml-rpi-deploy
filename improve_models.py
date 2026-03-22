import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGETS = ["height", "length", "weight", "leaves", "branches"]


def rmse(y_true, y_pred):
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], format="%Y-%m-%d", errors="coerce")
    out = out.sort_values(["date", "plant_system", "plant_no"]).reset_index(drop=True)

    out["day_of_year"] = out["date"].dt.dayofyear
    out["day_of_week_num"] = out["date"].dt.dayofweek
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)

    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 366.0)

    out["plant_no"] = pd.to_numeric(out["plant_no"], errors="coerce")
    out["plant_no_sq"] = out["plant_no"] ** 2

    return out


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def date_cv_score(X, y, dates, model, preprocessor, n_splits=5):
    unique_dates = np.array(sorted(pd.Series(dates).unique()))
    if len(unique_dates) < 4:
        return {"cv_r2": np.nan, "cv_rmse": np.nan, "cv_mae": np.nan}

    folds = min(n_splits, len(unique_dates) - 1)
    splitter = TimeSeriesSplit(n_splits=folds)

    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for tr_idx, va_idx in splitter.split(unique_dates):
        tr_dates = set(unique_dates[tr_idx])
        va_dates = set(unique_dates[va_idx])

        tr_mask = pd.Series(dates).isin(tr_dates).to_numpy()
        va_mask = pd.Series(dates).isin(va_dates).to_numpy()

        X_tr = X.loc[tr_mask]
        y_tr = y.loc[tr_mask]
        X_va = X.loc[va_mask]
        y_va = y.loc[va_mask]

        pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", model),
            ]
        )
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)

        r2_scores.append(float(r2_score(y_va, pred)))
        rmse_scores.append(rmse(y_va, pred))
        mae_scores.append(float(mean_absolute_error(y_va, pred)))

    return {
        "cv_r2": float(np.mean(r2_scores)),
        "cv_rmse": float(np.mean(rmse_scores)),
        "cv_mae": float(np.mean(mae_scores)),
    }


def build_search_space():
    spaces = []

    et_space = {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [None, 6, 10, 16, 24],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", "log2", 0.7, 1.0],
    }
    for params in ParameterSampler(et_space, n_iter=24, random_state=RANDOM_STATE):
        spaces.append(("ExtraTrees", params, "raw"))

    rf_space = {
        "n_estimators": [300, 500, 800],
        "max_depth": [None, 8, 14, 20],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", "log2", 0.7, 1.0],
    }
    for params in ParameterSampler(rf_space, n_iter=20, random_state=RANDOM_STATE + 7):
        spaces.append(("RandomForest", params, "raw"))

    gb_space = {
        "n_estimators": [200, 400, 700],
        "learning_rate": [0.02, 0.05, 0.08, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 0.85, 1.0],
        "min_samples_leaf": [1, 2, 4],
    }
    for params in ParameterSampler(gb_space, n_iter=18, random_state=RANDOM_STATE + 11):
        spaces.append(("GradientBoosting", params, "raw"))
        spaces.append(("GradientBoosting", params, "log1p"))

    ridge_space = {
        "alpha": [0.01, 0.1, 1.0, 5.0, 10.0, 20.0],
    }
    for params in ParameterSampler(ridge_space, n_iter=6, random_state=RANDOM_STATE + 19):
        spaces.append(("Ridge", params, "raw"))

    en_space = {
        "alpha": [0.0005, 0.001, 0.01, 0.05, 0.1],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "max_iter": [5000],
    }
    for params in ParameterSampler(en_space, n_iter=10, random_state=RANDOM_STATE + 23):
        spaces.append(("ElasticNet", params, "raw"))

    return spaces


def make_estimator(name, params):
    if name == "ExtraTrees":
        return ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if name == "RandomForest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    if name == "Ridge":
        return Ridge(random_state=RANDOM_STATE, **params)
    if name == "ElasticNet":
        return ElasticNet(random_state=RANDOM_STATE, **params)
    raise ValueError(f"Unknown model name: {name}")


def maybe_wrap_log(name, estimator, mode):
    if mode != "log1p":
        return estimator

    if name in {"GradientBoosting", "RandomForest", "ExtraTrees"}:
        return TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    return estimator


def main():
    root = Path(".")
    df_train = add_features(pd.read_csv(root / "training_data.csv"))
    df_test = add_features(pd.read_csv(root / "test_data.csv"))
    df_input = add_features(pd.read_csv(root / "test_this.csv"))

    feature_cols = [
        "ave_ph",
        "ave_do",
        "ave_tds",
        "ave_temp",
        "ave_humidity",
        "plant_no",
        "plant_no_sq",
        "day_of_year",
        "day_of_week_num",
        "week_of_year",
        "doy_sin",
        "doy_cos",
        "plant_system",
    ]
    numeric_features = [
        "ave_ph",
        "ave_do",
        "ave_tds",
        "ave_temp",
        "ave_humidity",
        "plant_no",
        "plant_no_sq",
        "day_of_year",
        "day_of_week_num",
        "week_of_year",
        "doy_sin",
        "doy_cos",
    ]
    categorical_features = ["plant_system"]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    search_space = build_search_space()

    improved_models_dir = root / "models_improved"
    improved_models_dir.mkdir(exist_ok=True)

    results_rows = []
    trial_rows = []

    pred_out = df_input[["date", "day", "plant_system", "plant_no"]].copy()

    print("=" * 88)
    print("IMPROVED DATE-AWARE MODEL TUNING")
    print("=" * 88)
    print(f"Training rows: {len(df_train)} | Test rows: {len(df_test)}")
    print(f"Candidates per target: {len(search_space)}")

    for target in TARGETS:
        print("\n" + "-" * 88)
        print(f"Target: {target}")

        y_train = pd.to_numeric(df_train[target], errors="coerce")
        y_test = pd.to_numeric(df_test[target], errors="coerce")

        tr_mask = y_train.notna()
        te_mask = y_test.notna()

        X_train = df_train.loc[tr_mask, feature_cols]
        y_train = y_train.loc[tr_mask]
        tr_dates = df_train.loc[tr_mask, "date"]

        best = None

        for i, (name, params, target_mode) in enumerate(search_space, start=1):
            est = make_estimator(name, params)
            est = maybe_wrap_log(name, est, target_mode)

            cv = date_cv_score(X_train, y_train, tr_dates, est, preprocessor, n_splits=5)

            trial = {
                "target": target,
                "model": name,
                "target_mode": target_mode,
                "params": json_safe(params),
                **cv,
            }
            trial_rows.append(trial)

            if best is None:
                best = {"name": name, "params": params, "target_mode": target_mode, **cv}
            else:
                better = cv["cv_r2"] > best["cv_r2"]
                tie = cv["cv_r2"] == best["cv_r2"] and cv["cv_rmse"] < best["cv_rmse"]
                if better or tie:
                    best = {"name": name, "params": params, "target_mode": target_mode, **cv}

            if i % 20 == 0:
                print(f"  checked {i}/{len(search_space)} candidates")

        best_est = make_estimator(best["name"], best["params"])
        best_est = maybe_wrap_log(best["name"], best_est, best["target_mode"])

        final_pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", best_est),
            ]
        )
        final_pipe.fit(X_train, y_train)

        model_path = improved_models_dir / f"{target}_best_model.joblib"
        dump(final_pipe, model_path)

        row = {
            "target": target,
            "best_model": best["name"],
            "target_mode": best["target_mode"],
            "best_params": json_safe(best["params"]),
            "cv_r2": best["cv_r2"],
            "cv_rmse": best["cv_rmse"],
            "cv_mae": best["cv_mae"],
            "model_path": str(model_path.resolve()),
        }

        if te_mask.sum() > 0:
            X_test = df_test.loc[te_mask, feature_cols]
            y_test_valid = y_test.loc[te_mask]
            test_pred = final_pipe.predict(X_test)
            row["test_r2"] = float(r2_score(y_test_valid, test_pred))
            row["test_rmse"] = rmse(y_test_valid, test_pred)
            row["test_mae"] = float(mean_absolute_error(y_test_valid, test_pred))

        results_rows.append(row)

        pred_out[f"{target}_pred"] = final_pipe.predict(df_input[feature_cols])

        print(
            f"  BEST: {best['name']} ({best['target_mode']}) | "
            f"CV_R2={row['cv_r2']:.4f} CV_RMSE={row['cv_rmse']:.4f} | "
            f"TEST_R2={row.get('test_r2', np.nan):.4f}"
        )

    results = pd.DataFrame(results_rows)
    trials = pd.DataFrame(trial_rows)

    results.to_csv(root / "best_models_summary_improved.csv", index=False)
    trials.to_csv(root / "model_trials_improved.csv", index=False)

    pred_out["date"] = pd.to_datetime(pred_out["date"]).dt.strftime("%Y-%m-%d")
    pred_out.to_csv(root / "test_this_predictions_improved.csv", index=False)

    with (root / "training_report_improved.txt").open("w", encoding="utf-8") as f:
        f.write("Improved training summary\n")
        f.write("=" * 30 + "\n")
        for _, r in results.iterrows():
            f.write(
                f"{r['target']}: {r['best_model']} ({r['target_mode']}) | "
                f"CV_R2={r['cv_r2']:.4f}, TEST_R2={r.get('test_r2', np.nan):.4f}\n"
            )

    print("\n" + "=" * 88)
    print("Improved tuning complete.")
    print("Saved:")
    print("  - best_models_summary_improved.csv")
    print("  - model_trials_improved.csv")
    print("  - test_this_predictions_improved.csv")
    print("  - training_report_improved.txt")
    print("  - models_improved/*.joblib")
    print("=" * 88)


def json_safe(params):
    safe = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            safe[k] = int(v)
        elif isinstance(v, (np.floating,)):
            safe[k] = float(v)
        else:
            safe[k] = v
    return safe


if __name__ == "__main__":
    main()
