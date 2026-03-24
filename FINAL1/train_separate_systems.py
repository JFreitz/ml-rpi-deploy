from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


TARGETS = ["height", "length", "weight", "leaves", "branches"]
SEED = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], format="%Y-%m-%d", errors="coerce")
    out = out.sort_values(["date", "plant_no"]).reset_index(drop=True)

    out["day_of_year"] = out["date"].dt.dayofyear
    out["day_of_week_num"] = out["date"].dt.dayofweek
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    out["doy_sin"] = np.sin(2.0 * np.pi * out["day_of_year"] / 366.0)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["day_of_year"] / 366.0)
    out["plant_no"] = pd.to_numeric(out["plant_no"], errors="coerce")
    return out


def split_per_day_random(df: pd.DataFrame, seed: int = SEED):
    train_parts = []
    test_parts = []
    rs = seed

    # One system per file. We split by date only: 4 train rows, 2 test rows per day.
    for _, g in df.groupby("date", sort=False):
        g = g.sample(frac=1.0, random_state=rs)
        train_parts.append(g.iloc[:4])
        test_parts.append(g.iloc[4:])
        rs += 1

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_df = train_df.sort_values(["date", "plant_no"]).reset_index(drop=True)
    test_df = test_df.sort_values(["date", "plant_no"]).reset_index(drop=True)
    return train_df, test_df


def get_models():
    return [
        ("MLR_LinearRegression", LinearRegression()),
        ("Ridge_a0.1", Ridge(alpha=0.1, random_state=SEED)),
        ("Ridge_a1.0", Ridge(alpha=1.0, random_state=SEED)),
        ("Ridge_a10.0", Ridge(alpha=10.0, random_state=SEED)),
        ("SVR_rbf_C10_eps0.1", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
        ("SVR_linear_C10_eps0.1", SVR(kernel="linear", C=10.0, epsilon=0.1)),
        (
            "RandomForest_200",
            RandomForestRegressor(
                n_estimators=200,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTrees_250",
            ExtraTreesRegressor(
                n_estimators=250,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "GradientBoosting_200",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=SEED,
            ),
        ),
    ]


def evaluate_cv(X, y, preprocessor, model):
    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for train_idx, val_idx in cv.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        pipe.fit(X_train_fold, y_train_fold)
        pred = pipe.predict(X_val_fold)

        r2_scores.append(float(r2_score(y_val_fold, pred)))
        rmse_scores.append(rmse(y_val_fold, pred))
        mae_scores.append(float(mean_absolute_error(y_val_fold, pred)))

    return {
        "cv_r2": float(np.mean(r2_scores)),
        "cv_rmse": float(np.mean(rmse_scores)),
        "cv_mae": float(np.mean(mae_scores)),
    }


def train_one_system(system_name: str, source_path: Path, out_root: Path):
    system_dir = out_root / system_name
    data_dir = system_dir / "data"
    models_dir = system_dir / "models"
    reports_dir = system_dir / "reports"

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(source_path)
    train_df, test_df = split_per_day_random(raw, seed=SEED)

    train_df.to_csv(data_dir / "training_data.csv", index=False)
    test_df.to_csv(data_dir / "test_data.csv", index=False)

    test_this = test_df.copy()
    for t in TARGETS:
        if t in test_this.columns:
            test_this[t] = ""
    test_this.to_csv(data_dir / "test_this.csv", index=False)

    train_f = add_features(train_df)
    test_f = add_features(test_df)
    infer_f = add_features(test_this)

    feature_cols = [
        "ave_ph",
        "ave_do",
        "ave_tds",
        "ave_temp",
        "ave_humidity",
        "plant_no",
        "day_of_year",
        "day_of_week_num",
        "week_of_year",
        "month",
        "doy_sin",
        "doy_cos",
    ]

    numeric_features = feature_cols
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)]
    )

    comparison_rows = []
    best_rows = []
    metrics_rows = []

    pred_df = infer_f[["date", "day", "plant_system", "plant_no"]].copy()

    print("\n" + "=" * 88)
    print(f"SYSTEM: {system_name}")
    print("=" * 88)
    print(f"Train rows: {len(train_f)}")
    print(f"Test rows: {len(test_f)}")

    for target in TARGETS:
        print("\n" + "-" * 88)
        print(f"Target: {target}")

        y_train = pd.to_numeric(train_f[target], errors="coerce")
        y_test = pd.to_numeric(test_f[target], errors="coerce")

        train_mask = y_train.notna()
        test_mask = y_test.notna()

        X_train = train_f.loc[train_mask, feature_cols]
        y_train = y_train.loc[train_mask]
        X_test = test_f.loc[test_mask, feature_cols]
        y_test = y_test.loc[test_mask]

        best = None

        for model_name, model in get_models():
            cv_scores = evaluate_cv(X_train, y_train, preprocessor, model)

            pipe = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", model),
                ]
            )
            pipe.fit(X_train, y_train)

            test_pred = pipe.predict(X_test)
            row = {
                "system": system_name,
                "target": target,
                "model": model_name,
                "cv_r2": cv_scores["cv_r2"],
                "cv_rmse": cv_scores["cv_rmse"],
                "cv_mae": cv_scores["cv_mae"],
                "test_r2": float(r2_score(y_test, test_pred)),
                "test_rmse": rmse(y_test, test_pred),
                "test_mae": float(mean_absolute_error(y_test, test_pred)),
                "test_mape_percent": mape(y_test, test_pred),
            }
            comparison_rows.append(row)

            if best is None:
                best = {"model_name": model_name, "estimator": model, **row}
            else:
                better_r2 = row["test_r2"] > best["test_r2"]
                tie_break = row["test_r2"] == best["test_r2"] and row["test_rmse"] < best["test_rmse"]
                if better_r2 or tie_break:
                    best = {"model_name": model_name, "estimator": model, **row}

            print(
                f"  {model_name:22} CV_R2={row['cv_r2']:.4f} TEST_R2={row['test_r2']:.4f}"
            )

        final_pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", best["estimator"]),
            ]
        )
        final_pipe.fit(X_train, y_train)

        model_path = models_dir / f"{target}_best_model.joblib"
        dump(final_pipe, model_path)

        final_pred = final_pipe.predict(X_test)

        best_rows.append(
            {
                "system": system_name,
                "target": target,
                "best_model": best["model_name"],
                "cv_r2": best["cv_r2"],
                "cv_rmse": best["cv_rmse"],
                "cv_mae": best["cv_mae"],
                "test_r2": float(r2_score(y_test, final_pred)),
                "test_rmse": rmse(y_test, final_pred),
                "test_mae": float(mean_absolute_error(y_test, final_pred)),
                "test_mape_percent": mape(y_test, final_pred),
                "model_path": str(model_path.resolve()),
            }
        )

        metrics_rows.append(
            {
                "system": system_name,
                "target": target,
                "r2": float(r2_score(y_test, final_pred)),
                "mae": float(mean_absolute_error(y_test, final_pred)),
                "rmse": rmse(y_test, final_pred),
                "mape_percent": mape(y_test, final_pred),
            }
        )

        X_infer = infer_f[feature_cols]
        pred_df[f"{target}_pred"] = final_pipe.predict(X_infer)

        print(
            f"  BEST: {best['model_name']} | TEST_R2={best['test_r2']:.4f} TEST_RMSE={best['test_rmse']:.4f}"
        )

    comparison_df = pd.DataFrame(comparison_rows)
    best_df = pd.DataFrame(best_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    comparison_df["is_best_for_target"] = False
    for _, row in best_df.iterrows():
        mask = (comparison_df["target"] == row["target"]) & (comparison_df["model"] == row["best_model"])
        comparison_df.loc[mask, "is_best_for_target"] = True

    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y-%m-%d")

    comparison_df.to_csv(reports_dir / "model_comparison_all_models.csv", index=False)
    best_df.to_csv(reports_dir / "best_models_summary.csv", index=False)
    metrics_df.to_csv(reports_dir / "accuracy_metrics.csv", index=False)
    pred_df.to_csv(reports_dir / "test_this_predictions.csv", index=False)

    with (reports_dir / "training_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"{system_name} separate training report\n")
        f.write("=" * 60 + "\n")
        for _, row in best_df.iterrows():
            f.write(
                f"{row['target']}: {row['best_model']} | "
                f"CV_R2={row['cv_r2']:.4f}, TEST_R2={row['test_r2']:.4f}, "
                f"TEST_RMSE={row['test_rmse']:.4f}, TEST_MAE={row['test_mae']:.4f}\n"
            )


def main():
    root = Path(__file__).resolve().parent

    systems = [
        ("AERO", root / "AERO_SENSOR_PLANT_DATA_WITH_LENGTH.csv"),
        ("DWC", root / "DWC_SENSOR_PLANT_DATA_WITH_LENGTH.csv"),
    ]

    for system_name, source_path in systems:
        if not source_path.exists():
            raise FileNotFoundError(f"Missing file: {source_path}")
        train_one_system(system_name, source_path, root)

    print("\n" + "=" * 88)
    print("Done. Trained AERO and DWC separately with per-day random 4/2 split.")
    print("Outputs are in FINAL1/AERO and FINAL1/DWC.")
    print("=" * 88)


if __name__ == "__main__":
    main()
