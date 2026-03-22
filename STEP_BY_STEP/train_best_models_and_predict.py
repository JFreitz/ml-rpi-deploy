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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    out = out.sort_values(["date", "plant_system", "plant_no"]).reset_index(drop=True)

    out["day_of_year"] = out["date"].dt.dayofyear
    out["day_of_week_num"] = out["date"].dt.dayofweek
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month

    # Cyclic encoding for day-of-year to capture seasonality-like periodicity.
    out["doy_sin"] = np.sin(2.0 * np.pi * out["day_of_year"] / 366.0)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["day_of_year"] / 366.0)

    out["plant_no"] = pd.to_numeric(out["plant_no"], errors="coerce")
    return out


def get_models():
    return [
        ("LinearRegression", LinearRegression()),
        ("Ridge_a0.1", Ridge(alpha=0.1, random_state=SEED)),
        ("Ridge_a1.0", Ridge(alpha=1.0, random_state=SEED)),
        ("Ridge_a10.0", Ridge(alpha=10.0, random_state=SEED)),
        (
            "RandomForest_400",
            RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=1,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTrees_500",
            ExtraTreesRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=1,
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
        (
            "GradientBoosting_400",
            GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                random_state=SEED,
            ),
        ),
    ]


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def evaluate_cv(X, y, preprocessor, model):
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
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


def main():
    root = Path(__file__).resolve().parent
    train_path = root / "training_data.csv"
    test_path = root / "test_data.csv"
    infer_path = root / "test_this.csv"

    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    train_df = add_features(pd.read_csv(train_path))
    test_df = add_features(pd.read_csv(test_path))
    infer_df = add_features(pd.read_csv(infer_path))

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
        "plant_system",
    ]

    numeric_features = [
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
    categorical_features = ["plant_system"]

    comparison_rows = []
    best_rows = []
    metrics_rows = []

    pred_df = infer_df[["date", "day", "plant_system", "plant_no"]].copy()

    print("=" * 88)
    print("STEP_BY_STEP MODEL TRAINING")
    print("=" * 88)
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Inference rows (test_this): {len(infer_df)}")

    for target in TARGETS:
        print("\n" + "-" * 88)
        print(f"Target: {target}")

        y_train = pd.to_numeric(train_df[target], errors="coerce")
        y_test = pd.to_numeric(test_df[target], errors="coerce")

        train_mask = y_train.notna()
        test_mask = y_test.notna()

        X_train = train_df.loc[train_mask, feature_cols]
        y_train = y_train.loc[train_mask]
        X_test = test_df.loc[test_mask, feature_cols]
        y_test = y_test.loc[test_mask]

        preprocessor = build_preprocessor(numeric_features, categorical_features)

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
            test_r2 = float(r2_score(y_test, test_pred))
            test_rmse = rmse(y_test, test_pred)
            test_mae = float(mean_absolute_error(y_test, test_pred))
            test_mape = mape(y_test, test_pred)

            row = {
                "target": target,
                "model": model_name,
                **cv_scores,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_mape_percent": test_mape,
            }
            comparison_rows.append(row)

            if best is None:
                best = {
                    "model_name": model_name,
                    "model": model,
                    "cv": cv_scores,
                    "test_r2": test_r2,
                    "test_rmse": test_rmse,
                    "test_mae": test_mae,
                    "test_mape": test_mape,
                }
            else:
                better_r2 = test_r2 > best["test_r2"]
                tie_break = test_r2 == best["test_r2"] and test_rmse < best["test_rmse"]
                if better_r2 or tie_break:
                    best = {
                        "model_name": model_name,
                        "model": model,
                        "cv": cv_scores,
                        "test_r2": test_r2,
                        "test_rmse": test_rmse,
                        "test_mae": test_mae,
                        "test_mape": test_mape,
                    }

            print(
                f"  {model_name:22} CV_R2={cv_scores['cv_r2']:.4f} "
                f"TEST_R2={test_r2:.4f}"
            )

        final_pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", best["model"]),
            ]
        )
        final_pipe.fit(X_train, y_train)

        model_path = models_dir / f"{target}_best_model.joblib"
        dump(final_pipe, model_path)

        # Predict on test_data for final chosen model metrics.
        final_test_pred = final_pipe.predict(X_test)

        best_rows.append(
            {
                "target": target,
                "best_model": best["model_name"],
                "cv_r2": best["cv"]["cv_r2"],
                "cv_rmse": best["cv"]["cv_rmse"],
                "cv_mae": best["cv"]["cv_mae"],
                "test_r2": float(r2_score(y_test, final_test_pred)),
                "test_rmse": rmse(y_test, final_test_pred),
                "test_mae": float(mean_absolute_error(y_test, final_test_pred)),
                "test_mape_percent": mape(y_test, final_test_pred),
                "model_path": str(model_path.resolve()),
            }
        )

        metrics_rows.append(
            {
                "target": target,
                "r2": float(r2_score(y_test, final_test_pred)),
                "mae": float(mean_absolute_error(y_test, final_test_pred)),
                "rmse": rmse(y_test, final_test_pred),
                "mape_percent": mape(y_test, final_test_pred),
            }
        )

        X_infer = infer_df[feature_cols]
        pred_df[f"{target}_pred"] = final_pipe.predict(X_infer)
        print(
            f"  BEST: {best['model_name']} | TEST_R2={best['test_r2']:.4f} "
            f"TEST_RMSE={best['test_rmse']:.4f}"
        )

    comparison_df = pd.DataFrame(comparison_rows)
    best_df = pd.DataFrame(best_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    comparison_df["is_best_for_target"] = False
    for _, r in best_df.iterrows():
        mask = (comparison_df["target"] == r["target"]) & (comparison_df["model"] == r["best_model"])
        comparison_df.loc[mask, "is_best_for_target"] = True

    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y-%m-%d")

    comparison_df.to_csv(root / "model_comparison_all_models.csv", index=False)
    best_df.to_csv(root / "best_models_summary.csv", index=False)
    metrics_df.to_csv(root / "accuracy_metrics.csv", index=False)
    pred_df.to_csv(root / "test_this_predictions.csv", index=False)

    with (root / "training_report.txt").open("w", encoding="utf-8") as f:
        f.write("STEP_BY_STEP training report\n")
        f.write("=" * 50 + "\n")
        for _, r in best_df.iterrows():
            f.write(
                f"{r['target']}: {r['best_model']} | "
                f"CV_R2={r['cv_r2']:.4f}, TEST_R2={r['test_r2']:.4f}, "
                f"TEST_RMSE={r['test_rmse']:.4f}, TEST_MAE={r['test_mae']:.4f}\n"
            )

    print("\n" + "=" * 88)
    print("Training complete.")
    print("Saved files:")
    print("  - models/*.joblib")
    print("  - model_comparison_all_models.csv")
    print("  - best_models_summary.csv")
    print("  - accuracy_metrics.csv")
    print("  - test_this_predictions.csv")
    print("  - training_report.txt")
    print("=" * 88)


if __name__ == "__main__":
    main()
