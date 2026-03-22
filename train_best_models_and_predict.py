import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS = ["height", "length", "weight", "leaves", "branches"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], format="%Y-%m-%d", errors="coerce")
    out = out.sort_values(["date", "plant_system", "plant_no"]).reset_index(drop=True)
    out["day_of_year"] = out["date"].dt.dayofyear
    out["day_of_week_num"] = out["date"].dt.dayofweek
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    return out


def get_model_candidates():
    return [
        ("LinearRegression", LinearRegression()),
        ("Ridge_a0.1", Ridge(alpha=0.1, random_state=42)),
        ("Ridge_a1.0", Ridge(alpha=1.0, random_state=42)),
        ("Ridge_a10.0", Ridge(alpha=10.0, random_state=42)),
        (
            "RandomForest_300",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTrees_300",
            ExtraTreesRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "GradientBoosting_300",
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
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


def evaluate_with_date_cv(X, y, dates, preprocessor, model, n_splits=4):
    unique_dates = np.array(sorted(pd.Series(dates).unique()))
    if len(unique_dates) < 3:
        return {"cv_r2": np.nan, "cv_rmse": np.nan, "cv_mae": np.nan}

    folds = min(n_splits, len(unique_dates) - 1)
    splitter = TimeSeriesSplit(n_splits=folds)

    r2_scores = []
    rmse_scores = []
    mae_scores = []

    for train_idx, val_idx in splitter.split(unique_dates):
        train_dates = set(unique_dates[train_idx])
        val_dates = set(unique_dates[val_idx])

        train_mask = pd.Series(dates).isin(train_dates).to_numpy()
        val_mask = pd.Series(dates).isin(val_dates).to_numpy()

        X_train_fold = X.loc[train_mask]
        y_train_fold = y.loc[train_mask]
        X_val_fold = X.loc[val_mask]
        y_val_fold = y.loc[val_mask]

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        pipe.fit(X_train_fold, y_train_fold)
        y_pred = pipe.predict(X_val_fold)

        r2_scores.append(float(r2_score(y_val_fold, y_pred)))
        rmse_scores.append(rmse(y_val_fold, y_pred))
        mae_scores.append(float(mean_absolute_error(y_val_fold, y_pred)))

    return {
        "cv_r2": float(np.mean(r2_scores)),
        "cv_rmse": float(np.mean(rmse_scores)),
        "cv_mae": float(np.mean(mae_scores)),
    }


def main():
    root = Path(".")
    training_path = root / "training_data.csv"
    test_path = root / "test_data.csv"
    test_this_path = root / "test_this.csv"

    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    df_train = add_date_features(pd.read_csv(training_path))
    df_test = add_date_features(pd.read_csv(test_path))
    df_test_this = add_date_features(pd.read_csv(test_this_path))

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
    ]
    categorical_features = ["plant_system"]

    df_train["plant_no"] = pd.to_numeric(df_train["plant_no"], errors="coerce")
    df_test["plant_no"] = pd.to_numeric(df_test["plant_no"], errors="coerce")
    df_test_this["plant_no"] = pd.to_numeric(df_test_this["plant_no"], errors="coerce")

    comparison_rows = []
    summary_rows = []

    base_pred = df_test_this[["date", "day", "plant_system", "plant_no"]].copy()

    print("=" * 90)
    print("DATE-AWARE TRAINING FOR PLANT PARAMETER PREDICTION")
    print("=" * 90)
    print(f"Training rows: {len(df_train)}")
    print(f"Test rows: {len(df_test)}")
    print(f"Prediction input rows (test_this): {len(df_test_this)}")

    for target in TARGETS:
        print("\n" + "-" * 90)
        print(f"Target: {target}")

        y_train = pd.to_numeric(df_train[target], errors="coerce")
        y_test = pd.to_numeric(df_test[target], errors="coerce")

        train_mask = y_train.notna()
        test_mask = y_test.notna()

        X_train = df_train.loc[train_mask, feature_cols]
        y_train = y_train.loc[train_mask]
        dates_train = df_train.loc[train_mask, "date"]

        preprocessor = build_preprocessor(numeric_features, categorical_features)

        best = None

        for model_name, model in get_model_candidates():
            cv_metrics = evaluate_with_date_cv(
                X_train,
                y_train,
                dates_train,
                preprocessor,
                model,
                n_splits=4,
            )

            row = {
                "target": target,
                "model": model_name,
                **cv_metrics,
            }
            comparison_rows.append(row)

            if best is None:
                best = {"model_name": model_name, "model": model, **cv_metrics}
            else:
                better_r2 = row["cv_r2"] > best["cv_r2"]
                tie_break = row["cv_r2"] == best["cv_r2"] and row["cv_rmse"] < best["cv_rmse"]
                if better_r2 or tie_break:
                    best = {"model_name": model_name, "model": model, **cv_metrics}

            print(
                f"  {model_name:22} CV_R2={cv_metrics['cv_r2']:.4f} "
                f"CV_RMSE={cv_metrics['cv_rmse']:.4f}"
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

        target_summary = {
            "target": target,
            "best_model": best["model_name"],
            "cv_r2": best["cv_r2"],
            "cv_rmse": best["cv_rmse"],
            "cv_mae": best["cv_mae"],
            "model_path": str(model_path.resolve()),
        }

        if test_mask.sum() > 0:
            X_test = df_test.loc[test_mask, feature_cols]
            y_test_valid = y_test.loc[test_mask]
            y_test_pred = final_pipe.predict(X_test)
            target_summary.update(
                {
                    "test_r2": float(r2_score(y_test_valid, y_test_pred)),
                    "test_rmse": rmse(y_test_valid, y_test_pred),
                    "test_mae": float(mean_absolute_error(y_test_valid, y_test_pred)),
                }
            )
            print(
                f"  BEST: {best['model_name']} | TEST_R2={target_summary['test_r2']:.4f} "
                f"TEST_RMSE={target_summary['test_rmse']:.4f}"
            )

        summary_rows.append(target_summary)

        X_pred = df_test_this[feature_cols]
        base_pred[f"{target}_pred"] = final_pipe.predict(X_pred)

    comparison_df = pd.DataFrame(comparison_rows)
    summary_df = pd.DataFrame(summary_rows)

    comparison_df.to_csv(root / "model_comparison_date_aware.csv", index=False)
    summary_df.to_csv(root / "best_models_summary.csv", index=False)

    base_pred["date"] = pd.to_datetime(base_pred["date"]).dt.strftime("%Y-%m-%d")
    base_pred.to_csv(root / "test_this_predictions.csv", index=False)

    with (root / "training_report.txt").open("w", encoding="utf-8") as f:
        f.write("Date-aware best model training report\n")
        f.write("=" * 45 + "\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"{row['target']}: {row['best_model']} | "
                f"CV_R2={row['cv_r2']:.4f}, CV_RMSE={row['cv_rmse']:.4f}, "
                f"TEST_R2={row.get('test_r2', np.nan):.4f}, "
                f"TEST_RMSE={row.get('test_rmse', np.nan):.4f}\n"
            )

    print("\n" + "=" * 90)
    print("Training complete.")
    print("Saved files:")
    print("  - models/*.joblib")
    print("  - model_comparison_date_aware.csv")
    print("  - best_models_summary.csv")
    print("  - test_this_predictions.csv")
    print("  - training_report.txt")
    print("=" * 90)


if __name__ == "__main__":
    main()
