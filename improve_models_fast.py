from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGETS = ["height", "length", "weight", "leaves", "branches"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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


def score_date_cv(X, y, dates, preprocessor, model, n_splits=4):
    unique_dates = np.array(sorted(pd.Series(dates).unique()))
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


def model_candidates():
    return [
        ("ExtraTrees_300", ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)),
        (
            "ExtraTrees_500_depth16",
            ExtraTreesRegressor(
                n_estimators=500,
                max_depth=16,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "ExtraTrees_700_depth20",
            ExtraTreesRegressor(
                n_estimators=700,
                max_depth=20,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "RandomForest_400",
            RandomForestRegressor(
                n_estimators=400,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "RandomForest_600_depth18",
            RandomForestRegressor(
                n_estimators=600,
                max_depth=18,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "GradientBoosting_600",
            GradientBoostingRegressor(
                n_estimators=600,
                learning_rate=0.04,
                max_depth=3,
                subsample=0.9,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "GradientBoosting_900",
            GradientBoostingRegressor(
                n_estimators=900,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
        ),
        ("Ridge_0.1", Ridge(alpha=0.1, random_state=RANDOM_STATE)),
        ("Ridge_1.0", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        (
            "ElasticNet_weak",
            ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=5000, random_state=RANDOM_STATE),
        ),
    ]


def main():
    root = Path(".")
    train = add_features(pd.read_csv(root / "training_data.csv"))
    test = add_features(pd.read_csv(root / "test_data.csv"))
    test_this = add_features(pd.read_csv(root / "test_this.csv"))

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

    models_dir = root / "models_improved"
    models_dir.mkdir(exist_ok=True)

    summary_rows = []
    compare_rows = []
    pred_df = test_this[["date", "day", "plant_system", "plant_no"]].copy()

    print("=" * 90)
    print("FAST IMPROVEMENT TUNING (DATE-AWARE)")
    print("=" * 90)

    for target in TARGETS:
        print(f"\nTarget: {target}")
        y_train = pd.to_numeric(train[target], errors="coerce")
        y_test = pd.to_numeric(test[target], errors="coerce")

        tr_mask = y_train.notna()
        te_mask = y_test.notna()

        X_train = train.loc[tr_mask, feature_cols]
        y_train = y_train.loc[tr_mask]
        tr_dates = train.loc[tr_mask, "date"]

        best = None

        for name, model in model_candidates():
            cv = score_date_cv(X_train, y_train, tr_dates, preprocessor, model)
            compare_rows.append({"target": target, "model": name, **cv})

            if best is None:
                best = {"name": name, "model": model, **cv}
            else:
                better = cv["cv_r2"] > best["cv_r2"]
                tie = cv["cv_r2"] == best["cv_r2"] and cv["cv_rmse"] < best["cv_rmse"]
                if better or tie:
                    best = {"name": name, "model": model, **cv}

            print(f"  {name:28} CV_R2={cv['cv_r2']:.4f} CV_RMSE={cv['cv_rmse']:.4f}")

        pipe = Pipeline(steps=[("prep", preprocessor), ("model", best["model"])])
        pipe.fit(X_train, y_train)

        model_path = models_dir / f"{target}_best_model.joblib"
        dump(pipe, model_path)

        row = {
            "target": target,
            "best_model": best["name"],
            "cv_r2": best["cv_r2"],
            "cv_rmse": best["cv_rmse"],
            "cv_mae": best["cv_mae"],
            "model_path": str(model_path.resolve()),
        }

        if te_mask.sum() > 0:
            X_test = test.loc[te_mask, feature_cols]
            y_test_valid = y_test.loc[te_mask]
            pred_test = pipe.predict(X_test)
            row["test_r2"] = float(r2_score(y_test_valid, pred_test))
            row["test_rmse"] = rmse(y_test_valid, pred_test)
            row["test_mae"] = float(mean_absolute_error(y_test_valid, pred_test))

        summary_rows.append(row)
        pred_df[f"{target}_pred"] = pipe.predict(test_this[feature_cols])

        print(
            f"  BEST -> {best['name']} | CV_R2={row['cv_r2']:.4f} | TEST_R2={row.get('test_r2', np.nan):.4f}"
        )

    pd.DataFrame(compare_rows).to_csv(root / "model_comparison_improved.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(root / "best_models_summary_improved.csv", index=False)

    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y-%m-%d")
    pred_df.to_csv(root / "test_this_predictions_improved.csv", index=False)

    with (root / "training_report_improved.txt").open("w", encoding="utf-8") as f:
        f.write("Improved model summary\n")
        f.write("=" * 30 + "\n")
        for r in summary_rows:
            f.write(
                f"{r['target']}: {r['best_model']} | CV_R2={r['cv_r2']:.4f} | TEST_R2={r.get('test_r2', np.nan):.4f}\n"
            )

    print("\nCompleted improved tuning.")


if __name__ == "__main__":
    main()
