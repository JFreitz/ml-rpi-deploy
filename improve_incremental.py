from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS = ["height", "length", "weight", "leaves", "branches"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_features(df):
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


def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def date_cv(X, y, dates, preprocessor, model):
    unique_dates = np.array(sorted(pd.Series(dates).unique()))
    splits = min(4, len(unique_dates) - 1)
    tscv = TimeSeriesSplit(n_splits=splits)

    r2s = []
    rmses = []
    maes = []

    for tr, va in tscv.split(unique_dates):
        tr_dates = set(unique_dates[tr])
        va_dates = set(unique_dates[va])

        tr_mask = pd.Series(dates).isin(tr_dates).to_numpy()
        va_mask = pd.Series(dates).isin(va_dates).to_numpy()

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model),
        ])
        pipe.fit(X.loc[tr_mask], y.loc[tr_mask])
        pred = pipe.predict(X.loc[va_mask])

        yv = y.loc[va_mask]
        r2s.append(float(r2_score(yv, pred)))
        rmses.append(rmse(yv, pred))
        maes.append(float(mean_absolute_error(yv, pred)))

    return {
        "cv_r2": float(np.mean(r2s)),
        "cv_rmse": float(np.mean(rmses)),
        "cv_mae": float(np.mean(maes)),
    }


def candidates():
    return [
        ("ExtraTrees_300", ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=1)),
        (
            "ExtraTrees_500_depth16",
            ExtraTreesRegressor(n_estimators=500, max_depth=16, random_state=42, n_jobs=1),
        ),
        (
            "RandomForest_500",
            RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=1),
        ),
        (
            "RandomForest_700_depth18",
            RandomForestRegressor(n_estimators=700, max_depth=18, random_state=42, n_jobs=1),
        ),
        (
            "GradientBoosting_700",
            GradientBoostingRegressor(n_estimators=700, learning_rate=0.04, max_depth=3, subsample=0.9, random_state=42),
        ),
        ("Ridge_1.0", Ridge(alpha=1.0, random_state=42)),
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
    num_cols = [
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
    cat_cols = ["plant_system"]

    prep = build_preprocessor(num_cols, cat_cols)

    models_dir = root / "models_improved"
    models_dir.mkdir(exist_ok=True)

    pred_out = test_this[["date", "day", "plant_system", "plant_no"]].copy()
    if (root / "test_this_predictions_improved.csv").exists():
        old = pd.read_csv(root / "test_this_predictions_improved.csv")
        for c in old.columns:
            if c.endswith("_pred"):
                pred_out[c] = old[c]

    summary_rows = []
    if (root / "best_models_summary_improved.csv").exists():
        old_summary = pd.read_csv(root / "best_models_summary_improved.csv")
        summary_rows = old_summary.to_dict("records")

    summary_by_target = {r["target"]: r for r in summary_rows}

    print("Incremental improvement run")

    for target in TARGETS:
        print(f"\nTarget: {target}")
        y_tr = pd.to_numeric(train[target], errors="coerce")
        y_te = pd.to_numeric(test[target], errors="coerce")

        tr_mask = y_tr.notna()
        te_mask = y_te.notna()

        X_tr = train.loc[tr_mask, feature_cols]
        y_tr = y_tr.loc[tr_mask]
        d_tr = train.loc[tr_mask, "date"]

        best = None
        for name, model in candidates():
            cv = date_cv(X_tr, y_tr, d_tr, prep, model)
            print(f"  {name:26} CV_R2={cv['cv_r2']:.4f} CV_RMSE={cv['cv_rmse']:.4f}")
            if best is None:
                best = {"name": name, "model": model, **cv}
            else:
                if cv["cv_r2"] > best["cv_r2"] or (
                    cv["cv_r2"] == best["cv_r2"] and cv["cv_rmse"] < best["cv_rmse"]
                ):
                    best = {"name": name, "model": model, **cv}

        pipe = Pipeline([
            ("prep", prep),
            ("model", best["model"]),
        ])
        pipe.fit(X_tr, y_tr)

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
            X_te = test.loc[te_mask, feature_cols]
            y_te_valid = y_te.loc[te_mask]
            pred_te = pipe.predict(X_te)
            row["test_r2"] = float(r2_score(y_te_valid, pred_te))
            row["test_rmse"] = rmse(y_te_valid, pred_te)
            row["test_mae"] = float(mean_absolute_error(y_te_valid, pred_te))

        summary_by_target[target] = row

        pred_out[f"{target}_pred"] = pipe.predict(test_this[feature_cols])

        out_summary = pd.DataFrame([summary_by_target[t] for t in TARGETS if t in summary_by_target])
        out_summary.to_csv(root / "best_models_summary_improved.csv", index=False)
        pred_save = pred_out.copy()
        pred_save["date"] = pd.to_datetime(pred_save["date"]).dt.strftime("%Y-%m-%d")
        pred_save.to_csv(root / "test_this_predictions_improved.csv", index=False)

        print(
            f"  BEST -> {row['best_model']} | CV_R2={row['cv_r2']:.4f} | TEST_R2={row.get('test_r2', np.nan):.4f}"
        )

    with (root / "training_report_improved.txt").open("w", encoding="utf-8") as f:
        f.write("Improved summary\n")
        f.write("=" * 30 + "\n")
        for t in TARGETS:
            r = summary_by_target[t]
            f.write(
                f"{t}: {r['best_model']} | CV_R2={r['cv_r2']:.4f} | TEST_R2={r.get('test_r2', np.nan):.4f}\n"
            )

    print("\nImprovement run completed.")


if __name__ == "__main__":
    main()
