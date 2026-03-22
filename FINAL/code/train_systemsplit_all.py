import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_pipe(model, features):
    pre = ColumnTransformer([("num", StandardScaler(), features)])
    return Pipeline([("preprocess", pre), ("model", model)])


def main():
    root = Path(".")
    train = pd.read_csv(root / "training_data.csv")
    test = pd.read_csv(root / "test_data.csv")
    test_this = pd.read_csv(root / "test_this.csv")

    for df in (train, test, test_this):
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df["day_of_year"] = df["date"].dt.dayofyear
        df["day_of_week_num"] = df["date"].dt.dayofweek
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    features = [
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
    targets = ["height", "length", "weight", "leaves", "branches"]
    systems = ["AERO", "DWC"]

    candidates = {
        "ExtraTrees_250": ExtraTreesRegressor(n_estimators=250, random_state=42, n_jobs=1),
        "GradientBoosting_300": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "Ridge": Ridge(alpha=1.0, random_state=42),
    }

    models_dir = root / "models_systemsplit_all"
    models_dir.mkdir(exist_ok=True)

    model_rows = []
    best_rows = []

    for system in systems:
        tr_sys = train[train["plant_system"] == system].copy()
        te_sys = test[test["plant_system"] == system].copy()
        Xtr = tr_sys[features]
        Xte = te_sys[features]

        for target in targets:
            ytr = tr_sys[target].astype(float)
            yte = te_sys[target].astype(float)

            best = None
            for model_name, model in candidates.items():
                pipe = make_pipe(model, features)
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)

                row = {
                    "system": system,
                    "target": target,
                    "model": model_name,
                    "test_r2": float(r2_score(yte, pred)),
                    "test_mae": float(mean_absolute_error(yte, pred)),
                    "test_rmse": rmse(yte, pred),
                }
                model_rows.append(row)

                if best is None or row["test_r2"] > best["test_r2"] or (
                    row["test_r2"] == best["test_r2"] and row["test_rmse"] < best["test_rmse"]
                ):
                    best = {**row, "pipe": pipe, "best_model": model_name}

            model_path = models_dir / f"{target}_{system.lower()}_best_model.joblib"
            joblib.dump(best["pipe"], model_path)
            best_rows.append(
                {
                    "system": system,
                    "target": target,
                    "best_model": best["best_model"],
                    "test_r2": best["test_r2"],
                    "test_mae": best["test_mae"],
                    "test_rmse": best["test_rmse"],
                    "model_path": str(model_path.resolve()),
                }
            )

    model_cmp = pd.DataFrame(model_rows)
    best_df = pd.DataFrame(best_rows)

    model_cmp.to_csv(root / "systemsplit_all_model_comparison.csv", index=False)
    best_df.to_csv(root / "systemsplit_all_best_models_summary.csv", index=False)

    pred_out = test_this[["date", "day", "plant_system", "plant_no"]].copy()
    for target in targets:
        pred_out[f"{target}_pred"] = np.nan

    for system in systems:
        mask = pred_out["plant_system"].eq(system)
        Xp = test_this.loc[mask, features]
        for target in targets:
            model_path = best_df[
                (best_df["system"] == system) & (best_df["target"] == target)
            ]["model_path"].iloc[0]
            model = joblib.load(model_path)
            pred_out.loc[mask, f"{target}_pred"] = model.predict(Xp)

    pred_out["date"] = pd.to_datetime(pred_out["date"]).dt.strftime("%Y-%m-%d")
    pred_out.to_csv(root / "test_this_predictions_systemsplit_all.csv", index=False)

    actual = pd.read_csv(root / "test_data.csv")
    merged = actual.merge(pred_out, on=["date", "day", "plant_system", "plant_no"], how="inner")

    metrics = []
    for target in targets:
        y = merged[target].astype(float)
        p = merged[f"{target}_pred"].astype(float)
        metrics.append(
            {
                "target": target,
                "r2": float(r2_score(y, p)),
                "mae": float(mean_absolute_error(y, p)),
                "rmse": rmse(y, p),
            }
        )
    pd.DataFrame(metrics).to_csv(root / "accuracy_metrics_systemsplit_all.csv", index=False)

    print("Generated systemsplit_all outputs.")


if __name__ == "__main__":
    main()
