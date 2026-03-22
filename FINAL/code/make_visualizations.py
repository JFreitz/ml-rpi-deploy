from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    pred_dir = root / "predictions"
    report_dir = root / "reports"
    viz_dir = report_dir / "visualizations"
    ensure_dir(viz_dir)

    train = pd.read_csv(data_dir / "training_data.csv")
    test = pd.read_csv(data_dir / "test_data.csv")
    full = pd.read_csv(data_dir / "COMBINED_SENSOR_PLANT_DATA_WITH_LENGTH.csv")
    final_pred = pd.read_csv(pred_dir / "final_predictions.csv")
    metrics = pd.read_csv(report_dir / "accuracy_metrics_final.csv")

    for df in (train, test, full, final_pred):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    targets = ["height", "length", "weight", "leaves", "branches"]
    sensors = ["ave_ph", "ave_do", "ave_tds", "ave_temp", "ave_humidity"]

    # 1) Split timeline (rows per day)
    tr_counts = train.groupby("date").size()
    te_counts = test.groupby("date").size()
    plt.figure(figsize=(10, 4))
    plt.plot(tr_counts.index, tr_counts.values, label="train", linewidth=2)
    plt.plot(te_counts.index, te_counts.values, label="test", linewidth=2)
    plt.title("Train/Test Rows Per Date")
    plt.xlabel("Date")
    plt.ylabel("Rows")
    plt.legend()
    save_fig(viz_dir / "01_split_rows_per_date.png")

    # 2) Sensor trends over time
    daily = full.groupby("date")[sensors].mean().reset_index()
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()
    for i, col in enumerate(sensors):
        axes[i].plot(daily["date"], daily[col], linewidth=1.8)
        axes[i].set_title(col)
        axes[i].set_ylabel("mean")
    axes[-1].axis("off")
    fig.suptitle("Daily Mean Sensor Trends")
    save_fig(viz_dir / "02_sensor_trends.png")

    # 3) Target distributions (train vs test)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))
    axes = axes.flatten()
    for i, t in enumerate(targets):
        axes[i].hist(train[t], bins=16, alpha=0.6, label="train")
        axes[i].hist(test[t], bins=16, alpha=0.6, label="test")
        axes[i].set_title(t)
        axes[i].legend()
    axes[-1].axis("off")
    fig.suptitle("Target Distributions: Train vs Test")
    save_fig(viz_dir / "03_target_distributions_train_vs_test.png")

    # 4) Correlation heatmap (numeric)
    num_cols = sensors + ["plant_no"] + targets
    corr = full[num_cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap (Numeric Features/Targets)")
    save_fig(viz_dir / "04_correlation_heatmap.png")

    # 5) Final metric bars (R2)
    plt.figure(figsize=(8, 4))
    plt.bar(metrics["target"], metrics["r2"])
    plt.title("Final Model Accuracy by Target (R2)")
    plt.xlabel("Target")
    plt.ylabel("R2")
    save_fig(viz_dir / "05_final_r2_by_target.png")

    # 6) Final metric bars (MAE/RMSE)
    x = np.arange(len(metrics["target"]))
    w = 0.35
    plt.figure(figsize=(9, 4))
    plt.bar(x - w / 2, metrics["mae"], width=w, label="MAE")
    plt.bar(x + w / 2, metrics["rmse"], width=w, label="RMSE")
    plt.xticks(x, metrics["target"])
    plt.title("Final Error by Target")
    plt.xlabel("Target")
    plt.ylabel("Error")
    plt.legend()
    save_fig(viz_dir / "06_final_error_by_target.png")

    # Merge actual and final predictions
    merged = test.merge(final_pred, on=["date", "day", "plant_system", "plant_no"], how="inner")

    # 7) Actual vs predicted scatter
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))
    axes = axes.flatten()
    for i, t in enumerate(targets):
        y = merged[t].astype(float)
        p = merged[f"{t}_pred"].astype(float)
        axes[i].scatter(y, p, s=14, alpha=0.65)
        lo = min(y.min(), p.min())
        hi = max(y.max(), p.max())
        axes[i].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        axes[i].set_title(t)
        axes[i].set_xlabel("actual")
        axes[i].set_ylabel("predicted")
    axes[-1].axis("off")
    fig.suptitle("Actual vs Predicted (Final Mixed Output)")
    save_fig(viz_dir / "07_actual_vs_predicted_scatter.png")

    # 8) Residual distributions
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))
    axes = axes.flatten()
    for i, t in enumerate(targets):
        residual = merged[f"{t}_pred"].astype(float) - merged[t].astype(float)
        axes[i].hist(residual, bins=20, alpha=0.75)
        axes[i].axvline(0, linestyle="--", linewidth=1)
        axes[i].set_title(t)
        axes[i].set_xlabel("residual")
    axes[-1].axis("off")
    fig.suptitle("Residual Distribution by Target")
    save_fig(viz_dir / "08_residual_distributions.png")

    print(f"Visualization folder: {viz_dir}")
    print("Created 8 PNG files.")


if __name__ == "__main__":
    main()
