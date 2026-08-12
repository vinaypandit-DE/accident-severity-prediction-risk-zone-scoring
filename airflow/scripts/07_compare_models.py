import os
import pandas as pd

metrics_dir = "../metrics"
reports_dir = "../reports"

os.makedirs(reports_dir, exist_ok=True)


def read_metrics(file_path):

    metrics = {}

    with open(file_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)

            metrics[key.strip()] = float(value.strip())

    return metrics


rf_metrics = read_metrics(os.path.join(metrics_dir, "model_metrics.txt"))
xgb_metrics = read_metrics(os.path.join(metrics_dir, "xgb_metrics.txt"))

comparison = pd.DataFrame({
    "Metric": list(rf_metrics.keys()),
    "Random Forest": list(rf_metrics.values()),
    "XGBoost": [xgb_metrics[k] for k in rf_metrics.keys()]
})

comparison["Better Model"] = comparison.apply(
    lambda row: "Random Forest"
    if row["Random Forest"] > row["XGBoost"]
    else "XGBoost",
    axis=1,
)

comparison.to_csv(
    os.path.join(reports_dir, "model_comparison.csv"),
    index=False,
)

rf_avg = comparison["Random Forest"].mean()
xgb_avg = comparison["XGBoost"].mean()

best_model = "Random Forest" if rf_avg > xgb_avg else "XGBoost"

with open(os.path.join(reports_dir, "model_comparison_report.txt"), "w") as f:
    f.write("MODEL COMPARISON REPORT\n")
    f.write("=" * 40 + "\n\n")
    f.write(comparison.to_string(index=False))
    f.write("\n\n")
    f.write(f"Average Random Forest Score : {rf_avg:.4f}\n")
    f.write(f"Average XGBoost Score       : {xgb_avg:.4f}\n")
    f.write(f"\nBest Overall Model : {best_model}\n")

print(comparison)
print("\nBest Overall Model :", best_model)