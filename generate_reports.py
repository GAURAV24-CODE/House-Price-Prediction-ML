import pandas as pd

from src.report_generator import (
    generate_r2_chart,
    generate_rmse_chart,
    generate_mae_chart,
    generate_training_time_chart,
    generate_prediction_time_chart,
)

results = pd.read_csv(
    "reports/model_comparison/model_results.csv"
)

generate_r2_chart(results)

generate_rmse_chart(results)

generate_mae_chart(results)

generate_training_time_chart(results)

generate_prediction_time_chart(results)

print("Charts generated successfully!")