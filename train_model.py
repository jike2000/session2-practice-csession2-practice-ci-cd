from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


def main():
    project_dir = Path(__file__).resolve().parent
    training_data = pd.read_csv(project_dir / "training_data.csv")

    model = LinearRegression().fit(
        training_data[["YearsExperience"]], training_data["Salary"]
    )
    joblib.dump(model, project_dir / "linear_model.pkl")

    coefficients = (
        f"Coefficients: [{model.coef_[0]:.6f}]\n"
        f"Intercept: {model.intercept_:.6f}\n"
    )
    (project_dir / "linear_model.txt").write_text(coefficients, encoding="utf-8")
    print(f"Trained on {len(training_data)} rows.")
    print(coefficients, end="")


if __name__ == "__main__":
    main()
