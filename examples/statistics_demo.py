"""
Demonstration script showcasing c-mathlib's statistics and OLS regression engine in native C.
"""

from c_mathlib.statistics import (
    mean,
    variance,
    std_dev,
    median,
    quantile,
    covariance,
    correlation,
    linear_regression,
)


def main():
    print("=== c-mathlib Statistics & OLS Regression Demonstration ===")
    print("All statistical routines are executed in native C.\n")

    # Sample dataset: Student study hours vs exam scores
    study_hours = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    exam_scores = [52.0, 55.0, 60.0, 65.0, 68.0, 74.0, 79.0, 85.0, 88.0, 95.0]

    # 1. Descriptive Statistics
    print("--- 1. Descriptive Statistics (Study Hours) ---")
    print(f"Data: {study_hours}")
    print(f"Mean:              {mean(study_hours):.2f}")
    print(f"Sample Variance:   {variance(study_hours, ddof=1):.4f}")
    print(f"Sample Std Dev:    {std_dev(study_hours, ddof=1):.4f}")
    print(f"Median:            {median(study_hours):.2f}")
    print(f"25th Percentile (Q1): {quantile(study_hours, 0.25):.2f}")
    print(f"75th Percentile (Q3): {quantile(study_hours, 0.75):.2f}")
    print(f"Interquartile Range:  {quantile(study_hours, 0.75) - quantile(study_hours, 0.25):.2f}")

    print("\n--- 2. Descriptive Statistics (Exam Scores) ---")
    print(f"Data: {exam_scores}")
    print(f"Mean:              {mean(exam_scores):.2f}")
    print(f"Sample Std Dev:    {std_dev(exam_scores, ddof=1):.4f}")
    print(f"Median:            {median(exam_scores):.2f}")

    # 2. Bivariate Statistics
    print("\n--- 3. Bivariate Association ---")
    cov = covariance(study_hours, exam_scores, ddof=1)
    r = correlation(study_hours, exam_scores)
    print(f"Sample Covariance:       {cov:.4f}")
    print(f"Pearson Correlation (r): {r:.6f}")

    # 3. Simple Linear Regression (OLS)
    print("\n--- 4. Ordinary Least Squares (OLS) Linear Regression ---")
    reg = linear_regression(study_hours, exam_scores)
    print(f"Regression Result: {reg}")
    print(f"  Slope (beta_1):     {reg.slope:.4f}")
    print(f"  Intercept (beta_0): {reg.intercept:.4f}")
    print(f"  R-squared (R^2):    {reg.r_squared:.6f} ({reg.r_squared * 100:.2f}% of score variance explained)")
    print(f"  Fitted Model:       Score = {reg.intercept:.2f} + {reg.slope:.2f} * Hours")

    # Predict exam scores for new study times
    future_hours = [0.0, 3.5, 7.5, 12.0]
    print("\nPredictions for new students:")
    for h in future_hours:
        pred = reg.predict(h)
        print(f"  Study {h:4.1f} hours -> Predicted Score: {pred:5.2f}")


if __name__ == "__main__":
    main()
