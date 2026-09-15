import unittest
import math
from c_mathlib.statistics import (
    mean,
    variance,
    std_dev,
    median,
    quantile,
    covariance,
    correlation,
    linear_regression,
    LinearRegressionResult,
)


class TestStatistics(unittest.TestCase):
    """Unit tests for c-mathlib statistics and regression module."""

    # --------------------------------------------------------------------------
    # Descriptive Statistics Tests
    # --------------------------------------------------------------------------

    def test_mean(self):
        """Test arithmetic mean."""
        self.assertAlmostEqual(mean([1, 2, 3, 4, 5]), 3.0)
        self.assertAlmostEqual(mean([10, -10]), 0.0)
        self.assertAlmostEqual(mean([42]), 42.0)

        with self.assertRaises(ValueError):
            mean([])
        with self.assertRaises(TypeError):
            mean("123")  # type: ignore

    def test_variance_and_std_dev(self):
        """Test sample and population variance and standard deviation."""
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        # Population mean = 5.0
        # Sum of squared diffs: (2-5)^2 + 3*(4-5)^2 + 2*(5-5)^2 + (7-5)^2 + (9-5)^2
        # = 9 + 3 + 0 + 4 + 16 = 32
        # Population var (ddof=0): 32 / 8 = 4.0, std = 2.0
        # Sample var (ddof=1): 32 / 7 = 4.57142857, std = 2.138089935
        self.assertAlmostEqual(variance(data, ddof=0), 4.0)
        self.assertAlmostEqual(std_dev(data, ddof=0), 2.0)

        self.assertAlmostEqual(variance(data, ddof=1), 32.0 / 7.0)
        self.assertAlmostEqual(std_dev(data, ddof=1), math.sqrt(32.0 / 7.0))

        with self.assertRaises(ValueError):
            variance([1], ddof=1)

    def test_median(self):
        """Test median with odd, even, and unsorted datasets."""
        self.assertAlmostEqual(median([1, 3, 5]), 3.0)
        self.assertAlmostEqual(median([5, 1, 3]), 3.0)
        self.assertAlmostEqual(median([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(median([4, 1, 3, 2]), 2.5)
        self.assertAlmostEqual(median([7]), 7.0)

    def test_quantile(self):
        """Test quantile calculation with linear interpolation."""
        data = [10, 20, 30, 40, 50]
        self.assertAlmostEqual(quantile(data, 0.0), 10.0)
        self.assertAlmostEqual(quantile(data, 0.5), 30.0)
        self.assertAlmostEqual(quantile(data, 1.0), 50.0)
        self.assertAlmostEqual(quantile(data, 0.25), 20.0)
        self.assertAlmostEqual(quantile(data, 0.75), 40.0)

        with self.assertRaises(ValueError):
            quantile(data, -0.1)
        with self.assertRaises(ValueError):
            quantile(data, 1.1)

    # --------------------------------------------------------------------------
    # Bivariate & Regression Tests
    # --------------------------------------------------------------------------

    def test_covariance(self):
        """Test covariance between two variables."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        # y = 2x => cov(x, 2x) = 2 * var(x) = 2 * 2.5 = 5.0 (sample)
        self.assertAlmostEqual(covariance(x, y, ddof=1), 5.0)
        self.assertAlmostEqual(covariance(x, y, ddof=0), 4.0)

        with self.assertRaises(ValueError):
            covariance([1, 2], [1, 2, 3])  # Length mismatch

    def test_correlation(self):
        """Test Pearson correlation coefficient."""
        x = [1, 2, 3, 4, 5]
        y_pos = [2, 4, 6, 8, 10]
        y_neg = [10, 8, 6, 4, 2]
        self.assertAlmostEqual(correlation(x, y_pos), 1.0)
        self.assertAlmostEqual(correlation(x, y_neg), -1.0)

        # Uncorrelated
        x_uncorr = [-1, 0, 1]
        y_uncorr = [1, 0, 1]
        self.assertAlmostEqual(correlation(x_uncorr, y_uncorr), 0.0)

    def test_linear_regression(self):
        """Test Ordinary Least Squares (OLS) linear regression."""
        # Exact line: y = 3x - 2
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 4.0, 7.0, 10.0, 13.0]

        res = linear_regression(x, y)
        self.assertIsInstance(res, LinearRegressionResult)
        self.assertAlmostEqual(res.slope, 3.0)
        self.assertAlmostEqual(res.intercept, -2.0)
        self.assertAlmostEqual(res.r_squared, 1.0)

        # Predict
        self.assertAlmostEqual(res.predict(6.0), 16.0)
        self.assertAlmostEqual(res.predict(0.0), -2.0)
        self.assertIn("LinearRegression", repr(res))

        # Zero variance in x raises ValueError
        with self.assertRaises(ValueError):
            linear_regression([2, 2, 2], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
