# skillcraft_task1_house_price.py
# Linear Regression on house prices with robust loading + visuals (matplotlib only)

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========= User settings =========
FILE_PATH = r"C:\Users\prana\OneDrive\Desktop\ml-1.csv"  # <- change if needed
TEST_SIZE = 0.2
RANDOM_STATE = 42
# =================================

def detect_columns(df):
    """
    Map common column names to: sqft_living, bedrooms, bathrooms, price.
    Returns a dict {std_name: actual_col_name}.
    """
    candidates = {
        "sqft_living": ["sqft_living", "square_footage", "square_feet", "sqft", "area", "size", "living_area"],
        "bedrooms": ["bedrooms", "beds", "no_bedrooms", "bed", "br"],
        "bathrooms": ["bathrooms", "baths", "no_bathrooms", "bath", "ba"],
        "price": ["price", "sale_price", "target", "label", "cost"]
    }
    lower_to_actual = {c.lower(): c for c in df.columns}
    cols_lower = set(lower_to_actual.keys())

    def pick(options):
        for name in options:
            if name in cols_lower:
                return lower_to_actual[name]
        return None

    mapping = {std: pick(opts) for std, opts in candidates.items()}
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(
            "Could not detect required columns for: "
            + ", ".join(missing)
            + f"\nAvailable columns: {list(df.columns)}\n"
            "Please rename your columns or update the candidates list."
        )
    return mapping

def clean_numeric(df, cols):
    """Coerce to numeric and drop rows with NaNs/inf in these columns."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    return out

def plot_actual_vs_pred(y_test, y_pred):
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    mn = min(np.min(y_test), np.min(y_pred))
    mx = max(np.max(y_test), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="red")
    plt.tight_layout()
    plt.show()

def plot_residuals_vs_pred(y_pred, residuals):
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6, color="green")
    plt.axhline(0, linestyle="--", color="red")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.show()

def plot_residual_hist(residuals):
    plt.figure()
    plt.hist(residuals, bins=40, density=True, color="purple", alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.title("Residuals Distribution")
    plt.tight_layout()
    plt.show()

def plot_3d_sqft_beds_price(data, sqft_col, bed_col, price_col):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        data[sqft_col].values,
        data[bed_col].values,
        data[price_col].values,
        alpha=0.6,
        color="orange"
    )
    ax.set_xlabel("Square Footage")
    ax.set_ylabel("Bedrooms")
    ax.set_zlabel("Price")
    plt.title("3D: Price vs Sqft vs Bedrooms")
    plt.tight_layout()
    plt.show()

def plot_corr_heatmap(df_subset):
    corr = df_subset.corr().values
    labels = list(df_subset.columns)

    plt.figure()
    im = plt.imshow(corr, interpolation="nearest", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()

def main():
    # 1) Load
    if not Path(FILE_PATH).exists():
        raise FileNotFoundError(f"CSV not found at: {FILE_PATH}")
    df = pd.read_csv(FILE_PATH, encoding="utf-8", engine="c", low_memory=False)

    print("\nFirst 5 rows:\n", df.head(), "\n")
    print("Columns found:\n", list(df.columns), "\n")

    # 2) Detect required columns
    mapping = detect_columns(df)
    sqft_col = mapping["sqft_living"]
    bed_col = mapping["bedrooms"]
    bath_col = mapping["bathrooms"]
    price_col = mapping["price"]
    print("Detected columns:", mapping)

    # 3) Keep only needed columns; clean numerics
    data = df[[sqft_col, bed_col, bath_col, price_col]].copy()
    data = clean_numeric(data, [sqft_col, bed_col, bath_col, price_col])

    # 4) Train/test split
    X = data[[sqft_col, bed_col, bath_col]].values
    y = data[price_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 5) Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6) Predict + metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Model Summary ===")
    for name, coef in zip([sqft_col, bed_col, bath_col], model.coef_):
        print(f"Coefficient for {name:>14}: {coef: .6f}")
    print(f"Intercept: {model.intercept_: .6f}")
    print(f"MSE : {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

    # 7) Visualizations
    plot_actual_vs_pred(y_test, y_pred)
    residuals = y_test - y_pred
    plot_residuals_vs_pred(y_pred, residuals)
    plot_residual_hist(residuals)
    plot_3d_sqft_beds_price(data, sqft_col, bed_col, price_col)
    plot_corr_heatmap(data[[sqft_col, bed_col, bath_col, price_col]])

if __name__ == "__main__":
    main()
