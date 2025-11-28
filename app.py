import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CSV_FILE = "matches.csv"

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    # Compute total engagements
    df["total_engagements"] = df["kills"] + df["deaths"] + df["assists"]
    return df

def plot_stats(df):
    # KD Ratio vs Total Engagements
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="total_engagements", y="kd_ratio", alpha=0.6)
    sns.regplot(data=df, x="total_engagements", y="kd_ratio", scatter=False, color="red")
    plt.xlabel("Total Engagements (Kills + Deaths + Assists)")
    plt.ylabel("KD Ratio")
    plt.title("KD Ratio vs Total Engagements")
    plt.show()

    # HS% vs Total Engagements
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="total_engagements", y="hs_percent", alpha=0.6)
    sns.regplot(data=df, x="total_engagements", y="hs_percent", scatter=False, color="red")
    plt.xlabel("Total Engagements (Kills + Deaths + Assists)")
    plt.ylabel("HS%")
    plt.title("Headshot % vs Total Engagements")
    plt.show()

def predictive_analysis(df):
    # analysis with the following variables: hs%, kd_ratio, total_engagements vs win
    features = df[["hs_percent", "kd_ratio", "total_engagements"]].fillna(0)
    target = df["win"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n==============================")
    print("=== Predictive Analysis (HS%, KD, Engagements) ===")
    print("==============================\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    importance = pd.DataFrame({"feature": features.columns, "coefficient": model.coef_[0]})
    print("\nFeature Importance:\n", importance)
    print("K/D ratio is by far the only meaningful statistic in relation to win-rate")

def kd_win_probability_curve(df):
    """Compute KD needed for 50%, 60%, 70% win chance using logistic regression"""
    df_clean = df.dropna(subset=["kd_ratio", "win"])
    X = df_clean[["kd_ratio"]]
    y = df_clean["win"]

    model = LogisticRegression()
    model.fit(X, y)

    print("\n==============================")
    print("=== KD Win Probability Curve ===")
    print("==============================\n")
    for target in [0.5, 0.6, 0.7]:
        a = model.coef_[0][0]
        b = model.intercept_[0]
        kd_needed = (np.log(target / (1 - target)) - b) / a
        print(f"KD needed for {int(target*100)}% win chance: {kd_needed:.3f}")

def kd_vs_engagement_regression(df):
    # Simple linear regression KD vs Total Engagements
    df_clean = df.dropna(subset=["kd_ratio", "total_engagements"])
    X = df_clean["total_engagements"]
    y = df_clean["kd_ratio"]
    slope, intercept = np.polyfit(X, y, 1)
    corr = np.corrcoef(X, y)[0, 1]
    print("\n==============================")
    print("=== KD vs Engagement Regression ===")
    print("==============================\n")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"Correlation (R): {corr:.4f}")
    print("\nInterpretation:")
    if slope > 0:
        print("-> As engagements increase, KD tends to increase slightly.")
    else:
        print("-> As engagements increase, KD tends to decrease slightly.")

def main():
    df = load_data(CSV_FILE)
    plot_stats(df)
    predictive_analysis(df)
    kd_win_probability_curve(df)
    kd_vs_engagement_regression(df)

if __name__ == "__main__":
    main()
