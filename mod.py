import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def load_and_process_data():
    # Load CSV files
    historical_df = pd.read_csv("Processed_Election_Data.csv")
    sentiment_df = pd.read_csv("Sorted_Synthetic_Sentiment_Data.csv")
    
    # Map sentiment text to numeric values
    sentiment_mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
    sentiment_df["Sentiment"] = sentiment_df["Sentiment"].map(sentiment_mapping)
    
    # Aggregate sentiment scores per Constituency_No and Party (using mean)
    sentiment_agg = sentiment_df.groupby(["Constituency_No", "Party"])["Sentiment"].mean().reset_index()
    sentiment_agg.rename(columns={"Sentiment": "Avg_Sentiment_Score"}, inplace=True)
    
    # Merge historical and aggregated sentiment data (full outer join)
    merged_outer = pd.merge(historical_df, sentiment_agg, on=["Constituency_No", "Party"], how="outer")
    
    # Fill missing Avg_Sentiment_Score with 0 (assume neutral sentiment)
    merged_outer["Avg_Sentiment_Score"] = merged_outer["Avg_Sentiment_Score"].fillna(0)
    
    # Normalize party names to uppercase to avoid case mismatches
    merged_outer["Party"] = merged_outer["Party"].str.upper()
    
    # Impute missing historical data
    historical_numeric = ['Vote_Share_2016', 'Vote_Share_2021']
    historical_bool = ['Incumbent_2016', 'Incumbent_2021',
                       'Turncoat_2016', 'Turncoat_2021',
                       'Recontest_2016', 'Recontest_2021',
                       'Winner_2016', 'Winner_2021']
    for col in historical_numeric:
        merged_outer[col] = merged_outer[col].fillna(0.0)
    for col in historical_bool:
        merged_outer[col] = merged_outer[col].fillna(False)
    if "Unnamed: 0" in merged_outer.columns:
        merged_outer = merged_outer.drop(columns=["Unnamed: 0"])
    
    # Remove unwanted parties: "CONGRESS", "IND", and "PMK"
    merged_outer = merged_outer[~merged_outer["Party"].isin(["CONGRESS", "IND", "PMK"])]
    
    # Set baseline for parties absent in historical data (both vote shares are 0)
    new_parties_mask = (merged_outer["Vote_Share_2016"] == 0) & (merged_outer["Vote_Share_2021"] == 0)
    merged_outer.loc[new_parties_mask, "Vote_Share_2016"] = 9 * (merged_outer.loc[new_parties_mask, "Avg_Sentiment_Score"].clip(lower=0) + 1)
    merged_outer.loc[new_parties_mask, "Vote_Share_2021"] = 20 * (merged_outer.loc[new_parties_mask, "Avg_Sentiment_Score"].clip(lower=0) + 1)
    
    # Compute growth from 2016 to 2021
    merged_outer["Growth_2016_2021"] = merged_outer["Vote_Share_2021"] - merged_outer["Vote_Share_2016"]
    
    return merged_outer

def run_regression(merged_outer):
    # Prepare data for growth modeling
    growth_features = ['Vote_Share_2016', 'Vote_Share_2021', 'Avg_Sentiment_Score']
    growth_target = "Growth_2016_2021"
    X_growth = merged_outer[growth_features]
    y_growth = merged_outer[growth_target]
    
    X_train, X_test, y_train, y_test = train_test_split(X_growth, y_growth, test_size=0.2, random_state=42)
    
    # Train growth model
    reg_growth = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_growth.fit(X_train, y_train)
    y_growth_pred = reg_growth.predict(X_test)
    mse_growth = mean_squared_error(y_test, y_growth_pred)
    
    # Predict 2026 vote share for each record
    merged_outer["Predicted_Growth_2026"] = reg_growth.predict(merged_outer[growth_features])
    merged_outer["Predicted_Vote_Share_2026"] = merged_outer["Vote_Share_2021"] + merged_outer["Predicted_Growth_2026"]
    
    # Determine constituency-wise winners (regression)
    predicted_winners = (
        merged_outer.groupby("Constituency_No")
        .apply(lambda df: df.loc[df["Predicted_Vote_Share_2026"].idxmax(), ["Party", "Predicted_Vote_Share_2026"]])
        .reset_index()
    )
    predicted_winners.rename(columns={"Party": "Predicted_Winner"}, inplace=True)
    
    win_counts = predicted_winners["Predicted_Winner"].value_counts().reset_index()
    win_counts.columns = ["Party", "Constituencies_Won"]
    
    party_sum = merged_outer.groupby("Party")["Predicted_Vote_Share_2026"].sum()
    top10_parties = party_sum.sort_values(ascending=False).head(10)
    total_top10 = top10_parties.sum()
    vote_share_percent = top10_parties / total_top10 * 100
    
    regression_results = {
        "mse_growth": mse_growth,
        "predicted_winners": predicted_winners,
        "win_counts": win_counts,
        "vote_share_percent": vote_share_percent,
        "merged_data": merged_outer
    }
    
    return regression_results

def run_classification(merged_outer):
    # Create binary target: 1 if the party is the historical winner in the constituency (based on 2021 vote share)
    historical_winner = merged_outer.groupby("Constituency_No").apply(
        lambda df: df.loc[df["Vote_Share_2021"].idxmax(), "Party"]
    ).reset_index()
    historical_winner.columns = ["Constituency_No", "Historical_Winner"]
    
    merged_outer = pd.merge(merged_outer, historical_winner, on="Constituency_No", how="left")
    merged_outer["Historical_Winner_Label"] = (merged_outer["Party"] == merged_outer["Historical_Winner"]).astype(int)
    
    clf_features = ['Vote_Share_2016', 'Vote_Share_2021', 'Avg_Sentiment_Score', 'Growth_2016_2021']
    X_clf = merged_outer[clf_features]
    y_clf = merged_outer["Historical_Winner_Label"]
    
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVC": SVC(random_state=42, probability=True),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, verbosity=0),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1)
    }
    
    class_accuracy = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_clf, y_train_clf)
        y_pred_clf = clf.predict(X_test_clf)
        accuracy = accuracy_score(y_test_clf, y_pred_clf)
        class_accuracy[name] = accuracy
    
    # Hybrid Voting Classifier
    hybrid_estimators = [(name, clf) for name, clf in classifiers.items()]
    hybrid_clf = VotingClassifier(estimators=hybrid_estimators, voting='soft')
    hybrid_clf.fit(X_train_clf, y_train_clf)
    y_pred_hybrid = hybrid_clf.predict(X_test_clf)
    hybrid_accuracy = accuracy_score(y_test_clf, y_pred_hybrid)
    
    classification_results = {
        "class_accuracy": class_accuracy,
        "hybrid_accuracy": hybrid_accuracy,
        "merged_data": merged_outer
    }
    
    return classification_results

def run_models():
    # Run the entire modeling pipeline and return results in a dictionary.
    merged_outer = load_and_process_data()
    regression_results = run_regression(merged_outer)
    classification_results = run_classification(regression_results["merged_data"])
    
    results = {
        "regression": regression_results,
        "classification": classification_results
    }
    return results

if __name__ == "__main__":
    # For debugging: run the models and print some key metrics.
    results = run_models()
    print("Growth Model MSE (2016-2021):", results["regression"]["mse_growth"])
    print("Hybrid Model Accuracy:", results["classification"]["hybrid_accuracy"])
