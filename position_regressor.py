import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

class PositionRegressor:
    def __init__(self, model_path, scaler_path, feature_path):
        self.model = load_model(model_path, compile=False)
        self.scaler = joblib.load(scaler_path)
        with open(feature_path, "r") as f:
            self.features = json.load(f)

    def load_player_data(self, h5_file, key="df"):
        df = pd.read_hdf(h5_file, key=key)

        # Extract labels from player_id
        df["team_extracted"] = df["player_id"].str.extract(r"player_([A-Z]+)_")[0]
        df["number_extracted"] = df["player_id"].str.extract(r"player_[A-Z]+_([0-9]+)")[0]
        df["label"] = df["team_extracted"] + "_" + df["number_extracted"]

        # Compute relative position to team average
        team_means = df.groupby(["match_id", "team_extracted"])[["x", "y"]].mean().reset_index()
        team_means = team_means.rename(columns={"x": "x_team_avg", "y": "y_team_avg"})
        df = df.merge(team_means, on=["match_id", "team_extracted"], how="left")
        df["x_rel"] = df["x"] - df["x_team_avg"]
        df["y_rel"] = df["y"] - df["y_team_avg"]

        # Aggregate per-player statistics
        player_stats = df.groupby("player_id")[["x_rel", "y_rel", "speed"]].mean()
        player_stats["x_rel_std"] = df.groupby("player_id")["x_rel"].std().values
        player_stats["y_rel_std"] = df.groupby("player_id")["y_rel"].std().values
        player_stats["speed_std"] = df.groupby("player_id")["speed"].std().values
        player_stats["label"] = df.groupby("player_id")["label"].first().values

        self.player_df = player_stats.reset_index()

    def predict_roles(self):
        X = self.scaler.transform(self.player_df[self.features].values)
        self.player_df["predicted_role_score"] = self.model.predict(X).flatten()
        return self.player_df

    def plot_predictions(self):
        df = self.player_df.sort_values("predicted_role_score")
        plt.figure(figsize=(10, 6))
        plt.hlines(y=range(len(df)), xmin=0, xmax=1, color='lightgray', alpha=0.3)
        plt.scatter(df["predicted_role_score"], range(len(df)), color="blue", alpha=0.7)

        for val, name in {0.0: "Defence", 0.5: "Midfield", 1.0: "Attack"}.items():
            plt.axvline(val, linestyle="--", color="gray", alpha=0.5)
            plt.text(val, -2, name, ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.xlabel("Predicted Role Score")
        plt.ylabel("Players")
        plt.title("Predicted Role Scores (All Players)")
        plt.grid(axis='x', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_role_distribution(self, bins=[0.0, 0.33, 0.66, 1.0]):
        df = self.player_df.copy()
        df["role_bin"] = pd.cut(df["predicted_role_score"], bins=bins, labels=["Def", "Mid", "Att"], include_lowest=True)
        counts = df["role_bin"].value_counts().sort_index()
        counts.plot(kind="bar", color="skyblue")
        plt.ylabel("Number of Players")
        plt.title("Predicted Role Distribution")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
