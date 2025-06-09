import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class RoleRegressor:
    def __init__(self, cluster_file, data_file, features=None):
        self.cluster_file = cluster_file
        self.data_file = data_file
        self.features = features or [
            "x_rel", "x_rel_std", "y_rel", "y_rel_std", "speed", "speed_std"
        ]
        self.model = None
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self):
        clusters = pd.read_csv(self.cluster_file)
        self.cluster_df = clusters[["label", "cluster"]].copy()

        df = pd.read_hdf(self.data_file, key="df")
        df["team_extracted"] = df["player_id"].str.extract(r"player_([A-Z]+)_")[0]
        df["number_extracted"] = df["player_id"].str.extract(r"player_[A-Z]+_([0-9]+)")[0]
        df["label"] = df["team_extracted"] + "_" + df["number_extracted"]

        team_means = df.groupby(["match_id", "team_extracted"])[["x", "y"]].mean().reset_index()
        team_means = team_means.rename(columns={"x": "x_team_avg", "y": "y_team_avg"})
        df = df.merge(team_means, on=["match_id", "team_extracted"], how="left")
        df["x_rel"] = df["x"] - df["x_team_avg"]
        df["y_rel"] = df["y"] - df["y_team_avg"]

        player_stats = df.groupby("player_id")[
            ["x_rel", "y_rel", "speed", "v_x", "v_y"]
        ].mean()
        player_stats["x_rel_std"] = df.groupby("player_id")["x_rel"].std().values
        player_stats["y_rel_std"] = df.groupby("player_id")["y_rel"].std().values
        player_stats["speed_std"] = df.groupby("player_id")["speed"].std().values
        player_stats["label"] = df.groupby("player_id")["label"].first().values

        self.player_stats = player_stats.merge(self.cluster_df, on="label", how="inner")

    def filter_training_clusters(self, train_clusters):
        role_map = {5: 0.0, 3: 0.5, 6: 0.5, 4: 1.0}
        filtered = self.player_stats[self.player_stats["cluster"].isin(train_clusters)].copy()
        filtered["role_score"] = filtered["cluster"].map(role_map)
        self.filtered_stats = filtered

    def train_on_filtered_data(self, test_frac=0.05):
        self.train_df = self.filtered_stats.groupby("cluster", group_keys=False).apply(
            lambda x: x.sample(frac=1 - test_frac, random_state=42)
        )
        self.test_df = self.filtered_stats.drop(self.train_df.index)

        self.X_train = self.train_df[self.features].values
        self.y_train = self.train_df["role_score"].values
        self.X_test = self.test_df[self.features].values
        self.y_test = self.test_df["role_score"].values

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def build_model(self):
        self.model = Sequential([
            Input(shape=(self.X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])

    def train(self, epochs=150, batch_size=16):
        self.model.fit(self.X_train_scaled, self.y_train, epochs=epochs,
                       batch_size=batch_size, validation_split=0.1, verbose=0)

    def save_model_and_scaler(self, model_path="model.h5", scaler_path="scaler.pkl", feature_path="features.json"):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        with open(feature_path, "w") as f:
            json.dump(self.features, f)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test_scaled).flatten()
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        smape = 100 * np.mean(
            np.abs(y_pred - self.y_test) / (np.abs(y_pred) + np.abs(self.y_test) + 1e-6)
        )
        print("\nKeras Model Evaluation:")
        print(f"Test MAE   : {mae:.4f}")
        print(f"Test RMSE  : {rmse:.4f}")
        print(f"Test R²    : {r2:.4f}")
        print(f"Test SMAPE : {smape:.2f}%")
        print(f"Min/Max prediction: {y_pred.min():.3f} / {y_pred.max():.3f}")
        self.y_pred = y_pred

    def get_players_by_cluster(self, cluster_list):
        return self.player_stats[self.player_stats["cluster"].isin(cluster_list)].copy()

    def predict(self, player_df):
        X = self.scaler.transform(player_df[self.features].values)
        return self.model.predict(X).flatten()

    def plot_predictions(self):
        role_name_map = {0.0: "Defence", 0.5: "Midfield", 1.0: "Attack"}
        comparison_df = self.test_df.copy()
        comparison_df["predicted_score"] = self.y_pred
        comparison_df = comparison_df.sort_values("role_score")

        plt.figure(figsize=(12, 6))
        plt.hlines(y=range(len(comparison_df)), xmin=0, xmax=1, color='lightgray', alpha=0.3)
        plt.scatter(comparison_df["predicted_score"], range(len(comparison_df)),
                    color='blue', label='Predicted', alpha=0.7)
        plt.scatter(comparison_df["role_score"], range(len(comparison_df)),
                    color='red', marker='x', label='True', alpha=0.7)

        for val, name in role_name_map.items():
            plt.axvline(val, color='gray', linestyle='--', alpha=0.4)
            plt.text(val, -2, name, ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.xlabel("Role Score (0 = Defence, 1 = Attack)")
        plt.ylabel("Players (sorted by role)")
        plt.title("Model Predictions vs True Role Labels (Test Set)")
        plt.legend()
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.show()

    def plot_unseen_predictions(self, cluster_ids):
        unseen_df = self.get_players_by_cluster(cluster_ids)
        preds = self.predict(unseen_df)
        unseen_df["predicted_role_score"] = preds

        plt.figure(figsize=(10, 6))
        cluster_names = {0: "Right Wing", 1: "Left Wing", 2: "Goalkeeper"}
        colors = {0: "blue", 1: "green", 2: "orange"}

        for cluster_id in cluster_ids:
            group = unseen_df[unseen_df["cluster"] == cluster_id]
            plt.scatter(
                group["predicted_role_score"],
                [cluster_names[cluster_id]] * len(group),
                alpha=0.7,
                color=colors[cluster_id],
                label=f"{cluster_names[cluster_id]} (n={len(group)})"
            )

        plt.axvline(0.0, linestyle="--", color="gray", alpha=0.4)
        plt.axvline(0.5, linestyle="--", color="gray", alpha=0.4)
        plt.axvline(1.0, linestyle="--", color="gray", alpha=0.4)

        plt.xlabel("Predicted Role Score (0 = Defence, 1 = Attack)")
        plt.title("Predicted Role Scores for Excluded Clusters")
        plt.legend()
        plt.grid(axis='x', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_unseen_predictions_with_names(self, cluster_ids, top_n=30):
        unseen_df = self.get_players_by_cluster(cluster_ids)
        preds = self.predict(unseen_df)
        unseen_df["predicted_role_score"] = preds

        # Sort by predicted role score
        unseen_df_sorted = unseen_df.sort_values("predicted_role_score", ascending=False)

        # Limit to top_n players for readability
        top_players = unseen_df_sorted.head(top_n)

        plt.figure(figsize=(12, 0.5 * top_n))
        bars = plt.barh(top_players["label"], top_players["predicted_role_score"], color='skyblue')
        plt.axvline(0.0, linestyle="--", color="gray", alpha=0.4)
        plt.axvline(0.5, linestyle="--", color="gray", alpha=0.4)
        plt.axvline(1.0, linestyle="--", color="gray", alpha=0.4)

        plt.xlabel("Predicted Role Score (0 = Defence, 1 = Attack)")
        plt.title(f"Top {top_n} Predicted Role Scores for Excluded Clusters")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
