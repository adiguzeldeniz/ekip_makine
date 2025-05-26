import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

class Visualizer:
    def __init__(self, datasets, team_prefix="FCK", opponent_prefix=None):
        self.datasets = datasets
        self.team_prefix = team_prefix
        self.opponent_prefix = opponent_prefix or self._infer_opponent()

    def _infer_opponent(self):
        df = self.datasets[0]
        all_teams = set(col.split("player_")[0] for col in df.columns if "_x" in col and "player_" in col)
        all_teams.discard(self.team_prefix)
        return list(all_teams)[0] if all_teams else None


    def plot_ball_heatmap(self, game_idx=0, bins=100):
        df = self.datasets[game_idx]
        x, y = df["Ball_x"], df["Ball_y"]

        plt.figure(figsize=(8, 6))
        sns.histplot(x=x, y=y, bins=bins, pthresh=0.1, cmap="magma")
        plt.title(f"Ball Position Heatmap — Game {game_idx}")
        plt.xlabel("Ball X")
        plt.ylabel("Ball Y")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_player_heatmap(self, game_idx=0, slot=0, side="team", bins=100):
        df = self.datasets[game_idx]
        prefix = self.team_prefix if side == "team" else self.opponent_prefix
        x = df[f"{prefix}player_{slot}_x"]
        y = df[f"{prefix}player_{slot}_y"]

        plt.figure(figsize=(8, 6))
        sns.histplot(x=x, y=y, bins=bins, pthresh=0.1, cmap="viridis")
        plt.title(f"{prefix} Player {slot} Position Heatmap — Game {game_idx}")
        plt.xlabel("Player X")
        plt.ylabel("Player Y")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_player_slot_histogram(self, game_idx=0, time_col="Time"):
        """
        Visualize which jersey number occupied each artificial player slot over time.
        """
        df = self.datasets[game_idx]
        slot_cols = [f"{self.team_prefix}player_{i}_number" for i in range(11)]
        matrix = df[slot_cols].to_numpy()

        # Find unique jersey numbers
        unique_numbers = sorted(np.unique(matrix[~np.isnan(matrix)]).astype(int))
        n_unique = len(unique_numbers)
        number_to_index = {num: idx for idx, num in enumerate(unique_numbers)}

        # Build index matrix for colormap
        index_matrix = np.full_like(matrix, fill_value=-1, dtype=int)
        for i, row in enumerate(matrix):
            for j, num in enumerate(row):
                if not np.isnan(num):
                    index_matrix[i, j] = number_to_index[int(num)]

        # Discrete colormap
        cmap = plt.get_cmap("tab20", n_unique)
        norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n_unique + 0.5), ncolors=n_unique)

        # Time axis
        y_vals = df[time_col].values if time_col in df.columns else np.arange(len(df))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(index_matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, origin="lower")

        ax.set_xticks(np.arange(11))
        ax.set_xticklabels([f"{i}" for i in range(11)])
        ax.set_xlabel("Artificial Player Slot")
        ax.set_ylabel("Time (s)" if time_col in df.columns else "Frame Index")
        ax.set_title(f"{self.team_prefix} Player Substitution Map — Game {game_idx}")

        # Colorbar with jersey numbers
        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(n_unique))
        cbar.ax.set_yticklabels([str(num) for num in unique_numbers])
        cbar.set_label("Jersey Number")

        plt.tight_layout()
        plt.show()
