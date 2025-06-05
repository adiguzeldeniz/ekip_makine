import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from functools import lru_cache


class Visualizer:
    def __init__(self, datasets, team_prefix="FCK", opponent_prefix=None):
        self.datasets = datasets
        self.team_prefix = team_prefix
        self.opponent_prefix = opponent_prefix or self._infer_opponent()

        # Normalized pitch bounds
        self.XMIN, self.XMAX = -1.0, 1.0
        self.YMAX = 68 / 105  # ≈ 0.6476
        self.YMIN = -self.YMAX

    @lru_cache(maxsize=1)
    def _infer_opponent(self):
        df = self.datasets[0]
        all_teams = set(col.split("player_")[0] for col in df.columns if "_x" in col and "player_" in col)
        all_teams.discard(self.team_prefix)
        return list(all_teams)[0] if all_teams else None

    def _normalize_coordinates(self, x, y, x_min, x_max, y_min, y_max):
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
        y_norm *= (68 / 105)  # compress Y to correct aspect
        return x_norm, y_norm

    def _get_bounds(self, df):
        x = df["Ball_x"].dropna()
        y = df["Ball_y"].dropna()
        return x.min(), x.max(), y.min(), y.max()

    def _draw_pitch_elements(self, ax):
        ax.set_facecolor("#175e1e")

        # Pitch outline
        ax.plot([self.XMIN, self.XMAX, self.XMAX, self.XMIN, self.XMIN],
                [self.YMIN, self.YMIN, self.YMAX, self.YMAX, self.YMIN],
                color="white", lw=2)

        # Center line
        ax.axvline(0, color="white", lw=2)

        # Center circle
        center_circle_radius = 9.15 / 52.5  # ≈ 0.1743
        center_circle = plt.Circle((0, 0), center_circle_radius, edgecolor="white",
                                facecolor="none", lw=1.5, alpha=0.8)
        ax.add_patch(center_circle)

        # Penalty & goal boxes
        penalty_x = 16.5 / 52.5         # ≈ 0.314
        penalty_y = 40.3 / 68 / 2       # ±0.296
        goal_x = 0.5 + (2.44 / 105)     # slightly outside the line
        goal_y = 7.32 / 68 / 2          # ±0.0538
        box5_x = 5.5 / 52.5             # ≈ 0.105
        box5_y = 18.32 / 68 / 2         # ±0.1347

        for side in [-1, 1]:
            # Penalty box
            ax.add_patch(plt.Rectangle(
                (side * (1 - penalty_x), -penalty_y),
                side * penalty_x,
                2 * penalty_y,
                edgecolor="white",
                facecolor="none",
                lw=2
            ))

            # Goal box
            ax.add_patch(plt.Rectangle(
                (side * (1 - box5_x), -box5_y),
                side * box5_x,
                2 * box5_y,
                edgecolor="white",
                facecolor="none",
                lw=1.5
            ))

            # Goal stub outside pitch
            ax.add_patch(plt.Rectangle(
                (side * 1.0, -goal_y),
                side * 0.02,
                2 * goal_y,
                edgecolor="white",
                facecolor="none",
                lw=1.2
            ))

        ax.set_xlim(self.XMIN - 0.05, self.XMAX + 0.05)
        ax.set_ylim(self.YMIN - 0.05, self.YMAX + 0.05)
        ax.set_aspect("equal", adjustable="box")


    def plot_ball_heatmap(self, game_idx=0, bins=100, half=None,
                        save_path=None, show_axes=True, clim=None):
        df = self.datasets[game_idx]
        if half is not None:
            df = df[df["half"] == half]

        # Normalize coordinates
        x_min, x_max, y_min, y_max = self._get_bounds(df)
        x, y = self._normalize_coordinates(df["Ball_x"].dropna(), df["Ball_y"].dropna(),
                                        x_min, x_max, y_min, y_max)

        # Compute 2D histogram
        counts, xedges, yedges = np.histogram2d(x, y, bins=bins)

        # Plot
        fig, ax = plt.subplots(figsize=(9, 6))
        mesh = ax.pcolormesh(xedges, yedges, counts.T, cmap="magma", shading="auto",
                            vmin=clim[0] if clim else None,
                            vmax=clim[1] if clim else None)

        # Colorbar (tall, narrow, right)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Density")

        self._draw_pitch_elements(ax)

        if show_axes:
            ax.set_xlabel("Normalized Ball X")
            ax.set_ylabel("Normalized Ball Y")
            ax.grid(True, linestyle="--", alpha=0.3)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)

        title = f"Ball Position Heatmap — Game {game_idx}"
        if half:
            title += f" (Half {half})"
        ax.set_title(title)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()




    def plot_player_heatmap(self, game_idx=0, slot=0, side="team", bins=100, save_path=None):
        df = self.datasets[game_idx]
        prefix = self.team_prefix if side == "team" else self.opponent_prefix

        x_min, x_max, y_min, y_max = self._get_bounds(df)
        x, y = self._normalize_coordinates(df[f"{prefix}player_{slot}_x"].dropna(),
                                           df[f"{prefix}player_{slot}_y"].dropna(),
                                           x_min, x_max, y_min, y_max)

        plt.figure(figsize=(9, 6))
        sns.histplot(x=x, y=y, bins=bins, pthresh=0.1, cmap="viridis")
        self._draw_pitch_elements(plt.gca())
        plt.title(f"{prefix} Player {slot} Position Heatmap — Game {game_idx}")
        plt.xlabel("Normalized X")
        plt.ylabel("Normalized Y")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_combined_ball_heatmap(self, bins=100, half=None, save_path=None):
        all_x, all_y = [], []
        for df in self.datasets:
            if half is not None:
                df = df[df["half"] == half]
            all_x.append(df["Ball_x"])
            all_y.append(df["Ball_y"])

        all_x = pd.concat(all_x).dropna()
        all_y = pd.concat(all_y).dropna()
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        x, y = self._normalize_coordinates(all_x, all_y, x_min, x_max, y_min, y_max)

        plt.figure(figsize=(9, 6))
        sns.histplot(x=x, y=y, bins=bins, pthresh=0.1, cmap="magma")
        self._draw_pitch_elements(plt.gca())
        plt.title(f"Combined Ball Position Heatmap — All Games" + (f" (Half {half})" if half else ""))
        plt.xlabel("Normalized Ball X")
        plt.ylabel("Normalized Ball Y")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_player_slot_histogram(self, game_idx=0, time_col="Time", save_path=None):
        df = self.datasets[game_idx]
        slot_cols = [f"{self.team_prefix}player_{i}_number" for i in range(11)]
        matrix = df[slot_cols].to_numpy()

        unique_numbers = sorted(np.unique(matrix[~np.isnan(matrix)]).astype(int))
        n_unique = len(unique_numbers)
        number_to_index = {num: idx for idx, num in enumerate(unique_numbers)}

        index_matrix = np.full_like(matrix, fill_value=-1, dtype=int)
        for i, row in enumerate(matrix):
            for j, num in enumerate(row):
                if not np.isnan(num):
                    index_matrix[i, j] = number_to_index[int(num)]

        cmap = plt.get_cmap("tab20", n_unique)
        norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n_unique + 0.5), ncolors=n_unique)

        y_vals = df[time_col].values if time_col in df.columns else np.arange(len(df))

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(index_matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, origin="lower")

        ax.set_xticks(np.arange(11))
        ax.set_xticklabels([f"{i}" for i in range(11)])
        ax.set_xlabel("Artificial Player Slot")
        ax.set_ylabel("Time (s)" if time_col in df.columns else "Frame Index")
        ax.set_title(f"{self.team_prefix} Player Substitution Map — Game {game_idx}")

        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(n_unique))
        cbar.ax.set_yticklabels([str(num) for num in unique_numbers])
        cbar.set_label("Jersey Number")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
