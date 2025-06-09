import os
import numpy as np
import pandas as pd
import pickle

class FootballDataLoader2:
    def __init__(self, data_dir, team):
        self.data_dir = data_dir.rstrip("/")
        self.team = team
        self.team_path = os.path.join(self.data_dir, team)
        self.all_data_path = os.path.join(self.team_path, "AllData")
        self.xg_data_path = os.path.join(self.team_path, "XGdata")
        self.passes_path = os.path.join(self.team_path, "Passes")
        self.others_path = os.path.join(self.team_path, "Others")

    def extract_teams_from_filename(self, filename):
        parts = filename.split("_")
        if len(parts) >= 4 and parts[0] == "Game" and "Score" in parts:
            return parts[1], parts[2]
        return None, None

    def list_all_games(self):
        sc_files = sorted(os.listdir(self.all_data_path))
        xg_files = sorted(os.listdir(self.xg_data_path))
        return sc_files, xg_files

    def sort_games(self):
        sc_files, xg_files = self.list_all_games()
        sc_dates = [
            int("".join(f.split("Day_20")[1].split("Z.pkl")[0].split("-")))
            for f in sc_files
        ]
        sorted_indices = np.argsort(sc_dates)
        sorted_sc = [sc_files[i] for i in sorted_indices]
        sorted_xg = [xg_files[i] for i in sorted_indices]
        return np.array(sorted_xg), np.array(sorted_sc)

    def load_sec_game(self, filename):
        print(f"Loading {filename}")
        path = os.path.join(self.all_data_path, filename)
        M = pd.read_pickle(path)
        print("Data read.")
        return M["Times"], M["Ball"], M[self.team], M["Opp"]

    def scrape_time(self, mTime):
        return pd.DataFrame({"Time": [x[0] for x in mTime], "game": [x[1] for x in mTime]})

    def scrape_ball(self, mBall, just_game=True, speed=True, col5=True, z=True):
        cols = {"Ball_x": [], "Ball_y": [], "game": []}
        if z: cols["Ball_z"] = []
        if speed: cols["Ball_Speed?"] = []
        if col5: cols["Ball_Col5"] = []

        for row in mBall:
            cols["Ball_x"].append(row[0])
            cols["Ball_y"].append(row[1])
            if z: cols["Ball_z"].append(row[2])
            if speed: cols["Ball_Speed?"].append(row[3])
            if col5: cols["Ball_Col5"].append(row[4])
            cols["game"].append(row[5])

        df = pd.DataFrame(cols)
        return df[df["game"] == 1] if just_game else df

    def scrape_team(self, mFcn, name="home", speed=True, z=True):
        numbers = [int(player[4]) for player in mFcn[0]]
        cols = []
        for i in numbers:
            cols += [f"{name}player_{i}_x", f"{name}player_{i}_y"]
            if z:
                cols.append(f"{name}player_{i}_z")
            if speed:
                cols.append(f"{name}player_{i}_speed_x")
            cols.append(f"{name}player_{i}_number")

        data_rows = []
        for frame in mFcn:
            row = dict.fromkeys(cols, np.nan)
            for i in range(frame.shape[0]):
                number = int(frame[i][4])
                prefix = f"{name}player_{number}"
                row[f"{prefix}_x"] = frame[i][0]
                row[f"{prefix}_y"] = frame[i][1]
                if z: row[f"{prefix}_z"] = frame[i][2]
                if speed: row[f"{prefix}_speed_x"] = frame[i][3]
                row[f"{prefix}_number"] = number
            data_rows.append(row)
        return pd.DataFrame(data_rows)

    def scrape_game(
        self,
        filename,
        in_play_only=True,
        speed_ball=True,
        speed_player=True,
        ball_z=True,
        player_z=True,
        col5=True,
        verbose=True
    ):
        mTime, mBall, mTeam, mOpp = self.load_sec_game(filename)
        team1, team2 = self.extract_teams_from_filename(filename)
        opponent_name = team2 if self.team == team1 else team1

        df_time = self.scrape_time(mTime)
        df_ball = self.scrape_ball(mBall, just_game=False, speed=speed_ball, col5=col5, z=ball_z)
        df_ball = df_ball.drop(columns=["game"], errors="ignore")  # Remove duplicate 'game'
        df_team = self.scrape_team(mTeam, name=self.team, speed=speed_player, z=player_z)
        df_opp = self.scrape_team(mOpp, name=opponent_name, speed=speed_player, z=player_z)

        df = pd.concat([df_time, df_ball, df_team, df_opp], axis=1)
        if in_play_only:
            df = df[df["game"] == 1]

        if verbose:
            print("Time shape:", df_time.shape)
            print("Ball shape:", df_ball.shape)
            print("Team shape:", df_team.shape)
            print("Opponent shape:", df_opp.shape)
            print("Total shape:", df.shape)
            print(f"Opponent name detected from filename: {opponent_name}")

        df.attrs["opponent_name"] = opponent_name
        return df

    def load_all_games(
        self,
        n_games=1,
        in_play_only=True,
        speed_ball=True,
        speed_player=True,
        ball_z=True,
        player_z=True,
        every_n=None,
        save=False,
        verbose=True,
    ):
        if isinstance(verbose, tuple):
            verbose_info, verbose_plots = verbose
        else:
            verbose_info = verbose_plots = verbose

        datasets = []
        _, sorted_sc = self.sort_games()

        if isinstance(n_games, str) and n_games.lower() == "all":
            num_to_load = len(sorted_sc)
        elif isinstance(n_games, int):
            num_to_load = n_games
        else:
            raise ValueError("n_games must be an integer or the string 'all'.")

        for i, filename in enumerate(sorted_sc[:num_to_load]):
            print(f"Reading game {i+1}: {filename}")
            df = self.scrape_game(
                filename,
                in_play_only=in_play_only,
                speed_ball=speed_ball,
                speed_player=speed_player,
                ball_z=ball_z,
                player_z=player_z,
                col5=True,
                verbose=verbose_info,
            )

            if every_n is not None and every_n > 1:
                df = df.iloc[::every_n].reset_index(drop=True)
                if verbose_info:
                    print(f"Downsampled: kept every {every_n}-th frame (rows left: {len(df)})")

            if save:
                os.makedirs(save, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(filename))[0]
                save_path = os.path.abspath(os.path.join(save, base_name + ".hdf"))
                print(f"Attempting to save to: {save_path}")
                if os.path.exists(save_path):
                    print(f"[Warning] File already exists, deleting: {save_path}")
                    os.remove(save_path)
                try:
                    df.to_hdf(save_path, key="df", mode="w")
                    print(f"Saved to {save_path}")
                except Exception as e:
                    print(f"[ERROR] Saving failed: {e}")

            datasets.append(df)

        return datasets



def main():
    # === Setup paths ===
    data_dir = "/Users/denizadiguzel/FootballData_FromMathias_May2025/RestructuredData_2425"
    team = "FCK"

    # === Initialize loader ===
    loader = FootballDataLoader2(data_dir, team)

    # === Load two games with all players real-numbered ===
    datasets = loader.load_all_games(
        n_games=2,
        in_play_only=True,
        speed_ball=False,
        speed_player=False,
        ball_z=False,
        player_z=False,
        every_n=5,
        save=False,
        verbose=(True, False)
    )

    # === Inspect variables ===
    for i, df in enumerate(datasets):
        print(f"\n=== Game {i+1} ===")
        print("Shape:", df.shape)

        # Opponent and team player number columns
        number_cols = [col for col in df.columns if "_number" in col]
        print(f"Number-related columns ({len(number_cols)}):")
        for col in number_cols:
            print(f"  {col}")

        print("\nFirst few rows of player identity columns:")
        print(df[number_cols].head())

        # Check opponent name from attrs
        print("Opponent name:", df.attrs.get("opponent_name", "Unknown"))

if __name__ == "__main__":
    main()
