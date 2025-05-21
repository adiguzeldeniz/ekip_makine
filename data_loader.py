import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd



class FootballDataLoader:
    def __init__(self, data_dir, team):
        self.data_dir = data_dir.rstrip("/")
        self.team = team
        self.team_path = os.path.join(self.data_dir, team)
        self.all_data_path = os.path.join(self.team_path, "AllData")
        self.xg_data_path = os.path.join(self.team_path, "XGdata")
        self.passes_path = os.path.join(self.team_path, "Passes")
        self.others_path = os.path.join(self.team_path, "Others")

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

    def extract_teams_from_filename(self, filename):
        file_copy = filename.copy()
        parts = file_copy.split("_")
        if parts[0] == "Game" and "Score" in parts:
            return parts[1], parts[2]  # team1, team2
        return None, None

    def load_sec_game(self, filename):
        print(f"Loading {filename}")
        path = os.path.join(self.all_data_path, filename)
        M = pd.read_pickle(path)
        print("Data read.")
        return M["Times"], M["Ball"], M[self.team], M["Opp"]

    def load_pass_data(self, filename):
        path = os.path.join(self.passes_path, filename)
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[Error loading passes] {filename}: {e}")
            return None

    def load_other_data(self, filename):
        path = os.path.join(self.others_path, filename)
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[Error loading other file] {filename}: {e}")
            return None

    def scrape_ball(self, mBall, just_game=True, speed=True, col5=True, z=True):
        cols = {"Ball_x": [], "Ball_y": [], "game": []}
        if z:
            cols["Ball_z"] = []
        if speed:
            cols["Ball_Speed?"] = []
        if col5:
            cols["Ball_Col5"] = []

        for row in mBall:
            cols["Ball_x"].append(row[0])
            cols["Ball_y"].append(row[1])
            if z:
                cols["Ball_z"].append(row[2])
            if speed:
                cols["Ball_Speed?"].append(row[3])
            if col5:
                cols["Ball_Col5"].append(row[4])
            cols["game"].append(row[5])

        df = pd.DataFrame(cols)
        return df[df["game"] == 1] if just_game else df

    # def scrape_team(self, mFcn, name="home", speed=True, z=True):
    #     numbers = [int(player[4]) for player in mFcn[0]]
    #     cols = []
    #     for i in numbers:
    #         cols += [f"{name}player_{i}_x", f"{name}player_{i}_y"]
    #         if z:
    #             cols.append(f"{name}player_{i}_z")
    #         if speed:
    #             cols.append(f"{name}player_{i}_speed_x")
    #         cols.append(f"{name}player_{i}_number")

    #     data_rows = []
    #     for frame in mFcn:
    #         row = dict.fromkeys(cols, np.nan)
    #         players = frame[:, 4].astype(int)
    #         players_set = set(players)
    #         for i in range(frame.shape[0]):
    #             number = int(frame[i][4])
    #             prefix = f"{name}player_{number}"
    #             row[f"{prefix}_x"] = frame[i][0]
    #             row[f"{prefix}_y"] = frame[i][1]
    #             if z:
    #                 row[f"{prefix}_z"] = frame[i][2]
    #             if speed:
    #                 row[f"{prefix}_speed_x"] = frame[i][3]
    #             row[f"{prefix}_number"] = number
    #         data_rows.append(row)

    #     return pd.DataFrame(data_rows)

    def scrape_team(self, mFcn, name="home", speed=True, z=True, use_artificial_players=False):
        if not use_artificial_players:
            # fallback to default behavior with real player numbers in column names
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
                players = frame[:, 4].astype(int)
                for i in range(frame.shape[0]):
                    number = int(frame[i][4])
                    prefix = f"{name}player_{number}"
                    row[f"{prefix}_x"] = frame[i][0]
                    row[f"{prefix}_y"] = frame[i][1]
                    if z:
                        row[f"{prefix}_z"] = frame[i][2]
                    if speed:
                        row[f"{prefix}_speed_x"] = frame[i][3]
                    row[f"{prefix}_number"] = number
                data_rows.append(row)
            return pd.DataFrame(data_rows)

        # -----------------------------
        # Artificial player slot logic
        # -----------------------------
        n_slots = 11
        cols = []
        for i in range(n_slots):
            cols += [f"{name}player_{i}_x", f"{name}player_{i}_y"]
            if z:
                cols.append(f"{name}player_{i}_z")
            if speed:
                cols.append(f"{name}player_{i}_speed_x")
            cols.append(f"{name}player_{i}_number")  # real player number

        data_rows = []
        active_mapping = {}  # maps real player number -> artificial slot
        slot_occupied = {}   # maps slot -> real player number

        for frame in mFcn:
            row = dict.fromkeys(cols, np.nan)
            current_players = frame[:, 4].astype(int)
            new_mapping = {}

            # Find which players are still on the field
            still_active = set(current_players).intersection(active_mapping.keys())
            freed_slots = set(range(n_slots)) - set(active_mapping[p] for p in still_active)

            # Assign existing players
            for i in range(frame.shape[0]):
                number = int(frame[i][4])
                if number in active_mapping:
                    slot = active_mapping[number]
                else:
                    if freed_slots:
                        slot = freed_slots.pop()
                    else:
                        raise RuntimeError("More than 11 players detected on field!")
                    active_mapping[number] = slot
                new_mapping[number] = slot

                prefix = f"{name}player_{slot}"
                row[f"{prefix}_x"] = frame[i][0]
                row[f"{prefix}_y"] = frame[i][1]
                if z:
                    row[f"{prefix}_z"] = frame[i][2]
                if speed:
                    row[f"{prefix}_speed_x"] = frame[i][3]
                row[f"{prefix}_number"] = number

            # Update mapping for next frame
            active_mapping = new_mapping.copy()
            data_rows.append(row)

        return pd.DataFrame(data_rows)



    def scrape_time(self, mTime):
        return pd.DataFrame(
            {"Time": [x[0] for x in mTime], "half": [x[1] for x in mTime]}
        )

    def scrape_game(self, filename, in_play_only=True, speed=True, z=True, col5=True, use_artificial_players=False, verbose=True):
        mTime, mBall, mFcn, mOpp = self.load_sec_game(filename)
        home_name, away_name = self.extract_teams_from_filename(filename)
        df_time = self.scrape_time(mTime)
        df_ball = self.scrape_ball(mBall, just_game=False, speed=speed, col5=col5, z=z)
        df_team = self.scrape_team(mFcn, name=home_name, speed=speed, z=z, use_artificial_players=use_artificial_players)
        df_opp = self.scrape_team(mOpp, name=away_name, speed=speed, z=z, use_artificial_players=use_artificial_players)

        df = pd.concat([df_time, df_ball, df_team, df_opp], axis=1)
        if in_play_only:
            df = df[df["game"] == 1]

        if verbose:
            print("Time shape:", df_time.shape)
            print("Ball shape:", df_ball.shape)
            print("Team shape:", df_team.shape)
            print("Opponent shape:", df_opp.shape)
            print("Total shape:", df.shape)

        return df


    def load_all_games(
        self,
        n_games=1,
        in_play_only=True,
        speed=True,
        z=True,
        use_artificial_players=False,
        every_n=None,
        save=False,
        verbose=True,
    ):
        """
        Load and optionally save multiple games as pandas DataFrames.

        Parameters:
        - n_games (int or "all"): Number of games to load, or "all" to load all available games
        - in_play_only (bool): Whether to keep only frames where the game is being played
        - speed (bool): Whether to include player speed
        - z (bool): Whether to include z-coordinates
        - use_artificial_players (bool): Assign players to artificial slots (0–10)
        - every_n (int | None): If set, keep only every n-th frame
        - save (str | False): Directory to save HDF5 files, or False to skip saving
        - verbose (bool or (bool, bool)): First flag for printing info, second for showing plots
        """

        # Normalize verbose flags
        if isinstance(verbose, tuple):
            verbose_info, verbose_plots = verbose
        else:
            verbose_info = verbose_plots = verbose

        datasets = []
        _, sorted_sc = self.sort_games()

        # Handle "all" case for n_games
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
                speed=speed,
                z=z,
                col5=True,  # Hardcoded as always True
                use_artificial_players=use_artificial_players,
                verbose=verbose_info,
            )

            # Subsample by keeping every n-th frame
            if every_n is not None and every_n > 1:
                df = df.iloc[::every_n].reset_index(drop=True)
                if verbose_info:
                    print(f"Downsampled: kept every {every_n}-th frame (rows left: {len(df)})")

            # Save to file if requested
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

            # Verbose output
            if verbose_info:
                print("\n=== DataFrame Overview ===")
                print("Shape:", df.shape)
                print("\nColumns:")
                print(df.columns.tolist())
                print("\nHead of Data:")
                print(df.head(3).T)

            if verbose_plots:
                import seaborn as sns
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 4))
                sns.heatmap(df.isna(), cbar=False)
                plt.title("Missing Data Pattern")
                plt.tight_layout()
                plt.show()

                if "game" in df.columns:
                    plt.figure(figsize=(10, 2))
                    plt.plot(df["game"].values, drawstyle="steps-post", color="black")
                    plt.title("Game State Over Time")
                    plt.xlabel("Frame")
                    plt.ylabel("Game")
                    plt.ylim(-0.1, 1.1)
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    plt.show()
                else:
                    print("[Warning] 'game' column not found in DataFrame.")

        return datasets





def main():
    # === Setup paths ===
    data_dir = "/Users/denizadiguzel/FootballData_FromMathias_May2025/RestructuredData_2425"
    team = "FCK"
    save_dir = "/Users/denizadiguzel/"  # Optional, used if save=True

    # === Initialize loader ===
    loader = FootballDataLoader(data_dir, team)

    # === Load multiple games ===
    datasets = loader.load_all_games(
        n_games= 2,                     # Or use "all" to load everything
        in_play_only=True,            # Only use frames when game is in play
        speed=False,                  # Skip speed column
        z=False,                      # Skip z-coordinate
        use_artificial_players=True,  # Use fixed player slots (0–10)
        every_n=5,                    # Downsample: keep every 5th frame
        save=False,                   # Don’t save to disk
        verbose=(True, False)        # Don’t print shapes or show plots
    )

    # === Inspect shape of each game ===
    for i, df in enumerate(datasets):
        print(f"Game {i}: shape = {df.shape}")

    if False:
        # === Optional: Look at artificial player mapping from the first game ===
        df_first = datasets[0]
        print("\nExample artificial player slots (first game):")
        artificial_number_cols = [col for col in df_first.columns if "_number" in col][:11]
        print(artificial_number_cols)
        print(df_first[artificial_number_cols].head())

#main()

