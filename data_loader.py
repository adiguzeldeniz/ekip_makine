import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from tqdm import tqdm

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

    def scrape_game(
        self,
        filename,
        in_play_only=True,
        speed_ball=True,
        speed_player=True,
        ball_z=True,
        player_z=True,
        col5=True,
        use_artificial_players=False,
        artificial_opponent=False,  # <-- NEW OPTION
        verbose=True
    ):
        # Load raw arrays
        mTime, mBall, mTeam, mOpp = self.load_sec_game(filename)

        # Extract team names
        team1, team2 = self.extract_teams_from_filename(filename)
        opponent_name = team2 if self.team == team1 else team1

        # Time and ball
        df_time = self.scrape_time(mTime)
        df_ball = self.scrape_ball(mBall, just_game=False, speed=speed_ball, col5=col5, z=ball_z)

        # Your team uses real or artificial players based on use_artificial_players
        df_team = self.scrape_team(
            mTeam,
            name=self.team,
            speed=speed_player,
            z=player_z,
            use_artificial_players=use_artificial_players
        )

        # Opponent uses artificial slots if artificial_opponent is True
        df_opp = self.scrape_team(
            mOpp,
            name="OPP",
            speed=speed_player,
            z=player_z,
            use_artificial_players=artificial_opponent
        )

        # Combine all
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
        use_artificial_players=False,
        artificial_opponent=False,  # <-- NEW
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
                speed_ball=speed_ball,
                speed_player=speed_player,
                ball_z=ball_z,
                player_z=player_z,
                col5=True,
                use_artificial_players=use_artificial_players,
                artificial_opponent=artificial_opponent,  # <-- NEW
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

    def load_game_for_cluster(self, max_games=None):
        """
        Loads unique games across all teams and extracts player positions and speeds
        with jersey-based IDs. x and y positions are normalized to [-1, 1] per game.

        Parameters:
            max_games (int or None): Max number of unique games to load (for testing/debug)

        Returns:
            pd.DataFrame with columns: ["player_id", "x", "y", "z", "speed"]
        """
        import pickle

        all_rows = []
        seen_games = set()
        loaded = 0

        root_dir = os.path.dirname(self.team_path)
        clubs = sorted(os.listdir(root_dir))

        for club in clubs:
            club_path = os.path.join(root_dir, club, "AllData")
            if not os.path.isdir(club_path):
                continue

            for fname in os.listdir(club_path):
                if not fname.endswith(".pkl"):
                    continue

                parts = fname.split("_")
                if len(parts) < 6:
                    continue

                # Unique game ID to avoid duplicates
                team1, team2 = parts[1], parts[2]
                date = parts[-1].replace("Day_", "").replace("Z.pkl", "")
                game_id = f"{team1}_{team2}_{date}"

                if game_id in seen_games:
                    continue
                seen_games.add(game_id)

                print(f"Processing {club}: {fname}")
                fpath = os.path.join(club_path, fname)
                try:
                    with open(fpath, "rb") as f:
                        M = pickle.load(f)
                except Exception as e:
                    print(f"[Error] Could not load {fpath}: {e}")
                    continue

                if club not in M:
                    print(f"[Warning] Club {club} not found in {fname}")
                    continue

                mTeam = M[club]

                try:
                    x_all = np.concatenate([frame[:, 0] for frame in mTeam])
                    y_all = np.concatenate([frame[:, 1] for frame in mTeam])
                    if len(x_all) == 0 or len(y_all) == 0:
                        print(f"[Skip] No player data in {fname}")
                        continue
                    x_min, x_max = np.min(x_all), np.max(x_all)
                    y_min, y_max = np.min(y_all), np.max(y_all)
                except Exception as e:
                    print(f"[Warning] Failed to compute bounds for {fname}: {e}")
                    continue

                def normalize(val, min_val, max_val):
                    if max_val - min_val == 0:
                        return 0.0
                    return 2 * (val - min_val) / (max_val - min_val) - 1

                for frame in mTeam:
                    for player in frame:
                        jersey = int(player[4])
                        player_id = f"player_{club}_{jersey}"
                        x = normalize(player[0], x_min, x_max)
                        y = normalize(player[1], y_min, y_max)
                        z = player[2]
                        speed = player[3]
                        all_rows.append([player_id, x, y, z, speed])

                loaded += 1
                if max_games is not None and loaded >= max_games:
                    print(f"[Info] Reached max_games limit ({max_games})")
                    break

            if max_games is not None and loaded >= max_games:
                break

        df = pd.DataFrame(all_rows, columns=["player_id", "x", "y", "z", "speed"])
        return df


import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm


import os
import numpy as np
import pandas as pd


class MultipleFootballDataLoader:
    def __init__(self, data_dir, teams):
        self.data_dir = data_dir.rstrip("/")
        if isinstance(teams, str):
            self.teams = [t.strip() for t in teams.split(",")]
        elif isinstance(teams, list):
            self.teams = teams
        else:
            raise ValueError("Teams should be a comma-separated string or list of team names.")

    def load_game_for_cluster(self, max_games=None, player_z=True, every_n=1, in_play_only=True, save_path=None):
        import pickle
        from tqdm import tqdm

        all_rows = []
        seen_games = set()
        loaded = 0

        clubs = sorted(self.teams)

        for club in tqdm(clubs, desc="Teams"):
            club_path = os.path.join(self.data_dir, club, "AllData")
            if not os.path.isdir(club_path):
                continue

            game_files = [f for f in os.listdir(club_path) if f.endswith(".pkl")]

            for fname in tqdm(game_files, desc=f"Games ({club})", leave=False):
                parts = fname.split("_")
                if len(parts) < 6:
                    continue

                team1, team2 = parts[1], parts[2]
                date = parts[-1].replace("Day_", "").replace("Z.pkl", "")
                game_id = f"{team1}_{team2}_{date}"

                if game_id in seen_games:
                    continue
                seen_games.add(game_id)

                fpath = os.path.join(club_path, fname)
                try:
                    with open(fpath, "rb") as f:
                        M = pickle.load(f)
                except Exception as e:
                    print(f"[Error] Could not load {fpath}: {e}")
                    continue

                if club not in (team1, team2):
                    continue
                if club not in M:
                    print(f"[Warning] Skipping {game_id}: No data found for {club}")
                    continue

                mTime = M.get("Times")
                mTeam = M[club]

                if in_play_only and mTime is not None:
                    in_play_mask = [x[1] == 1 for x in mTime]
                    mTime = [x[0] for x, play in zip(mTime, in_play_mask) if play]
                    mTeam = [frame for frame, play in zip(mTeam, in_play_mask) if play]
                else:
                    mTime = [x[0] for x in mTime] if mTime is not None else list(range(len(mTeam)))

                if every_n > 1:
                    mTime = mTime[::every_n]
                    mTeam = mTeam[::every_n]

                try:
                    x_all = np.concatenate([frame[:, 0] for frame in mTeam])
                    y_all = np.concatenate([frame[:, 1] for frame in mTeam])
                    if len(x_all) == 0 or len(y_all) == 0:
                        continue
                    x_min, x_max = np.min(x_all), np.max(x_all)
                    y_min, y_max = np.min(y_all), np.max(y_all)
                except Exception as e:
                    print(f"[Warning] Failed to compute bounds for {fname}: {e}")
                    continue

                def normalize(val, min_val, max_val):
                    if max_val - min_val == 0:
                        return 0.0
                    return 2 * (val - min_val) / (max_val - min_val) - 1

                for i in range(1, len(mTeam)):
                    frame_prev = mTeam[i - 1]
                    frame_curr = mTeam[i]
                    t_prev = mTime[i - 1]
                    t_curr = mTime[i]
                    dt = t_curr - t_prev
                    if dt == 0:
                        continue

                    jersey_map = {int(player[4]): player for player in frame_prev}
                    for player in frame_curr:
                        jersey = int(player[4])
                        player_id = f"player_{club}_{jersey}"
                        if jersey not in jersey_map:
                            continue

                        x_curr, y_curr = player[0], player[1]
                        x_prev, y_prev = jersey_map[jersey][0], jersey_map[jersey][1]

                        x_n = normalize(x_curr, x_min, x_max)
                        y_n = normalize(y_curr, y_min, y_max)
                        x_prev_n = normalize(x_prev, x_min, x_max)
                        y_prev_n = normalize(y_prev, y_min, y_max)

                        v_x = (x_n - x_prev_n) / dt
                        v_y = (y_n - y_prev_n) / dt

                        if player_z:
                            z = player[2]
                            all_rows.append([player_id, x_n, y_n, z, player[3], game_id, v_x, v_y])
                        else:
                            all_rows.append([player_id, x_n, y_n, player[3], game_id, v_x, v_y])

                loaded += 1
                if max_games is not None and loaded >= max_games:
                    print(f"[Info] Reached max_games limit ({max_games})")
                    break

            if max_games is not None and loaded >= max_games:
                break

        columns = ["player_id", "x", "y"] + (["z"] if player_z else []) + ["speed", "match_id", "v_x", "v_y"]
        df = pd.DataFrame(all_rows, columns=columns)

        if save_path:
            try:
                df.to_hdf(save_path, key="df", mode="w")
                print(f"[Saved] DataFrame saved to {save_path}")
            except Exception as e:
                print(f"[Error] Could not save to {save_path}: {e}")

        return df





def main():
    # === Setup paths ===
    data_dir = "/Users/denizadiguzel/FootballData_FromMathias_May2025/RestructuredData_2425"
    team = "FCK"

    # === Initialize loader ===
    loader = FootballDataLoader(data_dir, team)

    # === Load two games with opponent artificial slot mapping only ===
    datasets = loader.load_all_games(
        n_games=2,
        in_play_only=True,
        speed_ball=False,
        speed_player=False,
        ball_z=False,
        player_z=False,
        use_artificial_players=False,     
        artificial_opponent=True,           
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



