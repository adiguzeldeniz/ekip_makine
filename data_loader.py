import os
import numpy as np
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
        xg_dates = [
            int("".join(f.split("Day_20")[1].split("Z.txt")[0].split("-")))
            for f in xg_files
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

    def load_machine_learning_xg(self, filename):
        path = os.path.join(self.xg_data_path, filename)
        values = {
            "value": [],
            "half": [],
            "minute": [],
            "second": [],
            "number": [],
            "team": [],
            "x": [],
            "y": [],
        }
        with open(path, "r") as f:
            for line in f:
                v = line.strip().split(",")
                values["value"].append(float(v[0]))
                values["half"].append(int(v[1]))
                values["minute"].append(int(v[2]))
                values["second"].append(int(v[3]))
                values["number"].append(int(v[5]))
                values["team"].append(v[6])
                values["x"].append(v[7])
                values["y"].append(v[8])
        values["time"] = [
            m * 60 + s for m, s in zip(values["minute"], values["second"])
        ]
        return values

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

    def scrape_team(self, mFcn, name="home", speed=True, z=True):
        cols = []
        numbers = np.zeros(mFcn[0].shape[0], dtype=int)

        # reading the number to inizialize corretly the player
        for i in range(mFcn[0].shape[0]):
            numbers[i] = mFcn[0][i][4]

        for i in numbers:
            cols += [f"{name}player_{i}_x", f"{name}player_{i}_y"]
            if z:
                cols.append(f"{name}player_{i}_z")
            if speed:
                cols += [f"{name}player_{i}_speed_x"]
            cols.append(f"{name}player_{i}_number")

        df = pd.DataFrame(columns=cols)
        # Initialize the DataFrame with NaN values
        for col in df.columns:
            df[col] = np.nan

        # Fill the DataFrame with player data
        for idx, frame in enumerate(mFcn):
            player_on_the_field = frame.shape[0]
            players = frame[:, 4]
            # find the missing players
            missing_players = [i for i in numbers if i not in players]
            remaining_players = [i for i in numbers if i not in missing_players]

            # Fill data for players on the field
            for i in range(player_on_the_field):
                number = int(frame[i][4])
                if number in remaining_players:
                    col_prefix = f"{name}player_{number}"
                    df.at[idx, f"{col_prefix}_x"] = frame[i][0]
                    df.at[idx, f"{col_prefix}_y"] = frame[i][1]
                    if z:
                        df.at[idx, f"{col_prefix}_z"] = frame[i][2]
                    if speed:
                        df.at[idx, f"{col_prefix}_speed_x"] = frame[i][3]
                    df.at[idx, f"{col_prefix}_number"] = frame[i][4]

            # Fill NaN for missing players
            for missing_player in missing_players:
                col_prefix = f"{name}player_{missing_player}"
                df.at[idx, f"{col_prefix}_x"] = np.nan
                df.at[idx, f"{col_prefix}_y"] = np.nan
                if z:
                    df.at[idx, f"{col_prefix}_z"] = np.nan
                if speed:
                    df.at[idx, f"{col_prefix}_speed_x"] = np.nan
                df.at[idx, f"{col_prefix}_number"] = missing_player

        return df

    def scrape_time(self, mTime):
        return pd.DataFrame(
            {"Time": [x[0] for x in mTime], "half": [x[1] for x in mTime]}
        )

    def scrape_game(
        self, filename, just_game=True, speed=True, z=True, col5=True, verbose=True
    ):
        mTime, mBall, mFcn, mOpp = self.load_sec_game(filename)
        home_name, away_name = self.extract_teams_from_filename(filename)
        df_time = self.scrape_time(mTime)
        df_ball = self.scrape_ball(mBall, just_game=False, speed=speed, col5=col5, z=z)
        df_team = self.scrape_team(mFcn, name=home_name, speed=speed, z=z)
        df_opp = self.scrape_team(mOpp, name=away_name, speed=speed, z=z)

        df = pd.concat([df_time, df_ball, df_team, df_opp], axis=1)
        if just_game:
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
        just_game=True,
        speed=True,
        z=True,
        col5=True,
        save=False,
        verbose=True,
    ):
        datasets = []
        _, sorted_sc = self.sort_games()

        for i, filename in enumerate(sorted_sc[:n_games]):
            print(f"Reading game {i+1}: {filename}")
            df = self.scrape_game(filename, just_game, speed, z, col5, verbose)

            if save:
                os.makedirs(save, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(filename))[0]
                # save_path = os.path.abspath(os.path.join(save, base_name + ".hdf5"))
                save_path = os.path.abspath(
                    os.path.join(save, base_name + ".hdf")
                )  # ✅ Correct

                print(f"Attempting to save to: {save_path}")

                if os.path.exists(save_path):
                    print(f"[Warning] File already exists, deleting: {save_path}")
                    os.remove(save_path)

                try:
                    df.to_hdf(save_path, key="df", mode="w")
                    print(f"Saved to {save_path}")
                except Exception as e:
                    print(f"[ERROR] Saving failed: {e}")

            # ✅ Always append the DataFrame, even if not saving
            datasets.append(df)

        return datasets
