import os
import csv
import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc
from matplotlib import pyplot as plt, patches
from numpy import savetxt
import seaborn as sns

from tabulate import tabulate
from ML_Functions_LoadArrays import *

from ML_Functions_KeyNumbersAgorithms import *
import time

def ball_scraper(mBall, just_game = True, speed = True , col5 = True, z = True): 
    """
    Scrape the ball data from the given input.
    
    Args:
        ball: The input data containing ball information.
        just_game: If True, only return the game data.
        speed: If True, include the speed of the ball.
        col5: If True, include the additional column 5.
        z: If True, include the Z-coordinate of the ball.

    Returns:
        Dataset of ball with the following columns:
        - x: X-coordinate of the ball.
        - y: Y-coordinate of the ball.
        - z: Z-coordinate of the ball.
        - speed?: Speed of the ball.
        - col5: Additional column 5 (to investigate).
        - game: is the game active or not.(defualt function just returns 1)
    """
    mball1, mball2, mball3, mball4, mball5, mball6 = [], [], [], [], [], []
    for i in range(len(mBall)):
        mball1.append(mBall[i][0])
        mball2.append(mBall[i][1])
        if z:
            mball3.append(mBall[i][2])
        if speed:
            mball4.append(mBall[i][3])
        if col5:
            mball5.append(mBall[i][4])
        mball6.append(mBall[i][5])
    # Create a new DataFrame with the split columns
    if speed and col5 and z:
        df_ball_split = pd.DataFrame(
            {
                "Ball_x": mball1,
                "Ball_y": mball2,
                "Ball_z": mball3,
                "Ball_Speed?": mball4,
                "Ball_Col5": mball5,
                "game": mball6,
            }
        )
    elif speed and not col5 and z:
        df_ball_split = pd.DataFrame(
            {
                "Ball_x": mball1,
                "Ball_y": mball2,
                "Ball_z": mball3,
                "Ball_Speed?": mball4,
                "game": mball6,
            }
        )
    elif not speed and col5 and z:
        df_ball_split = pd.DataFrame(
            {
                "Ball_x": mball1,
                "Ball_y": mball2,
                "Ball_z": mball3,
                "Ball_Col5": mball5,
                "game": mball6,
            }
        )
    elif speed and col5 and not z:
        df_ball_split = pd.DataFrame(
            {
                "Ball_x": mball1,
                "Ball_y": mball2,
                "Ball_Speed?": mball4,
                "Ball_Col5": mball5,
                "game": mball6,
            }
        )
    elif not speed and not col5 and z:
        df_ball_split = pd.DataFrame(
            {
                "Ball_x": mball1,
                "Ball_y": mball2,
                "Ball_z": mball3,
                "game": mball6,
            }
        )
    elif speed and not col5 and not z:
        df_ball_split = pd.DataFrame(
            {
                "Ball_x": mball1,
                "Ball_y": mball2,
                "Ball_Speed?": mball4,
                "game": mball6,
            }
        )
    elif not speed and col5 and not z:
        df_ball_split = pd.DataFrame(
            {
                "Ball_x": mball1,
                "Ball_y": mball2,
                "Ball_Col5": mball5,
                "game": mball6,
            }
        )
    elif not speed and not col5 and not z:
        df_ball_split = pd.DataFrame(
            {
                "Ball_x": mball1,
                "Ball_y": mball2,
                "game": mball6,
            }
        )
    
    if just_game: 
        ball_column_6 = df_ball_split[df_ball_split["game"] == 1]
        return ball_column_6
    return df_ball_split

def team_scraper(mFcn, name = 'home', speed = True, z = True) :
    """
    Scrape the team data from the given input.
    Args:
        mFcn: The input data containing team information.
        name: The name of the team (default is 'home').
        speed: If True, include the speed of the players.
        z: If True, include the Z-coordinate of the players.
    Returns:
        Dataset of team with the following columns:
        - x: X-coordinate of each the player.
        - y: Y-coordinate of each the player.
        - z: Z-coordinate of each the player.
        - speed_x: Speed in the X-direction.
        - speed_y: Speed in the Y-direction.
    """ 
    columns = []
    for player in range(11):
        columns.append(name + "player_" + str(player) + "_x")
        columns.append(name + "player_" + str(player) + "_y")
        if z:
            columns.append(name + "player_" + str(player) + "_z")
        if speed:
            columns.append(name + "player_" + str(player) + "_speed_x")
            columns.append(name + "player_" + str(player) + "_speed_y")

    player_arrays = [[] for _ in range(11*5)]
    for i in range(len(mFcn)):
        for j in range(11) : 
            player_arrays[j*5].append(mFcn[i][j][0])
            player_arrays[j*5 + 1].append(mFcn[i][j][1])
            if z:
                player_arrays[j*5 + 2].append(mFcn[i][j][2])
            if speed:
                player_arrays[j*5 + 3].append(mFcn[i][j][3])
                player_arrays[j*5 + 4].append(mFcn[i][j][4])
    
    
    df = pd.DataFrame(player_arrays).T
    df.columns = columns
    return df

def time_scraper(mTime) : 
    time = []
    half = []
    for i in range(len(mTime)) : 
        time.append(mTime[i][0])
        half.append(mTime[i][1])
    
    df = pd.DataFrame(
        {
            "Time": time,
            "half": half,
        }
    )
    return df


def game_scraper(mFcn, mTime, mBall, mOpp, just_game = True, speed = True, z = True, col5 = True, save = True, verbose = True) :
    """
    Scrape the game data from the given input.
    
    Args:
        mFcn: The input data containing team information.
        mTime: The input data containing time information.
        mBall: The input data containing ball information.
        mOpp: The input data containing opponent information.
        just_game: If True, only return the game data.
        speed: If True, include the speed of the players and the ball.
        z: If True, include the Z-coordinate of the players and the ball.
        col5: If True, include the additional column 5.
        save: If True, save the DataFrame to an HDF5 file.
        verbose: If True, print the shapes of the datasets.

    Returns:
        Dataset of game with the following columns:
        - x: X-coordinate of each the player.
        - y: Y-coordinate of each the player.
        - z: Z-coordinate of each the player.
        - speed_x: Speed in the X-direction.
        - speed_y: Speed in the Y-direction.
    """

    time_dataset = time_scraper(mTime)
    ball_dataset = ball_scraper(mBall , just_game = False , speed = speed, col5 = col5, z = z)
    team_dataset = team_scraper(mFcn, name = 'home', speed = speed, z = z)
    opponent_dataset = team_scraper(mOpp, name = 'away', speed = speed, z = z)
    total_df = pd.concat([time_dataset, ball_dataset, team_dataset, opponent_dataset], axis=1)

    if just_game:
        total_df = total_df[total_df['game'] == 1]

    if verbose:
        print("Time dataset shape: ", time_dataset.shape)
        print("Ball dataset shape: ", ball_dataset.shape)
        print("Team dataset shape: ", team_dataset.shape)
        print("Opponent dataset shape: ", opponent_dataset.shape)
        print("Total dataset shape: ", total_df.shape)

    return total_df


def Load_games(TeamDir, n_games = 1, just_game = True, speed = True, z = True, col5 = True, save = True, verbose = True):
    """
    Load and scrape game data for a list of teams.
    
    Args:
        TeamDir: List of team directories.
        just_game: If True, only return the game data.
        speed: If True, include the speed of the players and the ball.
        z: If True, include the Z-coordinate of the players and the ball.
        col5: If True, include the additional column 5.
        save: If True, save the DataFrame to an HDF5 file.
        verbose: If True, print the shapes of the datasets.

    Returns:
        array of datasets
    """
    datasets = []
    for TeamA in TeamDir:
        NamesXG, NamesSC = SortGames("Deniz we love you", TeamA)

        for igame in range(1, n_games + 1):
            if verbose:
                print("reading files for game: ", igame)
            ########################### All game data #############################
            start_time = time.time()
            mTime, mBall, mFcn, mOpp = SecLoad(TeamA, NamesSC, igame)
            game_df = game_scraper(mFcn, mTime, mBall, mOpp, just_game=just_game, speed=speed, z=z, col5=col5, save=save, verbose=verbose)
            datasets.append(game_df)
            if save:
                # Save the DataFrame to an HDF5 file
                hdf_filename = NamesSC[igame].replace('.pkl', '') + ".hdf"
                game_df.to_hdf(hdf_filename, key="df", mode="w")
                print(f"Data saved to {hdf_filename}")

            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return datasets


Team = ['FCN']
Load_games(Team, n_games = 1, just_game = True, speed = True, z = True, col5 = True, save = True, verbose = True)
