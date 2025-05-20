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

def ball_scraper(mBall, just_game = True) : 
    """
    Scrape the ball data from the given input.
    
    Args:
        ball: The input data containing ball information.
        
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
        mball3.append(mBall[i][2])
        mball4.append(mBall[i][3])
        mball5.append(mBall[i][4])
        mball6.append(mBall[i][5])
    # Create a new DataFrame with the split columns
    df_ball_split = pd.DataFrame(
        {
            "x": mball1,
            "y": mball2,
            "z": mball3,
            "speed?": mball4,
            "col5": mball5,
            "game": mball6,
        }
    )
    if just_game: 
        ball_column_6 = df_ball_split[df_ball_split["game"] == 1]
        return ball_column_6
    return df_ball_split

def team_scraper(mFcn, name = 'home') : 
    columns = []
    for player in range(11):
        columns.append(name + "player_" + str(player) + "_x")
        columns.append(name + "player_" + str(player) + "_y")
        columns.append(name + "player_" + str(player) + "_z")
        columns.append(name + "player_" + str(player) + "_speed_x")
        columns.append(name + "player_" + str(player) + "_speed_y")
    
    
    player_arrays = [[] for _ in range(11*5)]
    for i in range(len(mFcn)):
        for j in range(11) : 
            player_arrays[j*5].append(mFcn[i][j][0])
            player_arrays[j*5 + 1].append(mFcn[i][j][1])
            player_arrays[j*5 + 2].append(mFcn[i][j][2])
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

# TeamDir = ['FCN','BIF','FCM','FCK','VFF','SIF','RFC','ACH','LYN','AAB','AGF','OB']
TeamDir = ["FCN"]

for TeamA in TeamDir:
    # DD,DirN = GenDir(TeamA);
    NamesXG, NamesSC = SortGames("pippo", TeamA)
    tmax = 10
    RevTime = 30
    PosTime = 20

    for igame in range(1, 2):
        ########################### All game data #############################
        start_time = time.time()

        print("reading files for game: ", igame)
        mTime, mBall, mFcn, mOpp = SecLoad(TeamA, NamesSC, igame)
        time_dataset = time_scraper(mTime)
        ball_dataset = ball_scraper(mBall , just_game = False)
        team_dataset = team_scraper(mFcn)
        opponent_dataset = team_scraper(mOpp, name = 'away')
        total_df = pd.concat([time_dataset, ball_dataset, team_dataset, opponent_dataset], axis=1)
        total_df = total_df[total_df['game'] == 1]

        print("Time dataset shape: ", time_dataset.shape)
        print("Ball dataset shape: ", ball_dataset.shape)
        print("Team dataset shape: ", team_dataset.shape)
        print("Opponent dataset shape: ", opponent_dataset.shape)
        print("Total dataset shape: ", total_df.shape)

        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        
