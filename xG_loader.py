import pickle
import os
import matplotlib.pyplot as plt
from ML_Functions_LoadArrays import *
import numpy as np

def get_frames_xg(mTime_masked, all_times, goals): 
    '''
    Gives the begining and end indices of the 10 seconds before and 1 second after the shot is taken. Indices of the mTime_masked list.

    inputs:
    mTime_masked (np.array): np.array(AllData["Times"][mask]), where mask is if ball is being played
    all_times (np.array): np.array of the times of the shots taken, from the XG data
    goals (np.array): np.array of the goals scored, from the XG data

    outputs:
    first_frames (np.array): np.array of the begining indices of the 10 seconds before the shot is taken
    final_frames (np.array): np.array of the end indices of the 1 second after the shot is taken
    '''
    
    # need: goals = data2[10], mTime_masked = np.array(data3["Times"][mask]), mask = played.astype(bool)

    final_frames = np.zeros(len(all_times), dtype=int)
    k=0
    for target_time in all_times:
        if bool(goals[k]):
            final_frames[k] = [i for i, t in enumerate(mTime_masked) if abs(t[0] -target_time) < 1.1][-1]  
        else:
            for i, t in enumerate(mTime_masked):
                if t[0] - target_time > 1:  #So we take the time of the shot according to XG, and take an extra second
                    break
            final_frames[k] = i 
        k+=1

    first_frames = final_frames - 11 * 25
    return first_frames, final_frames


def scrape_xg(Team , number_of_games) :
    Xg_our_team = pd.DataFrame(columns=['XG', 'Half', 'Time', 'Team', 'Opponent'])
    Xg_opponent = pd.DataFrame(columns=['XG', 'Half', 'Time', 'Team', 'Opponent'])
    NamesXG, NamesSC = SortGames('pippo', Team)
    for igame in range(number_of_games) :
        ########################### All game data #############################
        #mTime, mBall, mFcn, mOpp = SecLoad(Team, NamesSC, igame)
        print("igame", igame, NamesXG[igame])
        ######################### Event data ###################################
        XGNumbers, XGTeam, XGValue, XGHalf, XGMin, XGSec, XGTimes, XGPos1, XGPos2 = (
            MacihneLearning_OptaLoad(Team, NamesXG, igame)
        )
        
        XGTeam = np.array(XGTeam, dtype=str)
        XGValue = np.array(XGValue, dtype=float)
        for ii in range(len(XGValue)) :
            if XGValue[ii] < 0 or XGValue[ii] > 1 or np.isnan(XGValue[ii]):
                print("XGValue < 0 or > 1 or NaN:", XGValue[ii])
                print("XGTeam", XGTeam[ii])
                print("XGHalf", XGHalf[ii])
                print("XGMin", XGMin[ii])
                print("XGSec", XGSec[ii])
        if len(XGValue) == 0:
            print("No XG data for this game: ", NamesXG[igame])
            continue
        XGHalf = np.array(XGHalf, dtype=int)
        XGTimes = np.array(XGTimes, dtype=float)
        mask_team_1 = XGTeam == Team
        mask_team_2 = ~mask_team_1
        Xg_our_team = pd.concat([
            Xg_our_team,
            pd.DataFrame({
            'XG': XGValue[mask_team_1],
            'Half': XGHalf[mask_team_1],
            'Time': XGTimes[mask_team_1],
            'Team': [XGTeam[mask_team_1][0] if mask_team_1.any() else None] * np.sum(mask_team_1),
            'Opponent': [XGTeam[mask_team_2][0] if mask_team_2.any() else None] * np.sum(mask_team_1)
            })
        ], ignore_index=True)
        Xg_opponent = pd.concat([
            Xg_opponent,
            pd.DataFrame({
            'XG': XGValue[mask_team_2],
            'Half': XGHalf[mask_team_2],
            'Time': XGTimes[mask_team_2],
            'Team': [XGTeam[mask_team_2][0] if mask_team_2.any() else None] * np.sum(mask_team_2),
            'Opponent': [XGTeam[mask_team_1][0] if mask_team_1.any() else None] * np.sum(mask_team_2)
            })
        ], ignore_index=True)
    Xg_our_team['XG'] = Xg_our_team['XG'].astype(float , errors='raise')
    Xg_our_team = Xg_our_team[Xg_our_team['XG'] >= 0]
    Xg_our_team['Half'] = Xg_our_team['Half'].astype(int , errors='raise')
    Xg_our_team['Time'] = Xg_our_team['Time'].astype(float , errors='raise')
    Xg_opponent['XG'] = Xg_opponent['XG'].astype(float , errors='raise')
    Xg_opponent = Xg_opponent[Xg_opponent['XG'] >= 0]
    Xg_opponent['Half'] = Xg_opponent['Half'].astype(int , errors='raise')
    Xg_opponent['Time'] = Xg_opponent['Time'].astype(float , errors='raise')
    return Xg_our_team, Xg_opponent


def get_all_xg(Team, verbose=False):
    NamesXG, NamesSC = SortGames('pippo', Team)
    columns = ['XG', 'Time', 'Half','ball_x', 'ball_y', 'ball_z', 'ball_speed']
    for i in range(1, 12):
        columns += [f'us_{i}_x', f'us_{i}_y', f'us_{i}_speed']
    for i in range(1, 12):
        columns += [f'them_{i}_x', f'them_{i}_y', f'them_{i}_speed']

    rows = []
    for igame in range(30) :
        Time, Ball, Us, Them = SecLoad(Team, NamesSC, igame)

        XGNumbers, XGTeam, XGValue, XGHalf, XGMin, XGSec, XGTimes, XGPos1, XGPos2 = (
            MacihneLearning_OptaLoad(Team, NamesXG, igame)
        )

        times_x = np.array([t[0] for t in Time])
        
        for i, t in enumerate(XGTimes):
            if XGTeam[i] == Team and XGValue[i] > 0:
                idxs = np.where(np.isclose(times_x, t, atol=0.05))[0]
                if len(idxs) != 0: #time of XG is 'wrong; 
                    idx = idxs[0]
                    k=0
                    sign = 1
                    passed = True
                    while Ball[idx][5] == 0:
                        idx += sign*1
                        if k == 25:
                            if verbose:
                                print('Cannot find the ball being played in this second')
                            idx = idxs[0]
                            sign = -1
                        elif k > 75:
                            if verbose:
                                print('Nor 2 seconds before, so not including shot')
                            passed = False
                            break
                        k+=1
        
                    if passed:
                        if len(Us[idx]) < 11 or len(Them[idx]) < 11:
                            if verbose:
                                print(f"Red carded in game {igame}") 
                                #If this prints multiple times, it means theres multiple shots where a person is missing
        
                            continue
                        else:
                            row = {
                                'XG': XGValue[i],
                                'Time': XGTimes[i],
                                'Half': XGHalf[i],
                                'ball_x': Ball[idx][0],
                                'ball_y': Ball[idx][1],
                                'ball_z': Ball[idx][2],
                                'ball_speed': Ball[idx][3],
                            }
                            for j in range(11):
                                row[f'us_{j+1}_x'] = Us[idx][j][0]
                                row[f'us_{j+1}_y'] = Us[idx][j][1]
                                row[f'us_{j+1}_speed'] = Us[idx][j][3]
                            for j in range(11):
                                row[f'them_{j+1}_x'] = Them[idx][j][0]
                                row[f'them_{j+1}_y'] = Them[idx][j][1]
                                row[f'them_{j+1}_speed'] = Them[idx][j][3]
                            rows.append(row)

    total_XG_data = pd.DataFrame(rows, columns=columns)
    return total_XG_data



tXG_all = get_all_xg('AAB', verbose=False) #1 team takes +- 1 min 20 sec
print("Team AAB done")

for team in ['AGF', 'BIF', 'FCK', 'FCM', 'FCN', 'LYN', 'RFC', 'SIF', 'SJE', 'VB', 'VFF']:
    tXG = get_all_xg(team) 
    print(f"Team {team} done")
    tXG_all = pd.concat([tXG_all, tXG], ignore_index=True)
    