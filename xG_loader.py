import pickle
import os
import matplotlib.pyplot as plt
from ML_Functions_LoadArrays import *
import numpy as np

def get_all_xg(Team, verbose=False):
    broken_NamesSC = ['Game_SJE_BIF_Score_2_2_Day_2024-11-24Z.pkl', 'Game_SJE_VFF_Score_2_2_Day_2024-09-01Z.pkl', 'Game_SJE_FCN_Score_1_4_Day_2024-10-06Z.pkl']
    NamesXG, NamesSC = SortGames('pippo', Team)
    columns = ['XG', 'Time', 'Half','ball_x', 'ball_y', 'ball_z', 'ball_speed']
    for i in range(1, 12):
        columns += [f'us_{i}_x', f'us_{i}_y', f'us_{i}_speed']
    for i in range(1, 12):
        columns += [f'them_{i}_x', f'them_{i}_y', f'them_{i}_speed']

    rows = []
    for igame in range(30):
        if NamesSC[igame] not in broken_NamesSC:
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



tXG_all = get_all_xg('AAB', verbose=False) #1 team takes +- 1 min 
print("Team AAB done")

for team in ['AGF', 'BIF', 'FCK', 'FCM', 'FCN', 'LYN', 'RFC', 'SIF', 'SJE', 'VB', 'VFF']:
    tXG = get_all_xg(team) 
    print(f"Team {team} done")
    tXG_all = pd.concat([tXG_all, tXG], ignore_index=True)

#tXG_all.to_csv("all_XG_data.csv", index=False)
tXG_all.head(10)

    