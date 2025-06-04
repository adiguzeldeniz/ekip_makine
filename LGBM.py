import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from ML_Functions_LoadArrays import *


def generate_XG(gbm_best, ball, us, them):
    '''

    inputs:
    picke_row (np.array): [ball_x, ball_y, ball_z, us_1_x, us_1_y, us_ ..., them_11_y] (if speed is there, us_1_y, us_1_speed )
    speed (bool): whether speed is included in the input features 
    '''

    xylist = np.zeros(3+ 22 + 22)

    xylist[0] = ball[0]
    xylist[1] = ball[1]
    xylist[2] = ball[2]

    for i in range(11):
        print(i*2+3, i*2+4)
        xylist[i*2+3] = us[i][0]
        xylist[i*2+4] = us[i][1]
    for i in range(11):
        xylist[3+22+i*2] = them[i][0]
        xylist[3+22+i*2+1] = them[i][1]

    xylist = np.array(xylist)
   
    us_dist = np.zeros(11)
    them_dist = np.zeros(11)
    for i in range(11):
        us_dist[i]=np.sqrt((xylist[0] -xylist[3+i*2])**2+ (xylist[1]-xylist[3+i*2+1])**2)  # player_x^2 + player_y^2
        them_dist[i] = np.sqrt((xylist[0] -xylist[3+22+i*2])**2+ (xylist[1]-xylist[3+22+i*2+1])**2)
    us_dist = sorted(us_dist)
    them_dist = sorted(them_dist)

    interleaved = np.array([val for pair in zip(us_dist, them_dist) for val in pair])

    lgbminput = np.concatenate((xylist[:3], interleaved))
    lgbminput = lgbminput.reshape(1, -1)
    feature_names = ['ball_x', 'ball_y', 'ball_z'] + \
    [f'us_{i}_ball_dist' for i in range(1, 12)] + \
    [f'them_{i}_ball_dist' for i in range(1, 12)]
    lgbminput_df = pd.DataFrame(lgbminput, columns=feature_names)
    xg = gbm_best.predict(lgbminput_df)    
    xg = np.exp(xg[0]) 
    return xg

def get_gbm_best(df):
    new_params =  {'boosting_type': 'dart', 'learning_rate': np.float64(0.06630205132311011), 'max_depth': 8, 'num_leaves': 38, 'n_estimators': 328}
    lgbm_best = LGBMRegressor(**new_params, objective='regression', verbose=-1)
    variables = ['ball_x', 'ball_y', 'ball_z'] + [f'us_{i}_ball_dist' for i in range(1, 12)] + [f'them_{i}_ball_dist' for i in range(1, 12)]
    X = df[variables]
    y_log = np.log(df['XG'])
    lgbm_best.fit(X, y_log)
    print("Model trained with best parameters.")
    print(f"Model score: {lgbm_best.score(X, y_log)}")
    return lgbm_best

test_row = [0, 0, 0]
test_row += [item for i in range(22) for item in (np.random.randint(0, 100), np.random.randint(0, 60), np.random.randint(0, 80))]  # Example row with ball_x=0, ball_y=0, ball_z=0 and players at (1,2), (3,4), ..., (21,22)  

df = pd.read_csv('LightGBM_train_data.csv')
meow = get_gbm_best(df)

NamesXG, NamesSC = SortGames('pippo', 'AAB')
_, ball, us, them = SecLoad('AAB', NamesSC,9)
generate_XG(meow,ball[200], us[200], them[200])

    
#0.04618892114190254 best, 