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
    nr_of_us = len(us)
    nr_of_them = len(them)

    xylist[0] = ball[0]
    xylist[1] = ball[1]
    xylist[2] = ball[2]

    for i in range(11):
        if i >= nr_of_us:
            xylist[i*2+3] = 10000
            xylist[i*2+4] = 0
        else:
            xylist[i*2+3] = us[i][0]
            xylist[i*2+4] = us[i][1]
        if i >= nr_of_them:
            xylist[3+22+i*2] = 10000
            xylist[3+22+i*2+1] = 0
        else:
            xylist[3+22+i*2] = them[i][0]
            xylist[3+22+i*2+1] = them[i][1]
    
    us_x = xylist[3 + np.arange(11) * 2]
    us_y = xylist[3 + np.arange(11) * 2 + 1]

    us_dist = np.sqrt((xylist[0] - us_x) ** 2 + (xylist[1] - us_y) ** 2)

    them_x = xylist[3 + 22 + np.arange(11) * 2]
    them_y = xylist[3 + 22 + np.arange(11) * 2 + 1]

    them_dist = np.sqrt((xylist[0] - them_x) ** 2 + (xylist[1] - them_y) ** 2)

    us_dist = sorted(us_dist)
    them_dist = sorted(them_dist)

    interleaved = np.array([val for pair in zip(us_dist, them_dist) for val in pair])
    interleaved = [val if val < 10000 else np.nan for val in interleaved]  # Replace 10000 with a large value to avoid outliers

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
    new_params = {'boosting_type': 'dart', 'learning_rate': np.float64(0.06630205132311011), 'max_depth': 8, 'num_leaves': 38, 'n_estimators': 328}
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

NamesXG, NamesSC = SortGames('pippo', 'AGF')
_, ball, us, them = SecLoad('AGF', NamesSC,10)
index=152847

generate_XG(meow,ball[index], us[index], them[index])

    
#0.04618892114190254 best, 