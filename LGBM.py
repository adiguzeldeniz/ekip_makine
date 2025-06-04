import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

df = pd.read_csv('LightGBM_train_data.csv')

def generate_XG(gbm_best, pickle_row, speed=True):
    '''

    inputs:
    picke_row (np.array): [ball_x, ball_y, ball_z, us_1_x, us_1_y, us_ ..., them_11_y] (if speed is there, us_1_y, us_1_speed )
    speed (bool): whether speed is included in the input features 
    '''

    if speed:
        xylist = [pickle_row[0], pickle_row[1], pickle_row[2]]  # ball_x, ball_y, ball_z

        xylist += [item for i in range(22) for item in (pickle_row[3 + i * 3], pickle_row[4 + i * 3])]
        xylist = np.array(xylist)
    else:
        xylist = pickle_row.copy()

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
    xg = gbm_best.predict(lgbminput)
    xg = np.exp(xg[0]) 
    print(f"Predicted XG: {xg}")
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
generate_XG(get_gbm_best(df), np.array(test_row), speed=True)

    
#0.04618892114190254 best, 