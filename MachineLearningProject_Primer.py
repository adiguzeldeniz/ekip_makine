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
from Functions_Plot import *
from tabulate import tabulate
from ML_Functions_LoadArrays import *
#from Functions_GenerateFiles import *
from ML_Functions_KeyNumbersAgorithms import *
#from Functions_DefinePlayAlgorithms import *
#############################################################

#TeamDir = ['FCN','BIF','FCM','FCK','VFF','SIF','RFC','ACH','LYN','AAB','AGF','OB']
TeamDir = ['FCN']

for TeamA in TeamDir:    
    DD,DirN = GenDir(TeamA);
    NamesXG,NamesSC = SortGames(DirN,TeamA);
    tmax = 10; RevTime = 30; PosTime = 20;
    
    for igame in range(32):
        ########################### All game data #############################                                                                                                   
        mTime,mBall,mFcn,mOpp = SecLoad(TeamA,NamesSC,igame)
        print('igame',igame,NamesXG[igame])
        ######################### Event data ###################################                                                                            
        XGNumbers,XGTeam,XGValue,XGHalf,XGMin,XGSec,XGTimes,XGPos1,XGPos2 = MacihneLearning_OptaLoad(TeamA,NamesXG,igame)
        sw = 0; click = 0;    itest = 1;    TeamF = 0;    bLive = 1; cxg = 0
        TFinalmax = len(mTime)-2
        if (len(XGHalf)>0):
            while (itest < len(mTime)-1):
                hf = mTime[itest][1]; ti = mTime[itest][0]; ti0 =  mTime[itest-1][0]
                if (ti >= XGTimes[cxg] and ti0 < XGTimes[cxg]):
                    itmp = itest;                
                    GoalTeam = 1
                    if (XGTeam[cxg]==TeamA):
                        GoalTeam = 0
                    print('Chance number: ',cxg,XGValue[cxg])
                    XPos = XGPos1[cxg]
                    YPos = XGPos2[cxg]
                    igoal = FindTimeOfChance_With_POS_and_XGPlayer(GoalTeam,itest+25*5,mBall,mFcn,mOpp,XGNumbers[cxg],TFinalmax,XPos,YPos)
                    #igoal = FindTimeOfChance_WithXGPlayer(GoalTeam,itest+25*5,mBall,mFcn,mOpp,XGNumbers[cxg])
                    istart = np.max([itest-25*15,1])
                    RunAnimation_WithZone(istart,itest+25*5,mFcn,mOpp,mBall,igoal,int(XGNumbers[cxg]),GoalTeam,XPos,YPos)

                    cxg+=1
                    if (cxg == len(XGTeam)):
                        break
                itest+=1
