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
import seaborn as sns
from Functions_Plot import *

#############################################################

bLive = 1;


def FindTimeOfChance_With_POS_and_XGPlayer(GoalTeam,itest,mBall,mFcn,mOpp,Number,Tmax,XPos,YPos):
    acCon = 0; acc = 0; acc2 = 1; acc3 = 1; Tspan = 20; itmp = itest; fac = -1
    xr = 105*float(XPos)/100.0-52.5
    yr = 68*float(YPos)/100.-34
    if GoalTeam == 1:
        xr = -xr; yr = -yr

    while acc == 0:
        if (GoalTeam == 0):
            M = np.array(mFcn[itmp])
            fac = 1
        else:
            M = np.array(mOpp[itmp])
        Num = M[:,4]
        idx = np.argwhere(Num == Number) ### This is the player with the recorded chance                                                                                                                              
        MBall = np.array(mBall[itmp])
        pDist = np.sqrt( (MBall[0]-M[idx,0])**2 + (MBall[1]-M[idx,1])**2)
        r0x = fac*52.5-mBall[itmp][0]; r0y = mBall[itmp][1]
        r0 = [r0x,r0y];  rhat = r0/np.sqrt(r0[0]**2 + r0[1]**2)
        v0 = [fac*25*(mBall[itmp+1][0]-mBall[itmp][0]),25*(mBall[itmp+1][1]-mBall[itmp][1])]
        sv0 = np.sqrt(v0[0]**2 + v0[1]**2)
        vhat = [0,0]
        if (sv0 >0):
            vhat = v0/sv0
        vdot = np.dot(vhat,rhat)

        if (pDist < 1.5 and np.sqrt((mBall[itmp][0]-xr)**2 + (mBall[itmp][1]-yr)**2) < 5 and MBall[4]==GoalTeam and MBall[5]==1):
            acc = 1
            igoal = itmp

        if (itest-itmp >= 25*Tspan or itmp == 1):
            acc = 1;
            acc2 = 0
            igoal = itmp
        else:
            itmp-=1

    print(acc2)
    return igoal


def FindTimeOfChance_WithXGPlayer(GoalTeam,itest,mBall,mFcn,mOpp,Number,XPos,YPos):
    acCon = 0; acc = 0;
    acc2 = 1
    Tspan = 20
    itmp = itest;
    fac = -1
    while acc == 0:
        if (GoalTeam == 0):
            M = np.array(mFcn[itmp])
            fac = 1
        else:
            M = np.array(mOpp[itmp])

        Num = M[:,4]
        idx = np.argwhere(Num == Number) ### This is the player with the recorded chance

        MBall = np.array(mBall[itmp])
        pDist = np.sqrt( (MBall[0]-M[idx,0])**2 + (MBall[1]-M[idx,1])**2)
        
        r0x = fac*52.5-mBall[itmp][0]; r0y = mBall[itmp][1]
        r0 = [r0x,r0y];  rhat = r0/np.sqrt(r0[0]**2 + r0[1]**2)
        v0 = [fac*25*(mBall[itmp+1][0]-mBall[itmp][0]),25*(mBall[itmp+1][1]-mBall[itmp][1])]
        sv0 = np.sqrt(v0[0]**2 + v0[1]**2)
        vhat = v0/sv0
        vdot = np.dot(vhat,rhat)

        if (pDist < 1.5 and np.abs(vdot)>0.9 and sv0 > 15 and np.sqrt(r0[0]**2 + r0[1]**2) < 25):            
            M3 = mOpp[itmp]
            if (GoalTeam==1):
                M3 = mFcn[itmp]
            
            if (np.min(np.sqrt( (MBall[0]-M3[:,0])**2 + (MBall[1]-M3[:,1])**2))>pDist):
                acc = 1
                igoal = itmp

        if (itest-itmp >= 25*Tspan or itmp == 1):
            acc = 1;
            acc2 = 0
            igoal = itmp
        else:
            itmp -= 1
    ########### Run again with relaxed criteria
    print('This situation is found ',acc2)
    itmp = itest; fac = -1
    while acc2 == 0:
        if (GoalTeam == 0):
            M = np.array(mFcn[itmp])
            fac = 1
        else:
            M = np.array(mOpp[itmp])
        Num = M[:,4]
        idx = np.argwhere(Num == Number) ### This is the player with the recorded chance                                                                                                                              
        MBall = np.array(mBall[itmp])
        r0x = fac*52.5-mBall[itmp][0]; r0y = mBall[itmp][1]
        r0 = [r0x,r0y];
        
        rhat = r0/np.sqrt(r0[0]**2 + r0[1]**2)
        pDist = np.sqrt( (MBall[0]-M[idx,0])**2 + (MBall[1]-M[idx,1])**2)

        if (pDist < 3 and np.sqrt(r0[0]**2 + r0[1]**2)<25):
            acc2 = 1
            igoal = itmp
        if (itest-itmp >= 25*Tspan or itmp == 1):
            igoal = itest
            acc2 = 1
            acc3 = 0
        else:
            itmp -= 1

    itmp = itest; fac = -1
    while acc3 == 0:
        if (GoalTeam == 0):
            M = np.array(mFcn[itmp])
            fac = 1
        else:
            M = np.array(mOpp[itmp])
        Num = M[:,4]
        idx = np.argwhere(Num == Number) ### This is the player with the recorded chance                                                                                   
        MBall = np.array(mBall[itmp])
        r0x = fac*52.5-mBall[itmp][0]; r0y = mBall[itmp][1]
        r0 = [r0x,r0y];

        rhat = r0/np.sqrt(r0[0]**2 + r0[1]**2)
        pDist = np.sqrt( (MBall[0]-M[idx,0])**2 + (MBall[1]-M[idx,1])**2)
        if (pDist < 1.5 and np.sqrt((mBall[itmp][0]-xr)**2 + (mBall[itmp][1]-yr)**2) < 5 and MBall[4]==GoalTeam and MBall[5]==1):
            acc2 = 1
            igoal = itmp
    return igoal



def Position_XG(igoal,mFcn,mOpp,mBall,GoalTeam):
    if (GoalTeam == 0):
        GX = 52.5;Mattack = mFcn;Mdefend = mOpp
    else:
        GX = -52.5;Mattack = mOpp;Mdefend = mFcn

    DX = np.abs(GX-mBall[igoal][0])

    DY = mBall[igoal][1]
    Z = 10
    if (DX < 16.5 and np.abs(DY) <= 3.75):
        Z = 0
    elif (DX < 20 and np.abs(DY) > 3.75 and np.abs(DY) <= 18.75):
        gr = 4./3
        if (DX > np.abs(DY)*gr):
            Z = 1
        else:
            Z = 2
    else:
        Z = 2
    return DX,DY,Z

def Established_TimeInPossession(igoal,istart,mFcn,mOpp,mBall,GoalTeam,RevTime):
    if (GoalTeam == 0):
        GX = 52.5;Mattack = mFcn;Mdefend = mOpp                
    else:
        GX = -52.5;Mattack = mOpp;Mdefend = mFcn                

    cOut = 0; cPos = 0
    Pos = mBall[istart][4]
    OutT = []; OutS = []; PosT = []; PosS = []; PosD = [];
    maxDef = 0
    maxOff = 0
    for i in range(istart,igoal):
        TB = mBall[i][4]             #### Defines who "has" the ball
        mDef = np.array(Mdefend[i])  
        mOff = np.array(Mattack[i])

        if (TB ==  Pos and mBall[i][5] == 1): 
            cPos += 1 ##### Counts for the team in possession
            if (np.sum(np.sqrt( (mDef[:,0]-GX)**2 + mDef[:,1]**2) < np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2)) > maxDef): ######## Counts the number of defenders closer to their goal than the ball
                maxDef = np.sum(np.sqrt( (mDef[:,0]-GX)**2 + mDef[:,1]**2) < np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2))
                
            if (TB == GoalTeam and np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2) < maxOff):   ##### Maximal offensive position for the goal team when in position 
                maxOff = np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2)
            elif (TB != GoalTeam and np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2) > maxOff): ##### Maximal offensive position for the receiving team in position
                maxOff = np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2)
        else: ######### <- transition
            Pos = TB
            if (cPos>0):
                PosS.append(cPos/25.0) #### <- append time of the position streak
                if (mBall[i-1][4]==GoalTeam):
                    PosT.append(1) #### <- append each ball loss for goal team
                else:
                    PosT.append(-1) #### <- append each ball loss for receiving team
                PosD.append([maxDef,maxOff])
            cPos = 0
            maxDef = 0
            if (TB == GoalTeam):
                maxOff = 10000
            else:
                maxOff = 0
        if (mBall[i][5] == 0):
            cOut += 1 ##### <- register out of play 

        if (mBall[i-1][5] ==  0 and mBall[i][5] == 1): #### If ball is initiated
            OutS.append(cOut/25.0)
            M2 = np.array(Mattack[i]);M3 = np.array(Mdefend[i])                        
            rdefend = np.sqrt( (M3[:,0]-mBall[i][0])**2 + (M3[:,1]-mBall[i][1])**2) #### Distance from players to ball
            rattack = np.sqrt( (M2[:,0]-mBall[i][0])**2 + (M2[:,1]-mBall[i][1])**2)
            if (np.min(rattack) < np.min(rdefend)):
                OutT.append(1)
            else:
                OutT.append(-1)
            cOut = 0
    if (cOut > 0):
        OutS.append(cOut/25.0)
    if (mBall[igoal][4]==GoalTeam):
        PosT.append(1)
    else:
        PosT.append(-1)

    PosS.append(cPos/25.0)
    PosD.append([maxDef,maxOff])
    PosD = np.array(PosD)
#    print(PosS,PosD,OutS,OutT)
    return PosS,PosT,PosD,OutS,OutT
            


def GenXY(istart,iend,mFcn,mOpp,mBall):
    XFcn = []; XOpp = []; YFcn = []; YOpp = []; XBall = []
    for itmp in range(istart,iend):
        M2 = np.array(mFcn[itmp])                            # M2 is data of FCN players                                                                                                                             
        M3 = np.array(mOpp[itmp])
        MBall = np.array(mBall[itmp])
        XFcn.append(M2[:,0])
        YFcn.append(M2[:,1])
        XOpp.append(M3[:,0])
        YOpp.append(M3[:,1])
        XBall.append(MBall)
    XFcn = np.array(XFcn)
    YFcn = np.array(YFcn)
    XOpp = np.array(XOpp)
    YOpp = np.array(YOpp)
    XBall = np.array(XBall)
    return XFcn,YFcn, XOpp, YOpp,XBall


def Standard_Get_DistAndDensity(idead,mFcn,mOpp,mBall,GoalTeam):
    GX = -52.5
    if (GoalTeam == 0):
        GX = 52.5
    M2 = np.array(mFcn[idead]); M3 = np.array(mOpp[idead])
    RFcn = np.sqrt( (mBall[idead][0]-M2[:,0])**2 + (mBall[idead][1]-M2[:,1])**2)
    Ropp = np.sqrt( (mBall[idead][0]-M3[:,0])**2 + (mBall[idead][1]-M3[:,1])**2)
    ############### Ball position to goal                                                                                                            
    Bxv = np.abs(mBall[idead][0]-GX)
    ############## Player position to goal
    rgFcn = np.sqrt( (GX-M2[:,0])**2 + (M2[:,1])**2)
    rgOpp = np.sqrt( (GX-M3[:,0])**2 + (M3[:,1])**2)
#    print(np.min(RFcn),np.min(Ropp),mBall[idead][4],GoalTeam)
    TeamAttack = 0
    if (np.min(RFcn) < 2 and np.min(Ropp) < 2):
        if (mBall[idead][4]==GoalTeam):
            TeamAttack = 1
 #           print('here')
    else:
        if (GoalTeam == 0 and np.min(RFcn) < np.min(Ropp)): ########## Was if FCN that had the freekick?
            TeamAttack = 1 
        if (GoalTeam == 1 and np.min(RFcn) > np.min(Ropp)):
            TeamAttack = 1 
    Number_close20 = np.sum(rgFcn<20)+np.sum(rgOpp<20)

    return Bxv,TeamAttack,Number_close20


def HasTheBallbeenOutOfPlay(igoal,iend,mBall):
    itmp = igoal; acCon = 0; BallDead = []; BallOut = 0; idead = -1; duration = -1
    balleveralive = 0
    if (mBall[itmp][5] == bLive):
        balleveralive = 1

    while acCon == 0:
        if (balleveralive == 0 and mBall[itmp][5] == 1):
            balleveralive = 1
        if (igoal-itmp > iend or itmp == 1): ### Rewind 25 seconds
            acCon = 1
        if(mBall[itmp-1][5] != bLive and mBall[itmp][5] == bLive):
            BallDead.append(itmp); idead = itmp

        if (len(BallDead)==1):
            if(mBall[itmp-1][5] == bLive and mBall[itmp][5] != bLive or acCon == 1):
                 duration = (idead-itmp)*0.04
            
        itmp -= 1
    if (balleveralive == 0):
        print('The ball has never been registered as alive!')
    Time = 0.04*(igoal-idead)
    
    if (len(BallDead)>0):
        BallOut = 1
    return BallOut,idead,duration,Time

def WasItPenalty(igoal,idead,mFcn,mOpp,mBall,GoalTeam,xgv):
    GX = -52.5
    if (GoalTeam == 0):
        GX = 52.5
    M2 = np.array(mFcn[idead]); M3 = np.array(mOpp[idead])
    Lmax = len(mBall[:][0])
    ################ Distances of all players to the ball ###############                                                                                                                                             
    RFcn = np.sqrt( (mBall[idead][0]-M2[:,0])**2 + (mBall[idead][1]-M2[:,1])**2)
    Ropp = np.sqrt( (mBall[idead][0]-M3[:,0])**2 + (mBall[idead][1]-M3[:,1])**2)
    ############### Ball position to goal                                                                                                                                                                 
    Bxv = np.abs(mBall[idead][0]-GX)
    ############## Player position to goal                                                                                                                                                                             
    rgFcn = np.sqrt( (GX-M2[:,0])**2 + (M2[:,1])**2)
    rgOpp = np.sqrt( (GX-M3[:,0])**2 + (M3[:,1])**2)
    Penalty = 0
    if (Bxv < 14 and np.abs(mBall[idead][1]) < 5 and np.sum(rgFcn > 15)>=9 and np.sum(rgOpp > 15)>=9):
        Penalty = 1

    ac = 0; itmp = igoal-1

    while ac == 0:
        if (mBall[itmp][5]==0 or itmp == 0):
            istart = itmp; ac = 1
        else:
            itmp -= 1

    ac = 0
    while ac == 0:
        if (mBall[itmp][5]==1 or itmp == 0):
            istart = itmp; ac = 1        
        else:
            itmp -= 1
    return Penalty,istart

def FindTimeOfChance(GoalTeam,itest,mBall):
    acCon = 0; acc = 0;
    itmp = itest;
    
    while acc == 0:

        if(mBall[itmp][5]==1 and mBall[itmp][4]==GoalTeam):
            acc = 1
            igoal = itmp
        if (itmp == 0):
            acc = 1;
        else:
            itmp -= 1
    itmp = igoal
    return igoal

def PassFinder(istart,iend,mFcn,mOpp,mBall):
    DetPas = []
    Cpas = []
    XFcn = []; XOpp = []; YFcn = []; YOpp = []; XBall = []
    for itmp in range(istart,iend):
        M2 = np.array(mFcn[itmp])                            # M2 is data of FCN players                                                                                                                  
        M3 = np.array(mOpp[itmp])
        MBall = np.array(mBall[itmp])
        XFcn.append(M2[:,0]); YFcn.append(M2[:,1])
        XOpp.append(M3[:,0]); YOpp.append(M3[:,1])
        XBall.append(MBall)
    XFcn = np.array(XFcn); YFcn = np.array(YFcn)
    XOpp = np.array(XOpp); YOpp = np.array(YOpp)
    XBall = np.array(XBall)
    
    DetPas = []; Cpas = []

    for ipass in range(1,len(XBall[:])-1):
        ###### Calculate ball acceleration (a) and distance to ball for players (dF)
        ax = XBall[ipass-1][0]+XBall[ipass+1][0]-2*XBall[ipass][0]; ay = XBall[ipass-1][1]+XBall[ipass+1][1]-2*XBall[ipass][1]; az = XBall[ipass-1][2]+XBall[ipass+1][2]-2*XBall[ipass][2]
        a = np.sqrt(ax**2+ay**2+az**2)
        dFx = XFcn[ipass]-XBall[ipass][0]; dFy = YFcn[ipass]-XBall[ipass][1]
        dF = np.sqrt(dFx**2+dFy**2)
        dOx = XOpp[ipass]-XBall[ipass][0]; dOy = YOpp[ipass]-XBall[ipass][1]
        dO = np.sqrt(dOx**2+dOy**2)
        DetPas.append([a,np.argmin(dF),np.min(dF),np.argmin(dO),np.min(dO),XBall[ipass,4],XBall[ipass,5]])
    DetPas = np.array(DetPas)
    

    sw = 0; NewP = []
    min_dist = 2
    min_v = 0.1
    for ipass in range(1,len(DetPas[:,0])-6):
        if (sw==0): ###### Looking for initiation of pass                                                                                                                                             
            xside = 1
            if (DetPas[ipass,2]<DetPas[ipass,4]):
                xside = 0
                
            dtmp = []; dztmp = []
            for ipasstmp in range(5):
                if xside == 0:
                    x = XFcn[ipass+ipasstmp]; y = YFcn[ipass+ipasstmp]; idx = int(DetPas[ipasstmp+ipass,1])
                else:
                    x = XOpp[ipasstmp+ipass]; y = YOpp[ipasstmp+ipass]; idx = int(DetPas[ipasstmp+ipass,3])
                x = x[idx]; y = y[idx] ### This is position of the player closest to the ball
                dxb = XBall[ipasstmp+ipass][0]-x; dyb = XBall[ipasstmp+ipass][1]-y; dzb = XBall[ipasstmp+ipass][2]

                dtmp.append(np.sqrt(dxb**2+dyb**2+dzb**2))
                dztmp.append(dzb)
                
            dtmp = np.array(dtmp); dztmp = np.array(dztmp)
            vtmp = dtmp[1:]-dtmp[:-1]
            atmp = vtmp[1:]-vtmp[:-1]
            if (np.min(dtmp) < min_dist and np.sum(vtmp > min_v)>1): #### This defines the initiation of a pass
                sw = 1; iniP = ipass; iniSide = xside
                iniX = XBall[ipass,0]; iniY = XBall[ipass,1];	
                if (xside == 0):
                    PasPl = DetPas[ipass,1]
                else:
                    PasPl = DetPas[ipass,3]

                if (xside == 0):
                    disball = np.abs(XBall[ipass,0]+52.5)
                    defplayer = np.abs(XFcn[ipass]+52.5)
                else:
                    disball = np.abs(XBall[ipass,0]-52.5)
                    defplayer = np.abs(XOpp[ipass]-52.5)
                nplayer_before = np.sum(disball>defplayer)

        elif (sw == 1):
            if (DetPas[ipass,0]>DetPas[ipass-1,0] and DetPas[ipass,0]>DetPas[ipass+1,0]):
                xside = 1
                if (DetPas[ipass,2]<DetPas[ipass,4]):
                    xside = 0
                
                dtmp = []; dztmp = [];
                for ipasstmp in range(5):
                    if xside == 0:
                        x = XFcn[ipasstmp+ipass]; y = YFcn[ipasstmp+ipass]; idx = int(DetPas[ipasstmp+ipass,1])
                    else:
                        x = XOpp[ipasstmp+ipass]; y = YOpp[ipasstmp+ipass]; idx = int(DetPas[ipasstmp+ipass,3])
                    x = x[idx]; y = y[idx]
                    dxb = XBall[ipasstmp+ipass][0]-x; dyb = XBall[ipasstmp+ipass][1]-y; dzb = XBall[ipasstmp+ipass][2]

                    dtmp.append(np.sqrt(dxb**2+dyb**2+dzb**2))
                    dztmp.append(dzb)
                dztmp = np.array(dztmp)
                dtmp = np.array(dtmp)

                vtmp = dtmp[1:]-dtmp[:-1]
                if (np.sum(vtmp<0) >= len(vtmp)-1 and np.min(dtmp)<2): #### This defines the receiver of a pass
                    ac = 0
                    if not (iniSide == xside and DetPas[ipass,1+xside+xside] == PasPl):
                        finX = XBall[ipass,0]; finY = XBall[ipass,1];
                        if (xside == 0):
                            disball = np.abs(XBall[ipass,0]+52.5)
                            defplayer = np.abs(XFcn[ipass]+52.5)
                        else:
                            disball = np.abs(XBall[ipass,0]-52.5)
                            defplayer = np.abs(XOpp[ipass]-52.5)
                        nplayer_after = np.sum(disball>defplayer)
                        NewP.append([len(NewP),iniP,ipass,iniP-ipass,xside,iniSide,iniX,iniY,finX,finY,nplayer_before,nplayer_after])
                    sw = 0

    NewP = np.array(NewP)
    return NewP



def ShowXGpositions(XGPos):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    ax.set_facecolor('#80a260')

    ax.set_xlim(00,60)
    ax.set_ylim(-40,40)

    LX = 105
    LY = 68;

    l1x = [-LX/2,LX/2]; l1y = [-LY/2,-LY/2]; plt.plot(l1x,l1y,'w')
    l2x = [-LX/2,LX/2]; l2y = [LY/2,LY/2]; plt.plot(l2x,l2y,'w')
    l3x = [-LX/2,-LX/2]; l3y = [-LY/2,LY/2]; plt.plot(l3x,l3y,'w')
    l4x = [LX/2,LX/2]; l4y = [-LY/2,LY/2]; plt.plot(l4x,l4y,'w')
    l5x = [0,0]; l5y = [-LY/2,LY/2]; plt.plot(l5x,l5y,'w')
    ll = np.linspace(0,2*3.141592,100); l6x = 9.15*np.cos(ll); l6y = 9.15*np.sin(ll) ;plt.plot(l6x,l6y,'w')

    l7x = [0-LX/2,16.5-LX/2]; l7y = [-20.1,-20.1]; plt.plot(l7x,l7y,'w')
    l8x = [0-LX/2,16.5-LX/2]; l8y = [20.1,20.1]; plt.plot(l8x,l8y,'w'); l9x = [16.5-LX/2,16.5-LX/2]; l9y = [-20.1,20.1]; plt.plot(l9x,l9y,'w')
    l7bx = [-16.5+LX/2,0+LX/2]; l7by = [-20.1,-20.1]; plt.plot(l7bx,l7by,'w')
    l8bx = [-16.5+LX/2,0+LX/2]; l8by = [20.1,20.1]; plt.plot(l8bx,l8by,'w')
    l9bx = [-16.5+LX/2,-16.5+LX/2]; l9by = [-20.1,20.1]; plt.plot(l9bx,l9by,'w')

    s1x = [0-LX/2,5.5-LX/2]; s1y = [-9.1,-9.1]; plt.plot(s1x,s1y,'w')
    s2x = [0-LX/2,5.5-LX/2]; s2y = [9.1,9.1]; plt.plot(s2x,s2y,'w')
    s3x = [5.5-LX/2,5.5-LX/2]; s3y = [-9.1,9.1]; plt.plot(s3x,s3y,'w')

    s4x = [-5.5+LX/2,0+LX/2]; s4y = [-9.1,-9.1]; plt.plot(s4x,s4y,'w');
    s5x = [-5.5+LX/2,0+LX/2]; s5y = [9.1,9.1]; plt.plot(s5x,s5y,'w')
    s6x = [-5.5+LX/2,-5.5+LX/2]; s6y = [-9.1,9.1]; plt.plot(s6x,s6y,'w')
    
    g1x = [2+LX/2,2+LX/2]; g1y = [-3.7,3.7]; plt.plot(g1x,g1y,'w',lw=1)
    g2x = [LX/2,2+LX/2]; g2y = [-3.7,-3.7]; plt.plot(g2x,g2y,'w',lw=1)
    g3x = [LX/2,2+LX/2]; g3y = [3.7,3.7]; plt.plot(g3x,g3y,'w',lw=1)

    za1x = [-16+LX/2,0+LX/2]; za1y = [-4,-4]; plt.plot(za1x,za1y,'--r',lw=1);
    za2x = [-16+LX/2,0+LX/2]; za2y = [4,4]; plt.plot(za2x,za2y,'--r',lw=1);
    za3x = [-16+LX/2,-16+LX/2]; za3y = [-4,4]; plt.plot(za3x,za3y,'--r',lw=1);

    zb1x = [0+LX/2,-20+LX/2]; zb1y = [-4,-16.5]; plt.plot(zb1x,zb1y,'--b',lw=1);
    zb2x = [0+LX/2,-20+LX/2]; zb2y = [4,16.5]; plt.plot(zb2x,zb2y,'--b',lw=1);
    zb3x = [-20+LX/2,-20+LX/2]; zb3y = [-16.5,16.5]; plt.plot(zb3x,zb3y,'--b',lw=1);
    t = np.linspace(0,2*np.pi,10)
    for i in range(len(XGPos)):
        plt.plot(XGPos[i][0]+XGPos[i][2]*np.cos(t), XGPos[i][1]+XGPos[i][2]*np.sin(t),'w')        
    
    plt.show()
    return LX




def RunAnimation(istart,iend,mFcn,mOpp,mBall):
    XFcn = []; XOpp = []; YFcn = []; YOpp = []; XBall = []

    for itmp in range(istart,iend):
        M2 = np.array(mFcn[itmp])                            # M2 is data of FCN players                                                                                                                  
        M3 = np.array(mOpp[itmp])
        MBall = np.array(mBall[itmp])
        XFcn.append(M2[:,0])
        YFcn.append(M2[:,1])
        XOpp.append(M3[:,0])
        YOpp.append(M3[:,1])
        XBall.append(MBall)
    XFcn = np.array(XFcn)
    YFcn = np.array(YFcn)
    XOpp = np.array(XOpp)
    YOpp = np.array(YOpp)
    XBall = np.array(XBall)
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    ax.set_facecolor('#80a260')
    
    ax.set_xlim(-60,60)
    ax.set_ylim(-40,40)
    
    x1, y1 = [0,0]
    x2, y2 = [0,0]
    x3, y3 = [0,0]

    mat10, = ax.plot(x1, y1, 'ko',markersize = 10)
    mat1, = ax.plot(x1, y1, 'ro',markersize = 8)

    mat20, = ax.plot(x2, y2, 'ko',markersize = 10)
    mat2, = ax.plot(x2, y2, 'bo',markersize = 8)
    mat30, = ax.plot(x3, y3, 'ko',markersize = 10)
    mat3, = ax.plot(x3, y3, 'wo',markersize = 5)
    flag_out, = ax.plot([100, 100], 'ks',markersize = 25)

    def animate(i):        
        xFcn = XFcn[i,:]; x1 = xFcn;
        yFcn = YFcn[i,:]; y1 = yFcn;
        xOpp = XOpp[i,:]; x2 = xOpp;
        yOpp = YOpp[i,:]; y2 = yOpp;
        xBall = XBall[i,0]; x3 = xBall
        yBall = XBall[i,1]; y3 = yBall
        
        mat1.set_data(x1, y1)
        mat10.set_data(x1, y1)
        mat20.set_data(x2, y2)
        mat30.set_data(x3, y3)
        mat2.set_data(x2, y2)
        mat3.set_data(x3, y3)

        if (XBall[i,5]<1):
            flag_out.set_data([0,40])
        else:
            flag_out.set_data([100,100])
        return mat10,mat1,mat20,mat2,mat30,mat3,flag_out

    ani = animation.FuncAnimation(fig, animate, interval=3,frames=len(XFcn[:,0]), blit=True,repeat=False)
    LX = 105
    LY = 68;
    l1x = [-LX/2,LX/2]; l1y = [-LY/2,-LY/2]; plt.plot(l1x,l1y,'w')
    l2x = [-LX/2,LX/2]; l2y = [LY/2,LY/2]; plt.plot(l2x,l2y,'w')
    l3x = [-LX/2,-LX/2]; l3y = [-LY/2,LY/2]; plt.plot(l3x,l3y,'w')
    l4x = [LX/2,LX/2]; l4y = [-LY/2,LY/2]; plt.plot(l4x,l4y,'w')
    l5x = [0,0]; l5y = [-LY/2,LY/2]; plt.plot(l5x,l5y,'w')
    ll = np.linspace(0,2*3.141592,100); l6x = 9.15*np.cos(ll); l6y = 9.15*np.sin(ll) ;plt.plot(l6x,l6y,'w')

    l7x = [0-LX/2,16.5-LX/2]; l7y = [-20.1,-20.1]; plt.plot(l7x,l7y,'w')
    l8x = [0-LX/2,16.5-LX/2]; l8y = [20.1,20.1]; plt.plot(l8x,l8y,'w'); l9x = [16.5-LX/2,16.5-LX/2]; l9y = [-20.1,20.1]; plt.plot(l9x,l9y,'w')
    l7bx = [-16.5+LX/2,0+LX/2]; l7by = [-20.1,-20.1]; plt.plot(l7bx,l7by,'w')
    l8bx = [-16.5+LX/2,0+LX/2]; l8by = [20.1,20.1]; plt.plot(l8bx,l8by,'w')
    l9bx = [-16.5+LX/2,-16.5+LX/2]; l9by = [-20.1,20.1]; plt.plot(l9bx,l9by,'w')
    
    s1x = [0-LX/2,5.5-LX/2]; s1y = [-9.1,-9.1]; plt.plot(s1x,s1y,'w')
    s2x = [0-LX/2,5.5-LX/2]; s2y = [9.1,9.1]; plt.plot(s2x,s2y,'w')
    s3x = [5.5-LX/2,5.5-LX/2]; s3y = [-9.1,9.1]; plt.plot(s3x,s3y,'w')
    
    s4x = [-5.5+LX/2,0+LX/2]; s4y = [-9.1,-9.1]; plt.plot(s4x,s4y,'w');
    s5x = [-5.5+LX/2,0+LX/2]; s5y = [9.1,9.1]; plt.plot(s5x,s5y,'w')
    s6x = [-5.5+LX/2,-5.5+LX/2]; s6y = [-9.1,9.1]; plt.plot(s6x,s6y,'w')

    g1x = [2+LX/2,2+LX/2]; g1y = [-3.7,3.7]; plt.plot(g1x,g1y,'w',lw=3)
    g2x = [LX/2,2+LX/2]; g2y = [-3.7,-3.7]; plt.plot(g2x,g2y,'w',lw=3)
    g3x = [LX/2,2+LX/2]; g3y = [3.7,3.7]; plt.plot(g3x,g3y,'w',lw=3)

    g4x = [-(2+LX/2),-(2+LX/2)]; g4y = [-3.7,3.7]; plt.plot(g1x,g1y,'w',lw=3)
    g5x = [-LX/2,-(2+LX/2)]; g5y = [-3.7,-3.7]; plt.plot(g2x,g2y,'w',lw=3)
    g6x = [-LX/2,-(2+LX/2)]; g6y = [3.7,3.7]; plt.plot(g3x,g3y,'w',lw=3)
    plt.show()
    return LX



def RunAnimation_WithZone(istart,iend,mFcn,mOpp,mBall,igoal,XGnum,GoalTeam,XPos,YPos):
    XFcn = []; XOpp = []; YFcn = []; YOpp = []; XBall = []
    PlayerA = []
    for itmp in range(istart,iend):
        M2 = np.array(mFcn[itmp])                            # M2 is data of FCN players
        M3 = np.array(mOpp[itmp])
        MBall = np.array(mBall[itmp])
        XFcn.append(M2[:,0])
        YFcn.append(M2[:,1])
        XOpp.append(M3[:,0])
        YOpp.append(M3[:,1])
        XBall.append(MBall)
        if (GoalTeam == 0):
            idx = np.argwhere(M2[:,4]==XGnum)
            idx0 = idx[0]; idx00 = idx0[0]
            PlayerA.append([M2[idx00,0],M2[idx00,1]])
        else:
            idx = np.argwhere(M3[:,4]==XGnum)
            idx0 = idx[0]; idx00 = idx0[0]
            PlayerA.append([M3[idx00,0],M3[idx00,1]])
    XFcn = np.array(XFcn)
    YFcn = np.array(YFcn)
    XOpp = np.array(XOpp)
    YOpp = np.array(YOpp)
    XBall = np.array(XBall)
    PlayerA = np.array(PlayerA)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    ax.set_facecolor('#80a260')

    ax.set_xlim(-60,60)
    ax.set_ylim(-40,40)

    x1, y1 = [0,0]
    x2, y2 = [0,0]
    x3, y3 = [0,0]
    x4, y4 = [0,0]

    mat10, = ax.plot(x1, y1, 'ko',markersize = 10)
    mat1, = ax.plot(x1, y1, 'ro',markersize = 8)
    mat20, = ax.plot(x2, y2, 'ko',markersize = 10)
    mat2, = ax.plot(x2, y2, 'bo',markersize = 8)
    mat30, = ax.plot(x3, y3, 'ko',markersize = 10)
    mat3, = ax.plot(x3, y3, 'wo',markersize = 5)
    mat40, = ax.plot(x4, y4, 'wx',markersize = 10)
    mat4, = ax.plot(x4, y4, 'om',markersize = 2)

    flag_out, = ax.plot([100, 100], 'ks',markersize = 25)
    
    #flag_fin, = ax.plot([mBall[iend][0], mBall[iend][1]], 'rs',markersize = 10)

    def animate(i):

        xFcn = XFcn[i,:]; x1 = xFcn;
        yFcn = YFcn[i,:]; y1 = yFcn;
        xOpp = XOpp[i,:]; x2 = xOpp;
        yOpp = YOpp[i,:]; y2 = yOpp;
        xBall = XBall[i,0]; x3 = xBall
        yBall = XBall[i,1]; y3 = yBall
        x4 = PlayerA[i,0]
        y4 = PlayerA[i,1]

        mat1.set_data(x1, y1)
        mat10.set_data(x1, y1)
        mat20.set_data(x2, y2)
        mat30.set_data(x3, y3)
        mat40.set_data(x4, y4)
        mat2.set_data(x2, y2)
        mat3.set_data(x3, y3)
        mat4.set_data(x4, y4)

        if (XBall[i,5]<1):
            flag_out.set_data([0,40])
        else:
            flag_out.set_data([100,100])
        return mat10,mat1,mat20,mat2,mat30,mat3,mat40,mat4,flag_out

    ani = animation.FuncAnimation(fig, animate, interval=3,frames=len(XFcn[:,0]), blit=True,repeat=False)
    LX = 105
    LY = 68;
    l1x = [-LX/2,LX/2]; l1y = [-LY/2,-LY/2]; plt.plot(l1x,l1y,'w')
    l2x = [-LX/2,LX/2]; l2y = [LY/2,LY/2]; plt.plot(l2x,l2y,'w')
    l3x = [-LX/2,-LX/2]; l3y = [-LY/2,LY/2]; plt.plot(l3x,l3y,'w')
    l4x = [LX/2,LX/2]; l4y = [-LY/2,LY/2]; plt.plot(l4x,l4y,'w')
    l5x = [0,0]; l5y = [-LY/2,LY/2]; plt.plot(l5x,l5y,'w')
    ll = np.linspace(0,2*3.141592,100); l6x = 9.15*np.cos(ll); l6y = 9.15*np.sin(ll) ;plt.plot(l6x,l6y,'w')

    l7x = [0-LX/2,16.5-LX/2]; l7y = [-20.1,-20.1]; plt.plot(l7x,l7y,'w')
    l8x = [0-LX/2,16.5-LX/2]; l8y = [20.1,20.1]; plt.plot(l8x,l8y,'w'); l9x = [16.5-LX/2,16.5-LX/2]; l9y = [-20.1,20.1]; plt.plot(l9x,l9y,'w')
    l7bx = [-16.5+LX/2,0+LX/2]; l7by = [-20.1,-20.1]; plt.plot(l7bx,l7by,'w')
    l8bx = [-16.5+LX/2,0+LX/2]; l8by = [20.1,20.1]; plt.plot(l8bx,l8by,'w')
    l9bx = [-16.5+LX/2,-16.5+LX/2]; l9by = [-20.1,20.1]; plt.plot(l9bx,l9by,'w')

    s1x = [0-LX/2,5.5-LX/2]; s1y = [-9.1,-9.1]; plt.plot(s1x,s1y,'w')
    s2x = [0-LX/2,5.5-LX/2]; s2y = [9.1,9.1]; plt.plot(s2x,s2y,'w')
    s3x = [5.5-LX/2,5.5-LX/2]; s3y = [-9.1,9.1]; plt.plot(s3x,s3y,'w')

    s4x = [-5.5+LX/2,0+LX/2]; s4y = [-9.1,-9.1]; plt.plot(s4x,s4y,'w');
    s5x = [-5.5+LX/2,0+LX/2]; s5y = [9.1,9.1]; plt.plot(s5x,s5y,'w')
    s6x = [-5.5+LX/2,-5.5+LX/2]; s6y = [-9.1,9.1]; plt.plot(s6x,s6y,'w')

    za1x = [-16+LX/2,0+LX/2]; za1y = [-4,-4]; plt.plot(za1x,za1y,'--r',lw=3);
    za2x = [-16+LX/2,0+LX/2]; za2y = [4,4]; plt.plot(za2x,za2y,'--r',lw=3);
    za3x = [-16+LX/2,-16+LX/2]; za3y = [-4,4]; plt.plot(za3x,za3y,'--r',lw=3);

    zb1x = [0+LX/2,-20+LX/2]; zb1y = [-4,-16.5]; plt.plot(zb1x,zb1y,'--b',lw=3);
    zb2x = [0+LX/2,-20+LX/2]; zb2y = [4,16.5]; plt.plot(zb2x,zb2y,'--b',lw=3);
    zb3x = [-20+LX/2,-20+LX/2]; zb3y = [-16.5,16.5]; plt.plot(zb3x,zb3y,'--b',lw=3);


    g1x = [2+LX/2,2+LX/2]; g1y = [-3.7,3.7]; plt.plot(g1x,g1y,'w',lw=3)
    g2x = [LX/2,2+LX/2]; g2y = [-3.7,-3.7]; plt.plot(g2x,g2y,'w',lw=3)
    g3x = [LX/2,2+LX/2]; g3y = [3.7,3.7]; plt.plot(g3x,g3y,'w',lw=3)

    g4x = [-(2+LX/2),-(2+LX/2)]; g4y = [-3.7,3.7]; plt.plot(g4x,g4y,'w',lw=3)
    g5x = [-LX/2,-(2+LX/2)]; g5y = [-3.7,-3.7]; plt.plot(g5x,g5y,'w',lw=3)
    g6x = [-LX/2,-(2+LX/2)]; g6y = [3.7,3.7]; plt.plot(g6x,g6y,'w',lw=3)

    t = np.linspace(0,2*np.pi,10)
    plt.plot(mBall[igoal][0]+np.cos(t), mBall[igoal][1]+np.sin(t),'w')
    print(105*float(XPos)/100.0,68*float(YPos)/100.,GoalTeam)
    xr = 105*float(XPos)/100.0-52.5
    yr = 68*float(YPos)/100.-34
    if GoalTeam == 1:
        xr = -xr; yr = -yr
    plt.plot(xr+np.cos(t), yr+np.sin(t),'--r')
    plt.show()
    return LX











def CountBuildupDetailed(igoal,istart,mFcn,mOpp,mBall,GoalTeam,RevTime):
    if (GoalTeam == 0):
        GX = 52.5;Mattack = mFcn;Mdefend = mOpp
    else:
        GX = -52.5;Mattack = mOpp;Mdefend = mFcn

    cOut = 0; cPos = 0
    Pos = mBall[istart][4]
    OutT = []; OutS = []; PassageOfPlay_Team = []; PassageOfPlay_Times = []; PassageOfPlay_NumDefBehind = []; PassageOfPlay_MaxOffBallPos = [];
    TransitionDistances = []
    maxDef = 0;    maxOff = 0
    xcoord_frontman_tmp_defender = 0; xcoord_defenders_tmp_defender =0; xcoord_All_tmp_defender =0;
    xcoord_frontman_tmp_goalteam = 0; xcoord_defenders_tmp_goalteam =0; xcoord_All_tmp_goalteam =0;
    cbalance = 0
    xcoord_frontman_defender = []; xcoord_defenders_defender = []; xcoord_All_defender = [];
    TimeInBalance = []
    xcoord_frontman_goalteam = []; xcoord_defenders_goalteam = []; xcoord_All_goalteam = [];
    for i in range(istart,igoal):
        TB = mBall[i][4]             #### Defines who "has" the ball
        mDef = np.array(Mdefend[i])
        mOff = np.array(Mattack[i])

        ####################### Characterize position of the defending team ################################

        if (TB ==  Pos and mBall[i][5] == 1):

            dis = np.sort(np.abs(GX-mDef[:,0]))
            if (np.sum(np.abs(GX-mBall[i][0]) > dis) > 5): ##### We assume a balance
                dis = np.sort(np.abs(GX-mDef[:,0]))
                cbalance += 1
                xcoord_frontman_tmp_defender += dis[-1]
                xcoord_defenders_tmp_defender += np.mean([dis[1],dis[2]])
                xcoord_All_tmp_defender += np.mean(dis)

            cPos += 1 ##### Counts for the team in possession                                                     
            if (np.sum(np.sqrt( (mDef[:,0]-GX)**2 + mDef[:,1]**2) < np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2)) > maxDef): ######## Counts the number of defenders closer to their goal than the ball
                maxDef = np.sum(np.sqrt( (mDef[:,0]-GX)**2 + mDef[:,1]**2) < np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2))
            if (TB == GoalTeam and np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2) < maxOff):   ##### Maximal offensive position for the goal team when in position
                maxOff = np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2)
            elif (TB != GoalTeam and np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2) > maxOff): ##### Maximal offensive position for the receiving team in position     
                maxOff = np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2)
        else: ######### <- transition                                                                                
            Pos = TB
            if (cPos>0):
                if (cbalance == 0):
                    xcoord_frontman_defender.append(0)
                    xcoord_defenders_defender.append(0)
                    xcoord_All_defender.append(0)

                else:
                    xcoord_frontman_defender.append(xcoord_frontman_tmp_defender/cbalance);
                    xcoord_defenders_defender.append(xcoord_defenders_tmp_defender/cbalance); xcoord_All_defender.append(xcoord_All_tmp_defender/cbalance); 
                TransitionDistances.append(np.sqrt( (mBall[i][0]-GX)**2 + mBall[i][1]**2))
                PassageOfPlay_Times.append(cPos/25.0) #### <- append time of the position streak
                if (mBall[i-1][4]==GoalTeam):
                    PassageOfPlay_Team.append(1) #### <- append each ball loss for goal team       
                else:
                    PassageOfPlay_Team.append(-1) #### <- append each ball loss for receiving team          
                PassageOfPlay_NumDefBehind.append(maxDef)
                PassageOfPlay_MaxOffBallPos.append(maxOff)
                TimeInBalance.append(1.0*cbalance/cPos)
            cPos = 0
            cbalance = 0
            xcoord_frontman_tmp_defender = 0
            xcoord_defenders_tmp_defender = 0
            xcoord_All_tmp_defender = 0

            maxDef = 0
            if (TB == GoalTeam):
                maxOff = 10000
            else:
                maxOff = 0
        if (mBall[i][5] == 0):
            cOut += 1 ##### <- register out of play
        if (mBall[i-1][5] ==  0 and mBall[i][5] == 1): #### If ball is initiated                                                                                             
            OutS.append(cOut/25.0)
            M2 = np.array(Mattack[i]);M3 = np.array(Mdefend[i])
            rdefend = np.sqrt( (M3[:,0]-mBall[i][0])**2 + (M3[:,1]-mBall[i][1])**2) #### Distance from players to ball                                          
            rattack = np.sqrt( (M2[:,0]-mBall[i][0])**2 + (M2[:,1]-mBall[i][1])**2)
            if (np.min(rattack) <2 and  np.min(rdefend) < 2):
                if (mBall[i][4]==GoalTeam):
                    OutT.append(1)
                else:
                    OutT.append(-1)
            else:
                if (np.min(rattack) < np.min(rdefend)):
                    OutT.append(1)
                else:
                    OutT.append(-1)
            cOut = 0
    if (cOut > 0):
        OutS.append(cOut/25.0)
    if (mBall[igoal][4]==GoalTeam):
        PassageOfPlay_Team.append(1)
    else:
        PassageOfPlay_Team.append(-1)
    PassageOfPlay_Times.append(cPos/25.0)
    PassageOfPlay_NumDefBehind.append(maxDef)
    PassageOfPlay_MaxOffBallPos.append(maxOff)

    if (cbalance == 0):
        xcoord_frontman_defender.append(0)
        xcoord_defenders_defender.append(0)
        xcoord_All_defender.append(0)
    else:
        xcoord_frontman_defender.append(xcoord_frontman_tmp_defender/cbalance)
        xcoord_defenders_defender.append(xcoord_defenders_tmp_defender/cbalance); xcoord_All_defender.append(xcoord_All_tmp_defender/cbalance);
    if (cPos>0):
        TimeInBalance.append(1.0*cbalance/cPos)
    PassageOfPlay_NumDefBehind = np.array(PassageOfPlay_NumDefBehind)
    PassageOfPlay_MaxOffBallPos = np.array(PassageOfPlay_MaxOffBallPos)
    FrontManPosition_defender = np.array(xcoord_frontman_defender)
    DefenderPosition_defender = np.array(xcoord_defenders_defender)
    MeanPosition_defender = np.array(xcoord_All_defender)
    TransitionDistances = np.array(TransitionDistances)
    TimeInBalance = np.array(TimeInBalance)

    NewP = [PassageOfPlay_Times,PassageOfPlay_Team,PassageOfPlay_NumDefBehind,PassageOfPlay_MaxOffBallPos,OutS,OutT,FrontManPosition_defender,DefenderPosition_defender,MeanPosition_defender,TransitionDistances,TimeInBalance]
    return NewP

def NumTimesTheBallWasOutOfPlay(OutS,OutT):
    Lmax = -1
    Team = -1
    if (len(OutS)>0 and len(OutT)>0):
        Lmax = np.max(OutS)
        idx = int(np.argwhere(OutS == np.max(OutS)))
        Team = OutT[idx]
    return Lmax,len(OutS),Team

def LongestPassagesOfPlay(PosS,PosT,PosmaxOff):
    cb = [0,0]
    ct = [0,0]
    maxOff = 0
    idOpp = np.argwhere(PosT == -1)
    idPos = np.argwhere(PosT == 1)
    Topp = 0
    if (len(idOpp)>0):
        Topp = np.max(PosS[idOpp])
        maxOff = np.max(PosmaxOff[idOpp])        

    Tpos = np.max(PosS[idPos])
    for it in range(len(PosS)):
        if (PosS[it]>2 and PosT[it]==1):
            cb[0]+=1
            ct[0]+=PosS[it]
        if (PosS[it]>2 and PosT[it]==-1):
            cb[1]+=1
            ct[1]+=PosS[it]
            

    NumPassages_Pos_p2 = cb[0]
    NumPassages_Opp_p2 = cb[1]
    Total_Pos = ct[0]
    Total_Opp = ct[1]
    return Tpos,Topp,NumPassages_Pos_p2,NumPassages_Opp_p2,Total_Pos,Total_Opp,maxOff

                                                                                 
def FinishFinder(mFcn,mOpp,mBall):
    DetPas = []
    Cpas = []
    XFcn = []; XOpp = []; YFcn = []; YOpp = []; XBall = []
    for itmp in range(len(mFcn[:])):
        M2 = np.array(mFcn[itmp])                            # M2 is data of FCN players                                                                                                                               
        M3 = np.array(mOpp[itmp])
        MBall = np.array(mBall[itmp])
        XFcn.append(M2[:,0])
        YFcn.append(M2[:,1])
        XOpp.append(M3[:,0])
        YOpp.append(M3[:,1])
        XBall.append(MBall)
    XFcn = np.array(XFcn); YFcn = np.array(YFcn); XOpp = np.array(XOpp); YOpp = np.array(YOpp); XBall = np.array(XBall)

    Fin = []
    nl = 10
    Cafs = np.zeros(nl)

    for iafs in range(1,len(XBall[:])-1):
        #### Ball acceleration                                                                                                                                                                                         
        ax = XBall[iafs-1][0]+XBall[iafs+1][0]-2*XBall[iafs][0]
        ay = XBall[iafs-1][1]+XBall[iafs+1][1]-2*XBall[iafs][1]
        az = XBall[iafs-1][2]+XBall[iafs+1][2]-2*XBall[iafs][2]
        a = np.sqrt(ax**2+ay**2+az**2)

        #### Player distances                                                                                                                                                                                          
        dFx = XFcn[iafs]-XBall[iafs][0]
        dFy = YFcn[iafs]-XBall[iafs][1]
        dF = np.sqrt(dFx**2+dFy**2)

        dOx = XOpp[iafs]-XBall[iafs][0]
        dOy = YOpp[iafs]-XBall[iafs][1]
        dO = np.sqrt(dOx**2+dOy**2)


        GX = 52.5 - np.abs(XBall[iafs][0])
        GY = np.abs(XBall[iafs][1])
        GD = np.sqrt(GX**2 + GY**2)

        if (a>2 and GD < 30):
            xl = np.linspace(1,nl,nl)
            for i2 in range(nl):
                GX = 52.5 - np.abs(XBall[iafs+i2][0])
                Cafs[i2] = GX
            fx = np.polyfit(xl,Cafs,1)
            if (fx[0]<5):
                Fin.append(iafs)

    Fin = np.array(Fin)

    return Fin

def Pass_keynumbers(Passes):
    Num_Passes = Passes[:,0]; ini_T = Passes[:,1]; fin_T_ = Passes[:,2]; DT = Passes[:,3]
    reciever_pass = Passes[:,5]; initiate_pass = Passes[:,4]
    ini_X = Passes[:,6]; ini_Y = Passes[:,7]; fin_X = Passes[:,8]; fin_Y = Passes[:,9]
    nplayer_before = Passes[:,10]; nplayer_after = Passes[:,11]
    c1 = 1
    passagePass = []
    passagePassTeam = []
    Team = initiate_pass[0]
    for ipas in range(len(initiate_pass)):
        if (initiate_pass[ipas]==reciever_pass[ipas]):
            c1+=1
            Team = initiate_pass[ipas]
        else:
            passagePass.append(c1)
            passagePassTeam.append(Team)
            c1 = 1
    passagePass.append(c1)
    passagePassTeam.append(int(initiate_pass[ipas]))
    return passagePass,passagePassTeam

def GenNetwork(mFcn,mOpp,mBall,itmp):
    M2 = np.array(mFcn[itmp])                            # M2 is data of FCN players                                                                                                                              \
    M3 = np.array(mOpp[itmp])
    MBall = np.array(mBall[itmp])
    Edge = []
    LinkEM = []
    LinkMM = []
    EdgeEl = []
    MidtEl = []
    for i in range(len(M2[:,0])):
        A = [M2[i,0],M2[i,1]]
        plt.plot(A[0],A[1],'*r')
        for j in range(i+1,len(M2[:,0])):
            B = [M2[j,0],M2[j,1]]
            Ss = []
            for k in range(len(M2[:,0])):
                if (k != i and k != j):
                    C = [M2[k,0],M2[k,1]]
                    m00 = A[0]-C[0]
                    m10 = A[1]-C[1]
                    m01 = B[0]-C[0]
                    m11 = B[1]-C[1]
                    s = m00*m11 - m01*m10 
                    Ss.append(np.sign(s))

            if (np.abs(np.sum(Ss))==len(Ss)):
                Edge.append([i,j])
                if (len(EdgeEl)==0):
                    EdgeEl.append(i)
                    EdgeEl.append(j)
                else:
                    if (EdgeEl[-1]<i):
                        EdgeEl.append(i)
                    if (EdgeEl[-1]<j):
                        EdgeEl.append(j)
                plt.plot([A[0],B[0]],[A[1],B[1]],'r')
    Edge = np.array(Edge)
    cedge = 0
    for i in range(len(M2[:,0])):
        if (i==EdgeEl[cedge] and cedge <= len(EdgeEl)):
            cedge+=1
        else:
            MidtEl.append(i)
    for i in MidtEl:
        A = [M2[i,0],M2[i,1]]
        for j in EdgeEl:
            B = [M2[j,0],M2[j,1]]
            
    print(EdgeEl,MidtEl)
    plt.show()
    
    return Edge
