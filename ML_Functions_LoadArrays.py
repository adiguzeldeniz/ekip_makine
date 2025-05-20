import os
import csv
import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import seaborn as sns

#############################################################

def GenDir(Team):
    filename = '../MasterScripts/GameDays.csv'
    DD = []
    DDsl = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            columns = line.split(';')
            if (columns[0] == Team):
                for t in range(1,len(columns)):
                    DD.append(columns[t])
                    c = columns[t].split('-')
                    DDsl.append(c[0]+'/'+c[1]+'/'+c[2])
    return DD,DDsl

def GenFileName(TeamA,dirname):
    Names = []
    nameOptaFolder = '../data/'+dirname
    name = os.listdir(nameOptaFolder)
    ntmp = name[0].split('_ma'); name0 = ntmp[0]

    for itmp in range(len(name)):
        if (name[itmp].find(TeamA)>=0):
            ntmp = name[itmp].split('_ma')
            name0 = ntmp[0]

    Names.append('%s/%s_ma1_metadata.json'%(nameOptaFolder,name0))
    Names.append('%s/%s_ma12_xg_data.json'%(nameOptaFolder,name0))
    Names.append('%s/%s_ma3_events.json'%(nameOptaFolder,name0))
    return Names
    
def GenSecName(TeamA,nameSecSpec,cmf):
    for filename in os.listdir(nameSecSpec):
        cmf += 1
        name = os.listdir(nameSecSpec)
        Player = []
        if (name[cmf-1].find('.json')>0 and name[cmf-1].find(TeamA)>0):
            nameS = name[cmf-1]
    return nameS

    
def SecLoad(TeamA,nameSC,igame):
    M = pd.read_pickle("../RestructuredData/%s/AllData/%s"%(TeamA,nameSC[igame]))
    mTime = M["Times"][:]
    mBall = M["Ball"][:]
    mFcn = M[TeamA][:]
    mOpp = M["Opp"][:]
    return mTime,mBall,mFcn,mOpp

def OptaLoad(TeamA,nameXG,igame):
    XGPlayers = []; XGTimes = []; XGHalf = []; XGMin = []; XGSec = []; XGNumbers = []; XGTeam = []; XGValue = []; XGPos1 = []; XGPos2 = [];

    oname = "../RestructuredData/%s/XGdata/%s"%(TeamA,nameXG[igame])
    xnvalue = np.loadtxt(oname, delimiter=',', usecols=[0])
    xnplay = np.loadtxt(oname, delimiter=',',dtype = 'U', usecols=[4])
    xnnum = np.loadtxt(oname, delimiter=',', usecols=[5])
    xnhalf = np.loadtxt(oname, delimiter=',', usecols=[1])
    xnmin = np.loadtxt(oname, delimiter=',', usecols=[2])
    xnsec = np.loadtxt(oname, delimiter=',', usecols=[3])
    xside = np.loadtxt(oname, delimiter=',', dtype = 'U',usecols=[6])
    Xpos = np.loadtxt(oname, delimiter=',', dtype = 'U',usecols=[7])
    Ypos = np.loadtxt(oname, delimiter=',', dtype = 'U',usecols=[8])
    ia = 0
    while ia < len(xnvalue):
        xvtmp = []
        ci = 1
        acc = 1
        xvtmp.append(ia)
        while acc == 1:
            if (ia+ci < len(xnvalue)-1):
                if (xnhalf[ia]==xnhalf[ia+ci] and (xnmin[ia+ci]*60+xnsec[ia+ci])-(xnmin[ia]*60+xnsec[ia]) < 10):
                    xvtmp.append(ia+ci)
                    ci+=1
                else:
                    acc = 0
            else:
                acc = 0
        if (len(xvtmp)==1):
            nidx = ia
        else:
            nidx = ia+np.argmax(xnvalue[xvtmp])
        ia+=ci
        XGPlayers.append(xnplay[nidx])
        XGNumbers.append(xnnum[nidx])
        XGTeam.append(xside[nidx])
        XGValue.append(xnvalue[nidx])
        XGHalf.append(xnhalf[nidx])
        XGMin.append(xnmin[nidx])
        XGSec.append(xnsec[nidx])
        XGTimes.append(xnmin[nidx]*60+xnsec[nidx])
    XGPlayers = np.array(XGPlayers); XGNumbers = np.array(XGNumbers); XGTeam = np.array(XGTeam); XGValue = np.array(XGValue)
    XGHalf = np.array(XGHalf); XGMin = np.array(XGMin); XGSec = np.array(XGSec); XGTimes = np.array(XGTimes)
    XGPos1 = XPos; XGPos2 = YPos; 
    return XGPlayers,XGNumbers,XGTeam,XGValue,XGHalf,XGMin,XGSec,XGTimes,XGPos1,XGPos2


def Fload(Dir,Team):
    nameSCtmp = os.listdir('../RestructuredData/%s/AllData'%(Team))
    nameXGtmp = os.listdir('../RestructuredData/%s/XGData'%(Team))
    nameXG = []
    nameSC = []
    c = 0
    for c0 in range(len(Dir)):
        for i in range(len(nameXGtmp)):
            if (nameXGtmp[i].find('.txt')>=0 and nameXGtmp[i].find(Dir[c0])>=0):
                nameXG.append(nameXGtmp[i])

    for c0 in range(len(Dir)):
        for i in range(len(nameSCtmp)):
            if (nameSCtmp[i].find('.pkl')>=0 and nameSCtmp[i].find(Dir[c0])>=0):
                nameSC.append(nameSCtmp[i])
    return nameXG,nameSC

def SortGames(DirN,TeamA):
    NamesSCOld = os.listdir('../RestructuredData/%s/AllData'%(TeamA))
    NamesXGOld = os.listdir('../RestructuredData/%s/XGdata'%(TeamA))
    print('LEN',len(NamesSCOld))
    LS = len(NamesSCOld)
    NamesSC = []; NamesXG = []; Num = []
    for i in range(LS):
        A = NamesSCOld[i].split('Day_20')
        A1 = A[1]; A2 = A1.split('Z.pkl')
        A3 = A2[0].split('-'); 
        Num.append(int(str(A3[0]) + str(A3[1])+str(A3[2])))
        
    Num = np.array(Num); NumS = np.sort(Num)
    
    for i1 in range(len(NumS)):
        for i2 in range(LS):
            A = NamesSCOld[i2].split('Day_20')
            A1 = A[1];
            A2 = A1.split('Z.pkl')
            A3 = A2[0].split('-');
            vA = int(str(A3[0]) + str(A3[1]) + str(A3[2]))

            B = NamesXGOld[i2].split('Day_20')
            B1 = B[1];
            B2 = B1.split('Z.txt')
            B3 = B2[0].split('-');
            vB = int(B3[0]+B3[1]+B3[2])
            if (vA == NumS[i1]):
                NamesSC.append(NamesSCOld[i2])
            if (vB == NumS[i1]):
                NamesXG.append(NamesXGOld[i2])

    NamesXG = np.array(NamesXG)
    NamesSC = np.array(NamesSC)
    return NamesXG,NamesSC

def SortGamesGoals(DirN,TeamA):
    NamesSCOld = os.listdir('../RestructuredData/%s/AllData'%(TeamA))
    NamesXGOld = os.listdir('../RestructuredData/%s/Goals'%(TeamA))

    NamesSC = []; NamesXG = []
    Num = []
    for i in range(len(DirN)):
        A = NamesSCOld[i].split('Day_20')
        A1 = A[1]; A2 = A1.split('Z.pkl')
        A3 = A2[0].split('-');
        Num.append(int(str(A3[0]) + str(A3[1])+str(A3[2])))

    Num = np.array(Num); NumS = np.sort(Num)

    for i1 in range(len(NumS)):
        for i2 in range(len(DirN)):
            A = NamesSCOld[i2].split('Day_20')
            A1 = A[1];
            A2 = A1.split('Z.pkl')
            A3 = A2[0].split('-');
            vA = int(str(A3[0]) + str(A3[1]) + str(A3[2]))

            B = NamesXGOld[i2].split('Day_20')
            B1 = B[1]
            B2 = B1.split('Z.txt')
            B3 = B2[0].split('-');
            vB = int(B3[0]+B3[1]+B3[2])
            if (vA == NumS[i1]):
                NamesSC.append(NamesSCOld[i2])
            if (vB == NumS[i1]):
                NamesXG.append(NamesXGOld[i2])

    NamesXG = np.array(NamesXG)
    NamesSC = np.array(NamesSC)

    return NamesXG

def GoalLoad(TeamA,nameXG,igame,ng):
    XGPlayers = []; XGTimes = []; XGHalf = []; XGMin = []; XGSec = []; XGNumbers = []; XGTeam = []; XGValue = []
    
    oname = "../RestructuredData/%s/Goals/%s"%(TeamA,nameXG[igame])

    xnplay = np.loadtxt(oname, delimiter=',',dtype = 'U', usecols=[3])
    xnnum = np.loadtxt(oname, delimiter=',', usecols=[4])
    xnhalf = np.loadtxt(oname, delimiter=',', usecols=[0])
    xnmin = np.loadtxt(oname, delimiter=',', usecols=[1])
    xnsec = np.loadtxt(oname, delimiter=',', usecols=[2])
    xside = np.loadtxt(oname, delimiter=',', dtype = 'U',usecols=[5])
    xntype = np.loadtxt(oname, delimiter=',', usecols=[6])
    if (ng > 1):
        for ia in range(len(xnhalf)):
            XGPlayers.append(xnplay[ia])
            XGNumbers.append(xnnum[ia])
            XGTeam.append(xside[ia])
            XGHalf.append(xnhalf[ia])
            XGMin.append(xnmin[ia])
            XGSec.append(xnsec[ia])
            XGTimes.append(xnmin[ia]*60+xnsec[ia])
            XGValue.append(xntype[ia])
    if (ng == 1):
        XGPlayers.append(xnplay)
        XGNumbers.append(xnnum)
        XGTeam.append(xside)
        XGHalf.append(xnhalf)
        XGMin.append(xnmin)
        XGSec.append(xnsec)
        XGTimes.append(xnmin*60+xnsec)
        XGValue.append(xntype)


    XGPlayers = np.array(XGPlayers); XGNumbers = np.array(XGNumbers); XGTeam = np.array(XGTeam); 
    XGHalf = np.array(XGHalf); XGMin = np.array(XGMin); XGSec = np.array(XGSec); XGTimes = np.array(XGTimes)
    XGValue = np.array(XGValue)
    return XGPlayers,XGNumbers,XGTeam,XGHalf,XGMin,XGSec,XGTimes,XGValue


def MacihneLearning_OptaLoad(TeamA,nameXG,igame):
    XGPlayers = []; XGTimes = []; XGHalf = []; XGMin = []; XGSec = []; XGNumbers = []; XGTeam = []; XGValue = []; XGPos1 = []; XGPos2 = [];

    oname = "../RestructuredData/%s/XGdata/%s"%(TeamA,nameXG[igame])
    with open(oname, 'r') as f:
        for line in f:
            line = line.strip()
            columns = line.split(',')
            XGValue.append(float(columns[0]))
            XGHalf.append(int(columns[1]))
            XGMin.append(int(columns[2]))
            XGSec.append(int(columns[3]))
            XGTimes.append(int(columns[2])*60+int(columns[3]))
            XGNumbers.append(int(columns[5]))
            XGTeam.append(columns[6]);
            XGPos1.append(columns[7]);
            XGPos2.append(columns[8]);

    return XGNumbers,XGTeam,XGValue,XGHalf,XGMin,XGSec,XGTimes,XGPos1,XGPos2
