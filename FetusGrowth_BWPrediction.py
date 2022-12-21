#!/usr/bin/env python
# coding: utf-8

from sklearn.experimental import enable_iterative_imputer
from scipy.optimize import minimize
from sklearn.impute import IterativeImputer
from matplotlib import pyplot as plt
from collections import Counter
from numpy import exp
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def makeParamList(Strparamlist):
    Fparamlist = []
    for i in range(len(Strparamlist)):
        Fparamlist.append([float(removeMultipleChar(Strparamlist[i].split()[j])) 
                           for j in range(len(Strparamlist[i].split()))])
    return(Fparamlist)

def removeMultipleChar(str_toSplit):
    original_string = str_toSplit
    characters_to_remove = '[,]'
    new_string = original_string
    for character in characters_to_remove:
        new_string = new_string.replace(character, "")
    return new_string

def minusRemove(GAdata):
    minusrmGA = []
    for i in range(len(GAdata)):
        if type(GAdata[i]) is str:
            splitga = GAdata[i].split('-')
            minusrmGA.append(min(splitga))
        else:
            minusrmGA.append(GAdata[i])
    return minusrmGA

'''Function to convert Gestational age format. '''
def GAconversion(mrmGA):
    convertedGA  =[]
    for i in range(len(mrmGA)):
        if type(mrmGA[i]) is str:
            splitga = mrmGA[i].split('+')
            if len(splitga) == 2 and splitga[1]!='':
                convertedGA.append(round((float(splitga[0])+float(splitga[1])/7),2))
            else:
                convertedGA.append(float(splitga[0]))
        else:
            convertedGA.append(mrmGA[i])
    return convertedGA

def getMeasurements_int(df):
    
    df['GA_AT_DELIVERY'] = GAconversion(minusRemove(df['GA_AT_DELIVERY']))
    df['GA'] = makeParamList(list(df['GA']))
    df['HC'] = makeParamList(list(df['HC']))
    df['AC'] = makeParamList(list(df['AC']))
    df['BPD'] = makeParamList(list(df['BPD']))
    df['FL_Rt'] = makeParamList(list(df['FL_Rt']))
    
    return df

def func(t,A,t0,c):
    return (A*exp(-exp(-c*(t-t0))))


def MSE(t0,c,A,GAValue,paramValue):
    xdata = np.array(GAValue[0])
    y = np.array(paramValue[0])
    yp = func(xdata,A[0],t0,c)
    for i in range(1,len(GAValue)):
        xdata = np.array(GAValue[i])
        ydata = np.array(paramValue[i])
        ypdata = func(xdata,A[i],t0,c)
        y = np.concatenate((y,ydata))
        yp = np.concatenate((yp,ypdata))
    return np.mean((y-yp)**2)

def MSE_one(t0,c,A, GA0, param0):
    ypdata = func(GA0,A,t0,c)
    return np.mean((param0-ypdata)**2)

def cal_A(A,t0,c,GAValue,paramValue):
    Avalue = []
    for i in range(len(GAValue)):
        xdata = np.array(GAValue[i])
        ydata = np.array(paramValue[i])
        minres = minimize(lambda x: MSE_one(t0,c,x,xdata,ydata),x0=np.array(A[i]),method = 'Nelder-Mead')
        Avalue.append(minres.x[0])
        
    return Avalue


def cal_t0c(A,t0,c,GAValue,paramValue):
    x0 = np.array([t0,c])
    mcres = minimize(lambda x: MSE(x[0],x[1],A,GAValue,paramValue), x0 = x0,method = 'Nelder-Mead')
    return mcres.x[0],mcres.x[1]


def MSPE_all(t0,c,A,t0last,clast,Alast):
    x0 = np.concatenate((np.array([t0,c]),A))
    x1 = np.concatenate((np.array([t0last,clast]),Alast))
    #return np.mean((((x0-x1)/x1)**2)**0.5) # effectively, MAPE; can also try MSE
    return np.mean(((x0-x1)/x1)**2)

def OptimizeVariable(t0,c,A,GAValue,paramValue):
    for i in range(10000):
        Alast = A
        t0last =t0
        clast = c
        A = cal_A(A, t0, c,GAValue,paramValue)
        t0,c = cal_t0c(A,t0,c,GAValue,paramValue)
        mpe = MSPE_all(t0,c,A,t0last,clast,Alast)
        mpe_data = MSE(t0,c,A,GAValue,paramValue)
        #print(i,'  ',np.mean(A),'  ', t0,'  ',c,'  ',mpe,'  ',mpe_data)
        if mpe < 0.0000000001:
            break
    return (A,t0,c,mpe,mpe_data)

# Function to calculate the mean Square error

def CalError(origdf,GAValue,paramValue,Avalue_param,t0_param,c_param):
    maperr = []
    n_badfit = int(0.10*len(origdf))
    n_goodfit = len(origdf)-n_badfit
    for i in range(len(GAValue)):
        xdata = np.array(GAValue[i])
        
        ydata = np.array(paramValue[i])
        hcp = np.array(func(xdata,Avalue_param[i],t0_param,c_param))
        err = np.sqrt(np.mean((ydata-hcp)**2))
        maperr.append([list(origdf['PATIENT_ID'])[i],err])

    maperr_arr = np.array(maperr)
    err_id = maperr_arr
    maperr_arr = maperr_arr[np.array([float(i) for i in maperr_arr[:,1]]).argsort()]
    pidLerrG = [maperr_arr[0:n_goodfit][i,0] for i in range(n_goodfit)]
    pidLerrB = [maperr_arr[n_goodfit:len(maperr_arr)][i,0] for i in range(n_badfit)]
                    
    return pidLerrG, pidLerrB, maperr, err_id


def OnlyCalError(origdf,GAValue,paramValue,Avalue_param,t0_param,c_param):
    maperr = []
    for i in range(len(GAValue)):
        xdata = np.array(GAValue[i])
        ydata = np.array(paramValue[i])
        hcp = np.array(func(xdata,Avalue_param,t0_param,c_param))
        err = np.sqrt(np.mean((ydata-hcp)**2))
        maperr.append(err)
                
    return maperr

def Binarise_BadGoodfit(origDf, goodfitID, badfitID):
    blist = []
    pidG = [origDf[origDf['PATIENT_ID'] == goodfitID[i]].index.tolist()[0] for i in range(len(goodfitID))]
    pidB = [origDf[origDf['PATIENT_ID'] == badfitID[i]].index.tolist()[0] for i in range(len(badfitID))]
    
    for i in range(len(origDf)):
        if i in pidG:
            blist.append(1)
        else:
            blist.append(0)
            
    return blist

def binarize_output(b_hc,b_ac,b_bpd,b_fl):
    binary_out = []
    for i in range(len(b_hc)):
        temp_str = str(b_hc[i])+str(b_ac[i])+str(b_bpd[i])+str(b_fl[i])
        if (temp_str == '1110' or temp_str == '1101' or temp_str == '1011' or 
            temp_str == '0111' or temp_str == '1111'):
            binary_out.append(1)
        else:
            binary_out.append(0)
    return binary_out



def plot_actual_predData(pidlist_hc,pidlist_ac,pidlist_bpd,pidlist_fl,df,t0_hc,c_hc,t0_ac,c_ac,t0_bpd,c_bpd,t0_fl,c_fl,str_val):
    
    df = df.set_index('PATIENT_ID')
    colors = ['b','g','r','c','m']
    fig, ax = plt.subplots(2, 2, figsize=(18, 14))
    
    GA_hc = df.loc[pidlist_hc]['GA']
    GA_ac = df.loc[pidlist_ac]['GA']
    GA_bpd = df.loc[pidlist_bpd]['GA']
    GA_fl = df.loc[pidlist_fl]['GA']
    
    hc_list = df.loc[pidlist_hc]['HC']
    ac_list = df.loc[pidlist_ac]['AC']
    bpd_list = df.loc[pidlist_bpd]['BPD']
    fl_list = df.loc[pidlist_fl]['FL_Rt']
    
    mseerr_hc = df.loc[pidlist_hc]['MAPE_HC']
    mseerr_ac = df.loc[pidlist_ac]['MAPE_AC']
    mseerr_bpd = df.loc[pidlist_bpd]['MAPE_BPD']
    mseerr_fl = df.loc[pidlist_fl]['MAPE_FL_Rt']
    
    A_hc = df.loc[pidlist_hc]['A_HC']
    A_ac = df.loc[pidlist_ac]['A_AC']
    A_bpd = df.loc[pidlist_bpd]['A_BPD']
    A_fl = df.loc[pidlist_fl]['A_FL_Rt']
    k = 0

    for i in range(0,5):
        xdata_hc = GA_hc[i]
        ydata_hc = hc_list[i]
        ax[0][0].scatter(xdata_hc,ydata_hc,color=colors[k]) 
        xpdata_hc = np.linspace(9,42,1000)
        pred_hc= np.array([func(xpdata_hc[j],A_hc[i],t0_hc,c_hc) for j in range(len(xpdata_hc))])
        label_str =   str(pidlist_hc[i])+" : "+str('{:.3f}'.format(mseerr_hc[i]))
        ax[0,0].plot(xpdata_hc,pred_hc,color=colors[k],alpha = 0.5,label = label_str)
        k = k+1
        ax[0][0].set_xlabel('Gestational age (weeks)')
        ax[0][0].set_ylabel('Head circumference (mm)')
        ax[0][0].set_xlim([5, 43])
        ax[0][0].set_ylim([0, 380])
        ax[0][0].legend(loc = 'upper left', frameon = False)
     
    k = 0
    for i in range(0,5):
        xdata_ac = GA_ac[i]
        ydata_ac = ac_list[i]
        ax[0][1].scatter(xdata_ac,ydata_ac,color=colors[k]) 
        xpdata_ac = np.linspace(9,42,1000)
        pred_ac= np.array([func(xpdata_ac[j],A_ac[i],t0_ac,c_ac) for j in range(len(xpdata_ac))])
        label_str =  str(pidlist_ac[i])+" : "+str('{:.3f}'.format(mseerr_ac[i]))
        ax[0,1].plot(xpdata_ac,pred_ac,color=colors[k],alpha = 0.5,label = label_str)
        k = k+1
        ax[0][1].set_xlabel('Gestational age (weeks)')
        ax[0][1].set_ylabel('Abdominal circumference (mm)')
        ax[0][1].set_xlim([5, 43])
        ax[0][1].set_ylim([0, 400])
        ax[0][1].legend(loc = 'upper left', frameon = False)
    
    k = 0
    for i in range(0,5):
        xdata_bpd = GA_bpd[i]
        ydata_bpd = bpd_list[i]
        ax[1][0].scatter(xdata_bpd,ydata_bpd,color=colors[k]) 
        xpdata_bpd = np.linspace(9,42,1000)
        pred_bpd= np.array([func(xpdata_bpd[j],A_bpd[i],t0_bpd,c_bpd) for j in range(len(xpdata_bpd))])
        label_str = str(pidlist_bpd[i])+" : "+str('{:.3f}'.format(mseerr_bpd[i]))
        ax[1,0].plot(xpdata_bpd,pred_bpd,color=colors[k],alpha = 0.5,label = label_str)
        k = k+1
        ax[1][0].set_xlabel('Gestational age (weeks)')
        ax[1][0].set_ylabel('Biparietal diameter (mm)')
        ax[1][0].set_xlim([5, 43])
        ax[1][0].set_ylim([0, 120])
        ax[1][0].legend(loc = 'upper left', frameon = False)
    
    k = 0
    for i in range(0,5):
        xdata_fl = GA_fl[i]
        ydata_fl = fl_list[i]
        ax[1,1].scatter(xdata_fl,ydata_fl,color=colors[k]) 
        xpdata_fl = np.linspace(9,42,1000)
        pred_fl= np.array([func(xpdata_fl[j],A_fl[i],t0_fl,c_fl) for j in range(len(xpdata_fl))])
        label_str = str(pidlist_fl[i])+" : "+str('{:.3f}'.format(mseerr_fl[i]))
        ax[1,1].plot(xpdata_fl,pred_fl,color=colors[k],alpha = 0.5,label = label_str)
        k = k+1
        ax[1][1].set_xlabel('Gestational age (weeks)')
        ax[1][1].set_ylabel('Femur length (mm)')
        ax[1][1].set_xlim([5, 43])
        ax[1][1].set_ylim([0, 90])
        ax[1][1].legend(loc = 'upper left', frameon = False)
    plt.tight_layout()
    plt.savefig('FigOutput/graph_'+str_val+'fit.png',dpi = 600,bbox_inches='tight')
    plt.savefig('FigOutput/graph_'+str_val+'fit.pdf',dpi = 600,bbox_inches='tight')



def dictForVisualization(common_dict):
    nc_4 = 0
    common_nc = []
    for i in range(len(common_dict)):
        common_nc = common_nc+list(common_dict.keys())[i].replace(" ", "").split(',')
        
    common_nc = dict(Counter(common_nc))

    for k,v in common_nc.items():
        if k in list(common_dict.keys()) and v > 1:
            common_nc[k] = common_dict[k]+(v-1)
            if k == '4':
                nc_4 = v-1
            else:
                pass

        if k in list(common_dict.keys()) and v == 1:
            common_nc[k] = common_dict[k]
    
    del common_nc['0']
    
    common_nc_venn = {}
    common_nc_venn.update(dict((key,value) for key, value in common_nc.items() if key == '0'))
    common_nc_venn.update({'Other Complications': sum(common_dict.values())-sum(common_nc_venn.values())})
    
    return common_nc,common_nc_venn


def hc_mean_sd(GA):
    mean = -28.2849 + 1.69267*(GA**2) - 0.397485*(GA**2)*np.log (GA)
    sd = 1.98735 + 0.0136772*(GA**3) - 0.00726264*(GA**3)*np.log(GA) + 0.000976253*(GA**3)*np.log(GA)**2
    return mean,sd


def ac_mean_sd(GA):
    mean = -81.3243 + 11.6772*GA - 0.000561865*(GA**3)
    sd = -4.36302 + 0.121445*(GA**2) - 0.0130256*(GA**3) + 0.00282143*(GA**3)*np.log(GA)
    return mean,sd


def bpd_mean_sd(GA):
    mean = 5.60878 + 0.158369*(GA**2) - 0.00256379*(GA**3)
    sd = np.exp(0.101242 + 0.00150557*(GA**3) - 0.000771535*(GA**3)*np.log(GA) + 0.0000999638*(GA**3)*np.log (GA)**2   )
    return mean,sd

def fl_mean_sd(GA):
    mean = -39.9616 + 4.32298*GA - 0.0380156*(GA**2)
    sd = np.exp(0.605843 - 42.0014*(GA**-2) + 0.00000917972*(GA**3))
    return mean,sd

# Function to calculate the mean Square error

def OnlyCalError(origdf,GAValue,paramValue,Avalue_param,t0_param,c_param):
    maperr = []
    for i in range(len(GAValue)):
        xdata = np.array(GAValue[i])
        ydata = np.array(paramValue[i])
        hcp = np.array(func(xdata,Avalue_param,t0_param,c_param))
        err = np.mean((ydata-hcp)**2)
        maperr.append(err)
                
    return maperr




def OptimizeVarAllMeasure(dataToOptimise,initA_hc,initt0_hc, initc_hc,initA_ac,initt0_ac, initc_ac,initA_bpd,
                          initt0_bpd,initc_bpd,initA_fl,initt0_fl, initc_fl):
    
    # Optimize the variables for Head Circumference
    A_hc,t0_hc,c_hc,mpe_hc,mpe_data_hc = OptimizeVariable(initt0_hc,initc_hc,
                                                    initA_hc,list(dataToOptimise['GA']),list(dataToOptimise['HC']))
    
    # Optimize the variables for Abdominal Circumference
    A_ac,t0_ac,c_ac,mpe_ac,mpe_data_ac = OptimizeVariable(initt0_ac,initc_ac,
                                                    initA_ac,list(dataToOptimise['GA']),list(dataToOptimise['AC']))
    
    # Optimize the variables for Biparietal Diameter
    A_bpd,t0_bpd,c_bpd,mpe_bpd,mpe_data_bpd = OptimizeVariable(initt0_bpd,initc_bpd,
                                                initA_bpd,list(dataToOptimise['GA']),list(dataToOptimise['BPD']))
    
    # Optimize the variables for Femur length
    A_fl,t0_fl,c_fl,mpe_fl,mpe_data_fl = OptimizeVariable(initt0_fl,initc_fl,
                                                initA_fl,list(dataToOptimise['GA']),list(dataToOptimise['FL_Rt']))
    
    return (A_hc,t0_hc,c_hc, A_ac,t0_ac,c_ac,A_bpd,t0_bpd,c_bpd,A_fl,t0_fl,c_fl)


def get_count_GALessThan35(galist):
    counter = 0
    for x in galist:
        if x <= 35:
            counter = counter+1
    return counter

def get_count_GALessThan24(galist):
    counter = 0
    for x in galist:
        if x <= 24:
            counter = counter+1
    return counter
def ImputeMissinVal(est,dataTo_impute):
    numeric_cols = dataTo_impute.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove('GA_AT_DELIVERY')
    data_new = dataTo_impute.copy()
    imputer = IterativeImputer(estimator = est, max_iter=10000, tol=1e-5, imputation_order='roman')
    imputer.fit(dataTo_impute[numeric_cols])
    data_new[numeric_cols] = imputer.transform(dataTo_impute[numeric_cols])
    
    data_new['ANAEMIA'] = np.where(np.array(dataTo_impute['ANAEMIA']) > 0.5, 1, 0)
    data_new['PRE_GDM'] = np.where(np.array(dataTo_impute['PRE_GDM']) > 0.5, 1, 0)
    data_new['THYROID'] = np.where(np.array(dataTo_impute['THYROID']) > 0.5, 1, 0)
    data_new['PREVIOUS_LSCS'] = np.where(np.array(dataTo_impute['PREVIOUS_LSCS']) > 0.5, 1, 0)
    data_new['HTN_DISORDERS_IN_PREGNANCY'] = np.where(np.array(dataTo_impute['HTN_DISORDERS_IN_PREGNANCY']) 
                                                           > 0.5, 1, 0)
    data_new['RISK_FACTOR_FOR_GDM'] = np.where(np.array(dataTo_impute['RISK_FACTOR_FOR_GDM']) > 0.5, 1, 0)
    data_new['GDM_ON_DIET'] = np.where(np.array(dataTo_impute['GDM_ON_DIET']) > 0.5, 1, 0)
    data_new['GDM_ON_METFORMIN'] = np.where(np.array(dataTo_impute['GDM_ON_METFORMIN']) > 0.5, 1, 0)
    data_new['GDM_ON_INSULIN'] = np.where(np.array(dataTo_impute['GDM_ON_INSULIN']) > 0.5, 1, 0)
    data_new['PPH'] = np.where(np.array(dataTo_impute['PPH']) > 0.5, 1, 0)
    return data_new

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    feature = list(model.feature_names_in_)
    fi = list(np.around(model.coef_, 4))
    intercept = np.around(model.intercept_, 4)
    f_coeff = list(zip(fi,feature))
    f_coeff = sorted(f_coeff, key=lambda sublist: abs(sublist[0]))
    f_coeff.reverse()
    return np.sqrt(mean_squared_error(y_test, y_pred)),y_pred,f_coeff,intercept

def rmMultipleUS(data):
    
    diff_GA = [[np.round(y-x,2) for x, y in zip(data['GA'][i][:-1], data['GA'][i][1:])] for i in range(len(data))]
    lessT1, greaterT1, lessT1_len3 = [], [], []
    for j in range(len(diff_GA)):
        if all(i >= 1 for i in diff_GA[j]):
            greaterT1.append(j)
        else:
            if len(diff_GA[j]) > 2:
                lessT1.append(j)
            else:
                lessT1_len3.append(j)

    k = 0
    for i in range(len(data)):
        if i in lessT1:
            indx_To_del = [idx for idx, element in enumerate(diff_GA[lessT1[k]]) if element < 1]
            k = k+1
            for index in sorted(indx_To_del, reverse=True):
                del data['GA'][i][index] 
                del data['HC'][i][index]
                del data['AC'][i][index]
                del data['BPD'][i][index]
                del data['FL_Rt'][i][index]

    # GA at delivery is less than the last ultrasound measurement.
    lessT1_len3 = lessT1_len3 + [i for i in range(len(data)) if list(data['GA_AT_DELIVERY'])[i] < data['GA'][i][-1]]

    data = data.drop(data.index[lessT1_len3]).reset_index(drop = True)
    return data


def IGbirthWeightPred(HC,AC):
    bw = []
    for i in range(len(HC)):
        bw.append(exp(5.084820 
                  - 54.06633*(AC[i]/1000)**3
                  - 95.80076*((AC[i]/1000)**3)*np.log(AC[i]/1000) 
                  + 3.136370*(HC[i]/1000))/1000)
    return bw


def ShepardbirthWeightPred(AC,BPD):
    bw = []
    for i in range(len(AC)):
        bw.append(10**(- 1.7492 
                       + 0.166 * (BPD[i]/10)
                       + 0.046 * (AC[i]/10)
                       - 0.002546 * (AC[i] * BPD[i])/100))
    return bw

def Hadlock1birthWeightPred(AC,FL):
    bw = []
    for i in range(len(AC)):
        bw.append(10**(1.304
                       + 0.05281 * (AC[i]/10)
                       + 0.1938 * (FL[i]/10)
                       - 0.004 * (AC[i]/10) * (FL[i]/10))/1000)
    return bw


def Hadlock2birthWeightPred(AC,BPD,FL):
    bw = []
    for i in range(len(AC)):
        bw.append(10**(1.335
                       + 0.0316 * (BPD[i]/10)
                       + 0.0457 * (AC[i]/10)
                       + 0.1623 * (FL[i]/10)
                       - 0.0034 * (AC[i]/10) * (FL[i]/10))/1000)
    return bw


def Hadlock3birthWeightPred(HC,AC,FL):
    bw = []
    for i in range(len(HC)):
        bw.append(10**(1.326
                       + 0.0107 * (HC[i]/10)
                       + 0.0438 * (AC[i]/10)
                       + 0.158 * (FL[i]/10)
                       - 0.00326 * (AC[i]/10) * (FL[i]/10))/1000)
    return bw


def Hadlock4birthWeightPred(HC,AC,BPD,FL):
    bw = []
    for i in range(len(HC)):
        bw.append(10**(1.3596
                       + 0.0064 * (HC[i]/10)
                       + 0.0424 * (AC[i]/10)
                       + 0.174 * (FL[i]/10)
                       + 0.00061 * (BPD[i]/10) * (AC[i]/10)
                       - 0.00386 * (AC[i]/10) * (FL[i]/10))/1000)
    return bw
