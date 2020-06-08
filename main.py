import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian
import numdifftools as ndt

import uncertainties
from uncertainties import ufloat

import csv

import warnings
warnings.filterwarnings('ignore')


"""To-do List:
    - need to figure out why we can't find the covariance matrix'
    - model upto March 15th before shelter in place
    """


# Global variables
N = 0
y0 = ()



"""DERIVATIVE OF SIR-----------------------------------------------------------
This function calculates and define the derviatives with our parameters.
----------------------------------------------------------------------------"""
def deriv(y, t, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, rho, a, b, c, d, f):
    S, E, I, R, D = y

    dSdt = -beta(t,L_E, k_E, t_0_E) * S * E / N  - beta(t,L_I,k_I,t_0_I) * S * I / N

    dEdt = beta(t,L_E, k_E, t_0_E) * S * E / N  + beta(t,L_I,k_I,t_0_I) * S * I / N - alpha * E

    dIdt = alpha * E - gamma * (1 - delta(t, a, b, c, d, f))* I - rho *  delta(t, a, b, c, d, f) * I

    dRdt = gamma *(1 -  delta(t, a, b, c, d, f))* I

    dDdt = rho *  delta(t, a, b, c, d, f) * I

    return dSdt, dEdt, dIdt, dRdt, dDdt


"""DERIVATIVE OF CONSTANT TERMS-----------------------------------------------
This function calculates and defines the deriatives with your constant 
parameters.
----------------------------------------------------------------------------"""
def derivConst(y, t, beta_E, beta_I, gamma, alpha, delta, rho):
    S, E, I, R, D = y
    
    dSdt = -beta_E * S * E / N  - beta_I * S * I / N

    dEdt = beta_E * S * E / N  + beta_I * S * I / N - alpha * E

    dIdt = alpha * E - gamma * (1 - delta)* I - rho *  delta * I

    dRdt = gamma *(1 -  delta)* I

    dDdt = rho *  delta * I

    return dSdt, dEdt, dIdt, dRdt, dDdt
    

"""CALCULATE TOTAL POPULATION IN THE US----------------------------------------
This function calculates the total population of the United States.
----------------------------------------------------------------------------"""
def totalPop():
    total_pop = 0
    with open("covid_county_population_usafacts(5-30).csv") as pop:
        reader_pop = csv.DictReader(pop)
        total_pop = sum (float(row["population"]) for row in reader_pop)
    return total_pop




"""CALCULATE TOTAL NUMBER OF CASES IN THE US-----------------------------------
This function calculates the total number of cases in the United States.
----------------------------------------------------------------------------"""
def totalNumberOfCases():
    # Get the total number of confirmed cases
    totConfirmed = []
    with open("covid_confirmed_usafacts(5-30).csv") as file:
        reader = csv.reader(file)

        # get the number of columns
        numDays = len(next(reader)) - 4

        totConfirmed = np.zeros(numDays)
        for row in reader:
            totConfirmed += [int(i) for i in row[4:]]

    return totConfirmed




"""CALCULATE TOTAL NUMBER OF DEATHS IN THE US----------------------------------
This function calculates the total number of deaths in the United States.
----------------------------------------------------------------------------"""
def totalNumberOfDeaths():
    # Get the total number of dead individuals from COVID-19
    totDeaths = []
    with open("covid_deaths_usafacts(5-30).csv") as file:
        reader = csv.reader(file)

        # get the number of columns
        numDays = len(next(reader)) - 4

        totDeaths = np.zeros(numDays)
        for row in reader:
            totDeaths += [int(i) for i in row[4:]]

    return totDeaths




"""INTEGRATE THE SEIRD EQUATIONS OVER TIME-------------------------------------
This function integrates the SEIRD equation over time.
----------------------------------------------------------------------------"""
def integrateEquationsOverTime(deriv, t, L_E, k_E, t_0_E, L_I, k_I, t_0_I, 
                               gamma, alpha, rho, a, b, c, d, f):
    ret = odeint(deriv, y0, t, args=(L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, 
                                     alpha, rho, a, b, c, d, f))
    S, E, I, R, D = ret.T
    return S, E, I, R, D

"""INTEGRATE THE SEIRD EQUATIONS OVER TIME FOR CONSTANT VALUES-----------------
This function integrates the SEIRD equation over time.
----------------------------------------------------------------------------"""
def integrateEquationsOverTimeConst(derivConst, t, beta_E, beta_I, gamma, alpha, 
                                    delta, rho):
    ret = odeint(derivConst, y0, t, args=(beta_E, beta_I, gamma, alpha, delta, 
                                          rho))
    S, E, I, R, D = ret.T
    return S, E, I, R, D


"""PLOT THE SEIRD MODEL--------------------------------------------------------
This function plots the SEIRD Model.
----------------------------------------------------------------------------"""
def plotSEIRD(t, S, E, I, R, D, S_q, E_q, I_q, R_q, D_q, title):
    fig, (ax, axQ) = plt.subplots(2,figsize=(10,4), sharex=True)

    fig.suptitle(title)
    
    ax.set_title('with Quarantine')

    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'm', alpha=0.7, linewidth=2, label='Exposed')
    ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, D, 'r', alpha=0.7, linewidth=2, label='Dead')
    ax.plot(t, S+E+I+R+D, 'c--', alpha=0.7, linewidth=2, label='Total')

    #ax.set_ylim(1, N)
    #ax.set_yscale('log')
    ax.set_ylim(0, 2500000)
    #ax.set_ylim(0, 6045189)
    ax.set_ylabel('Population')
    
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.e'))
    #ax.ticklabel_format(axis='y', style='sci', useMathText=True)

    # maxI = np.empty(len(t))
    # maxI.fill(max(I))
    # maxD = np.empty(len(t))
    # maxD.fill(max(D))
    legend = fig.legend()
    legend.get_frame()#.set_alpha(0.5)
    
    
    axQ.set_title('without Quarantine')
    
    axQ.plot(t, S_q, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    axQ.plot(t, E_q, 'm', alpha=0.7, linewidth=2, label='Exposed')
    axQ.plot(t, I_q, 'y', alpha=0.7, linewidth=2, label='Infected')
    axQ.plot(t, R_q, 'g', alpha=0.7, linewidth=2, label='Recovered')
    axQ.plot(t, D_q, 'r', alpha=0.7, linewidth=2, label='Dead')
    # axQ.plot(t, maxI,    'y--', alpha=0.7, linewidth=2, label='Max. Infected')
    # axQ.plot(t, maxD,    'r--', alpha=0.7, linewidth=2, label='Max. Dead')
    axQ.plot(t, S+E+I+R+D, 'c--', alpha=0.7, linewidth=2, label='Total')

    #axQ.set_ylim(1, N)
    #axQ.set_yscale('log')
    #axQ.set_ylim(0, 46000000)
    axQ.set_xlabel('Time (Days)')
    axQ.set_ylabel('Population')
    
    axQ.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.e'))
    

    
    plt.show()

    plt.clf()
    
    
    
    # fig, (axR, axD) = plt.subplots(2, sharex=True)
    
    # fig.suptitle('Fit of Dead Individuals')
    
    # axR.plot(t, residual)
    # axR.axhline(0, linestyle='--')

    # axR.set_ylabel('Residual')


    # axD.scatter(t, total_deaths, s=4, label='Data')
    # axD.plot(t, D, 'y', label='Best Fit')

    # axD.set_xlabel('Time (Days)')
    # axD.set_ylabel('Population')

    # legend = axD.legend()
    # legend.get_frame().set_alpha(0.5)


    # plt.show()
    # plt.clf()




"""FITTING THE SEIRD MODEL-----------------------------------------------------
This function varies the parameters in the SEIRD Model.
----------------------------------------------------------------------------"""
def fitter(t, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, rho, a, b, c, d, f):
    fatalityRate = delta(t, a, b, c, d, f)
    S, E, I, R, D = integrateEquationsOverTime(deriv, t, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, rho, a, b, c, d, f)
    return np.concatenate([I, D, fatalityRate])




"""BETA FUNCTION---------------------------------------------------------------
This function calcuates the rate in which individuals are becoming infected,
beta. We are using a logistics function because the rate of individuals
being infected should decrease over time as there are less suseptible
individuals.
----------------------------------------------------------------------------"""
def beta(time, L, k, t_0):
    # t_0 is the value of the sigmoid's midpoint,
    # L is the curve's maximum value,
    # k is the logistic growth rate or steepness of the curve
    beta = L/(1+np.power(np.e, k*(time-t_0)))
    return beta



"""DELTA FUNCTION--------------------------------------------------------------
This function calcuates the rate in which individuals dying,
delta. We are using a third-degree polynomial because the rate of individuals
dying should decrease over time as we get better at treating.
----------------------------------------------------------------------------"""
def  delta(t, a, b, c, d, f):
    return abs(((t-a) * (t-b) * (t-c) * d * np.exp(-f * t)) + a*b*c*d) * 0.2 


"""PLOTTING DELTA FUNCTION-----------------------------------------------------
This displays a graph of the delta function. It should have part of a bell
curve logistic function.
----------------------------------------------------------------------------"""
def plotDelta(times, cases, deaths, D, I):
    fig, axsG = plt.subplots()

    axsG.set_title('Function of Delta')
    axsG.set_xlabel('Days')
    axsG.set_ylabel('delta')
    axsG.plot(times, (deaths * 2)/(10 * cases))
    plt.show()
    plt.clf()
    
    fig, axsD = plt.subplots()

    axsD.set_title('Function of Delta')
    axsD.set_xlabel('Days')
    axsD.set_ylabel('Delta')
    axsD.plot(times, D/I)
    plt.show()
    plt.clf()




"""PLOTTING BETA FUNCTION------------------------------------------------------
This displays a graph of the beta function. It should look like a upside
logistic function.
----------------------------------------------------------------------------"""
def plotBeta(times, L_E, k_E, t_0_E, L_I, k_I, t_0_I):
    fig, ax = plt.subplots()
    
    ax.set_title('Beta Over Time')
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Beta')
    ax.plot(times, beta(times, L_E, k_E, t_0_E), 'c--', label='Exposed')
    ax.plot(times, beta(times, L_I, k_I, t_0_I), 'b', label='Infected')
    
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
        
    plt.show()
    plt.clf()




"""R_0 FUNCTION USING NGM------------------------------------------------------
This function calculates R_0 using the NGM.
----------------------------------------------------------------------------"""
def calculateR_0_NGM(time, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, a,b,c,d,f, rho):
    w = gamma * (1- delta(time,a,b,c,d,f)) + rho*delta(time,a,b,c,d,f)
    R_0 = (beta(time, L_E, k_E, t_0_E) / alpha) + (beta(time, L_I, k_E, t_0_I)/ w)  
    return R_0
 
    
"""PLOTTING R_0 FUNCTION NGM---------------------------------------------------
This displays a graph of the R0 function as modeled by the NGM. 
----------------------------------------------------------------------------"""
def plotR_0_NGM(time, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, a, b, c, d, f, rho, sigma_R_0):
    fig, (axs, axsR) = plt.subplots(2, sharex=True)

    fig.suptitle('Reproduction Number (R\u2080) Over Time')


    axsR.set_xlabel('Time (Days)')
    axs.set_ylabel('Uncertinty')

    axsR.set_ylabel('R\u2080')

    axsR.plot(time, calculateR_0_NGM(time, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, a,b,c,d,f, rho))

    axs.plot(time, sigma_R_0)

    plt.show()
    plt.clf()




"""PLOT THE BEST FIT OF INFECTED-----------------------------------------------
This function plots the data of cases and the best fit for our model.
----------------------------------------------------------------------------"""
def plotBestFitInfected(t, I, total_con, residual):
    fig, (axR1, axI) = plt.subplots(2, sharex=True)
    
    fig.suptitle('Fit of Infected Individuals')
  
    
    axR1.plot(t, residual)
    axR1.axhline(0, linestyle='--')

    axR1.set_ylabel('Residual')


    axI.scatter(t, total_con, s=4, label='Data')
    axI.plot(t, I, 'y', label='Best fit')

    axI.set_xlabel('Time (Days)')
    axI.set_ylabel('Population')

    legend = axI.legend()
    legend.get_frame().set_alpha(0.5)


    plt.show()
    plt.clf()
    

def plotBestFit(t, I, D, total_con, total_deaths, residual):
    fig, ((axIR, axDR), (axI, axD)) = plt.subplots(2,2, sharex=True)
    
    #fig.suptitle('Fitting to Data')
    axI.set_xlabel('Time (Days)')
    axD.set_xlabel('Time (Days)')
    
    #Infected
    axIR.plot(t, residual[:130])
    axIR.axhline(0, linestyle='--')

    axIR.set_ylabel('Residual')
    axIR.set_title('Fit of Infected')

    axI.scatter(t, total_con, s=4, label='Data')
    axI.plot(t, I, 'y', label='Best fit')

    axI.set_ylabel('Population')

    legendI = axI.legend()
    legendI.get_frame().set_alpha(0.5)
    
    
    #Dead
    axDR.plot(t, residual[130:260])
    axDR.axhline(0, linestyle='--')
    
    axDR.set_title('Fit of Dead')

    axD.scatter(t, total_deaths, s=4, label='Data')
    axD.plot(t, D, 'y', label='Best Fit')

    legendD = axD.legend()
    legendD.get_frame().set_alpha(0.5)
    
    plt.tight_layout()
    plt.show()
    plt.clf()

    
    
    
    


"""PLOT THE BEST FIT OF DEAD---------------------------------------------------
This function plots the data of dead and the best fit for our model.
----------------------------------------------------------------------------"""
def plotBestFitDied(t, D, total_deaths, residual):
    fig, (axR, axD) = plt.subplots(2, sharex=True)
    
    fig.suptitle('Fit of Dead Individuals')
    
    axR.plot(t, residual)
    axR.axhline(0, linestyle='--')

    axR.set_ylabel('Residual')


    axD.scatter(t, total_deaths, s=4, label='Data')
    axD.plot(t, D, 'y', label='Best Fit')

    axD.set_xlabel('Time (Days)')
    axD.set_ylabel('Population')

    legend = axD.legend()
    legend.get_frame().set_alpha(0.5)


    plt.show()
    plt.clf()
    


"""PLOT THE BEST FIT OF DELTA---------------------------------------------------
This function plots the data of delta and the best fit for our model.
----------------------------------------------------------------------------""" 
def plotBestFitDelta(t, D, I, total_deaths, total_con, residual):
    fig, (axR, axD) = plt.subplots(2, sharex=True)
    
    fig.suptitle('Fit of Fatality Rate')
    
    axR.plot(t, residual)
    axR.axhline(0, linestyle='--')


    axR.set_ylabel('Residual')


    axD.scatter(t, total_deaths/total_con, s=4, label='Data')
    axD.plot(t, D/I , 'y', label='Best fit')

    axD.set_xlabel('Time (Days)')
    axD.set_ylabel('Fatality Rate')

    legend = axD.legend()
    legend.get_frame().set_alpha(0.5)

    plt.show()
    plt.clf()
    
    
"""ERROR PROPOGATION----------------------------------------------------------
This function computes the variance of R_0. This is a manual function.
----------------------------------------------------------------------------"""
def errorProp(t,  L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, a, b, c, d, f, rho):
    R_0 = (L_E/(1+np.power(np.e, k_E*(t-t_0_E))))/alpha + (L_I/(1+np.power(np.e, k_I*(t-t_0_I))))/(gamma * (1- (abs(((t-a) * (t-b) * (t-c) * d * np.exp(-f * t)) + a*b*c*d) * 0.2 )) + rho*(abs(((t-a) * (t-b) * (t-c) * d * np.exp(-f * t)) + a*b*c*d) * 0.2 ))
    w = gamma * (1- delta(t,a,b,c,d,f)) + rho*delta(t,a,b,c,d,f)
    
    
    
    dR_0dL_E = 1/(1+np.power(np.e, k_E*(t-t_0_E)))/alpha
    
    dR_0dk_E = -beta(t, L_E, k_E ,t_0_E) * (t - t_0_E) * np.power(np.e, k_E*(t-t_0_E))/(alpha * (1 + np.power(np.e, k_E*(t-t_0_E))))
    
    dR_0dt_0_E = (k_E * np.power(np.e, k_E*(t-t_0_E)) * beta(t, L_E, k_E, t_0_E))/ (alpha * (1 + np.power(np.e, k_E*(t-t_0_E))))
    
    dR_0dL_I = (1/(1+np.power(np.e, k_I*(t-t_0_I))))/(gamma * (1- (abs(((t-a) * (t-b) * (t-c) * d * np.exp(-f * t)) + a*b*c*d) * 0.2 )) + rho*(abs(((t-a) * (t-b) * (t-c) * d * np.exp(-f * t)) + a*b*c*d) * 0.2 ))
    
    dR_0dk_I = -beta(t, L_I, k_I ,t_0_I) * (t - t_0_I) * np.power(np.e, k_I*(t-t_0_I))/(w * (1 + np.power(np.e, k_I*(t-t_0_I))))
    
    dR_0dt_0_I = (k_I * np.power(np.e, k_I*(t-t_0_I)) * beta(t, L_I, k_I, t_0_I))/ (w * (1 + np.power(np.e, k_I*(t-t_0_I))))
    
    dR_0dgamma = -1 * beta(t, L_I, k_I, t_0_I)/(gamma**2 * (1-delta(t, a, b, c, d, f)))
    
    dR_0dalpha = -1 * beta(t, L_E, k_E ,t_0_E)/alpha**2
    
    dR_0da = (-((rho*(b*c*d-d*(t-b)*(t-c)*np.exp(-f*t))*(b*c*d*a+d*(t-b)*(t-c)*np.exp(-f*t)*(t-a)))/(5*abs(b*c*d*a+d*(t-b)*(t-c)*np.exp(-f*t)*(t-a)))-(gamma*(b*c*d-d*(t-b)*(t-c)*np.exp(-f*t))*(b*c*d*a+d*(t-b)*(t-c)*np.exp(-f*t)*(t-a)))/(5*abs(b*c*d*a+d*(t-b)*(t-c)*np.exp(-f*t)*(t-a))))/((rho*abs(b*c*d*a+d*(t-b)*(t-c)*np.exp(-f*t)*(t-a)))/5+gamma*(1-abs(b*c*d*a+d*(t-b)*(t-c)*np.exp(-f*t)*(t-a))/5))**2) * beta(t, L_I, k_I, t_0_I)
    
    dR_0db = (-((rho*(a*c*d-d*(t-a)*(t-c)*np.exp(-f*t))*(a*c*d*b+d*(t-a)*(t-c)*np.exp(-f*t)*(t-b)))/(5*abs(a*c*d*b+d*(t-a)*(t-c)*np.exp(-f*t)*(t-b)))-(gamma*(a*c*d-d*(t-a)*(t-c)*np.exp(-f*t))*(a*c*d*b+d*(t-a)*(t-c)*np.exp(-f*t)*(t-b)))/(5*abs(a*c*d*b+d*(t-a)*(t-c)*np.exp(-f*t)*(t-b))))/((rho*abs(a*c*d*b+d*(t-a)*(t-c)*np.exp(-f*t)*(t-b)))/5+gamma*(1-abs(a*c*d*b+d*(t-a)*(t-c)*np.exp(-f*t)*(t-b))/5))**2) * beta(t, L_I, k_I, t_0_I)
    
    dR_0dc = (-((rho*(a*b*d-d*(t-a)*(t-b)*np.exp(-f*t))*(a*b*d*c+d*(t-a)*(t-b)*np.exp(-f*t)*(t-c)))/(5*abs(a*b*d*c+d*(t-a)*(t-b)*np.exp(-f*t)*(t-c)))-(gamma*(a*b*d-d*(t-a)*(t-b)*np.exp(-f*t))*(a*b*d*c+d*(t-a)*(t-b)*np.exp(-f*t)*(t-c)))/(5*abs(a*b*d*c+d*(t-a)*(t-b)*np.exp(-f*t)*(t-c))))/((rho*abs(a*b*d*c+d*(t-a)*(t-b)*np.exp(-f*t)*(t-c)))/5+gamma*(1-abs(a*b*d*c+d*(t-a)*(t-b)*np.exp(-f*t)*(t-c))/5))**2) * beta(t, L_I, k_I, t_0_I)
    
    dR_0dd = (-((rho*((t-a)*(t-b)*(t-c)*np.exp(-f*t)+a*b*c)*((t-a)*(t-b)*(t-c)*np.exp(-f*t)*d+a*b*c*d))/(5*abs((t-a)*(t-b)*(t-c)*np.exp(-f*t)*d+a*b*c*d))-(gamma*((t-a)*(t-b)*(t-c)*np.exp(-f*t)+a*b*c)*((t-a)*(t-b)*(t-c)*np.exp(-f*t)*d+a*b*c*d))/(5*abs((t-a)*(t-b)*(t-c)*np.exp(-f*t)*d+a*b*c*d)))/((rho*abs((t-a)*(t-b)*(t-c)*np.exp(-f*t)*d+a*b*c*d))/5+gamma*(1-abs((t-a)*(t-b)*(t-c)*np.exp(-f*t)*d+a*b*c*d)/5))**2) * beta(t, L_I, k_I, t_0_I)
    
    dR_0df = ((5*np.power(d,2)*(rho-gamma)*t*(t-a)*(t-b)*(t-c)*np.exp(-2*t*f)*(a*b*c*np.exp(t*f)+np.power(t,3)+(-c-b-a)*np.power(t,2)+((b+a)*c+a*b)*t-a*b*c))/(abs(d*(t-a)*(t-b)*(t-c)*np.exp(-t*f)+a*b*c*d)*np.power(((rho-gamma)*abs(d*(t-a)*(t-b)*(t-c)*np.exp(-t*f)+a*b*c*d)+5*gamma),2))) * beta(t, L_I, k_I, t_0_I)
    
    dR_0drho = -1 * beta(t, L_I, k_I, t_0_I)/(rho**2 * delta(t, a, b, c, d, f))
    
    sigL_E = 35.7762578
    sigk_E = 0.89701212
    sigt_0_E = 21785.9891
    sigL_I = 76.8633892
    sigk_I = 76.8633892
    sigt_0_I = 1.56513785
    sig_gamma = 0.23118479
    sig_alpha = 12.9266165
    sig_a = 25.4713490
    sig_b = 60.7551424
    sig_c = 8.69608492
    sig_d = 0.00797996
    sig_f = 0.00735950
    sig_rho = 190.565404
    
    sig_Lnk_E = -0.999
    sig_Lnt_E = -0.978
    sig_L_EnL_I = -0.996
    sig_L_Enk_I = 0.152
    sig_L_Ent_I = -0.556
    sig_Ln_gamma = -0.466 
    sig_Ln_alpha = 0.995
    sig_Lna = -0.326
    sig_Lnb = 0.123
    sig_Lnc = 0.363
    sig_Lnd = -0.305
    sig_Lnf = 0.099
    sig_Lnrho = 0.305
    
    sig_knt_E = 0.969
    sig_k_EnL_I = 0.998
    sig_k_Enk_I = -0.119
    sig_k_Ent_I = 0.586
    sig_k_En_gamma = 0.498
    sig_k_En_alpha = -0.998
    sig_k_Ena = 0.309
    sig_k_Enb = -0.131
    sig_k_Enc = -0.350
    sig_k_End = 0.340
    sig_k_Enf = 0.099
    sig_k_Enrho = -0.340
    
    sig_t_EnL_I = 0.954
    sig_t_Enk_I = -0.332
    sig_t_Ent_I = 0.372
    sig_t_En_gamma = 0.284
    sig_t_En_alpha = -0.953
    sig_t_Ena = 0.412
    sig_t_Enb = 0.099
    sig_t_Enc = -0.422
    sig_t_End = 0.118
    sig_t_Enf = 0.099
    sig_t_Enrho = -0.118
    
    sig_Lnk_I = 0.099
    sig_Lnt_I = 0.629
    sig_L_In_gamma = 0.540
    sig_L_In_alpha = -1.000
    sig_L_Ina = 0.284
    sig_L_Inb = 0.146
    sig_L_Inc = -0.331
    sig_L_Ind = 0.384
    sig_L_Inf = 0.099
    sig_L_Inrho = -0.384
    
    sig_knt_I = 0.675
    sig_k_In_gamma = 0.671
    sig_k_In_alpha = 0.099
    sig_k_Ina = -0.789
    sig_k_Inb = 0.099
    sig_k_Inc = 0.630
    sig_k_Ind = 0.784
    sig_k_Inf = -0.271
    sig_k_Inrho = -0.784
    
    sig_t_In_gamma = 0.932
    sig_t_In_alpha = -0.631
    sig_t_Ina = -0.268
    sig_t_Inb = -0.172
    sig_t_Inc = 0.152
    sig_t_Ind = 0.885
    sig_t_Inf = -0.278
    sig_t_Inrho = -0.885
    
    sig_gamman_alpha = -0.543
    sig_gammana = -0.232
    sig_gammanb = -0.149
    sig_gammanc = 0.099
    sig_gammand = 0.978
    sig_gammanf = -0.244
    sig_gammanrho = -0.978
    
    sig_alphana = -0.283
    sig_alphanb = 0.146
    sig_alphanc = 0.330
    sig_alphand = -0.386
    sig_alphanf = 0.099
    sig_alphanrho = 0.386
    
    sig_anb = -0.363
    sig_anc = -0.744
    sig_and = -0.387
    sig_anf = 0.099
    sig_anrho = 0.386
    
    sig_bnc = 0.099
    sig_bnd = 0.099
    sig_bnf = 0.942
    sig_bnrho = 0.099
    
    sig_cnd = -0.222
    sig_cnf = -0.141
    sig_cnrho = -0.222
    
    sig_dnf = -0.214
    sig_dnrho = -1.000
    
    sig_fnrho = 0.217
    
    
    sigma_R_0 = ((abs(dR_0dL_E)**2 * sigL_E**2 
                 + abs(dR_0dk_E)**2 * sigk_E**2 
                 + abs(dR_0dt_0_E)**2 * sigt_0_E**2
                 + abs(dR_0dL_I)**2 * sigL_I**2
                 + abs(dR_0dk_I)**2 * sigk_I**2
                 + abs(dR_0dt_0_I)**2 * sigt_0_I**2
                 + abs(dR_0dgamma)**2 * sig_gamma**2
                 + abs(dR_0dalpha)**2 * sig_alpha**2
                 + abs(dR_0da)**2 * sig_a**2
                 + abs(dR_0db)**2 * sig_b**2
                 + abs(dR_0dc)**2 * sig_c**2
                 + abs(dR_0dd)**2 * sig_d**2
                 + abs(dR_0df)**2 * sig_f**2
                 + abs(dR_0drho)**2 * sig_rho**2) 
                 + (2 * dR_0dL_E * (dR_0dk_E * sig_Lnk_E
                 + dR_0dt_0_E * sig_Lnt_E
                 + dR_0dL_I * sig_L_EnL_I
                 + dR_0dk_I * sig_L_Enk_I
                 + dR_0dt_0_I * sig_L_Ent_I
                 + dR_0dgamma * sig_Ln_gamma
                 + dR_0dalpha * sig_Ln_alpha
                 + dR_0da * sig_Lna
                 + dR_0db * sig_Lnb
                 + dR_0dc * sig_Lnc
                 + dR_0dd * sig_Lnd
                 + dR_0df * sig_Lnf
                 + dR_0drho * sig_Lnrho))
                 + (2 * dR_0dk_E * (dR_0dt_0_E * sig_knt_E
                 + dR_0dL_I * sig_k_EnL_I
                 + dR_0dk_I * sig_k_Enk_I
                 + dR_0dt_0_I * sig_k_Ent_I
                 + dR_0dgamma * sig_k_En_gamma
                 + dR_0dalpha * sig_k_En_alpha
                 + dR_0da * sig_k_Ena
                 + dR_0db * sig_k_Enb
                 + dR_0dc * sig_k_Enc
                 + dR_0dd * sig_k_End
                 + dR_0df * sig_k_Enf
                 + dR_0drho * sig_k_Enrho))
                 + (2 * dR_0dt_0_E * (dR_0dL_I * sig_t_EnL_I
                 + dR_0dk_I * sig_t_Enk_I
                 + dR_0dt_0_I * sig_t_Ent_I
                 + dR_0dgamma * sig_t_En_gamma
                 + dR_0dalpha * sig_t_En_alpha
                 + dR_0da * sig_t_Ena
                 + dR_0db * sig_t_Enb
                 + dR_0dc * sig_t_Enc
                 + dR_0dd * sig_t_End
                 + dR_0df * sig_t_Enf
                 + dR_0drho * sig_t_Enrho))
                 + (2 * dR_0dL_I * (dR_0dk_I * sig_Lnk_I
                 + dR_0dt_0_I * sig_Lnt_I
                 + dR_0dgamma * sig_L_In_gamma
                 + dR_0dalpha * sig_L_In_alpha
                 + dR_0da * sig_L_Ina
                 + dR_0db * sig_L_Inb
                 + dR_0dc * sig_L_Inc
                 + dR_0dd * sig_L_Ind
                 + dR_0df * sig_L_Inf
                 + dR_0drho * sig_L_Inrho))
                 + (2 * dR_0dk_I * (dR_0dt_0_I * sig_knt_I
                 + dR_0dgamma * sig_k_In_gamma
                 + dR_0dalpha * sig_k_In_alpha
                 + dR_0da * sig_k_Ina
                 + dR_0db * sig_k_Inb
                 + dR_0dc * sig_k_Inc
                 + dR_0dd * sig_k_Ind
                 + dR_0df * sig_k_Inf
                 + dR_0drho * sig_k_Inrho))
                 + (2 * dR_0dt_0_I * (dR_0dgamma * sig_t_In_gamma
                 + dR_0dalpha * sig_t_In_alpha
                 + dR_0da * sig_t_Ina
                 + dR_0db * sig_t_Inb
                 + dR_0dc * sig_t_Inc
                 + dR_0dd * sig_t_Ind
                 + dR_0df * sig_t_Inf
                 + dR_0drho * sig_t_Inrho))
                 + (2 * dR_0dgamma * (dR_0dalpha * sig_gamman_alpha
                 + dR_0da * sig_gammana
                 + dR_0db * sig_gammanb
                 + dR_0dc * sig_gammanc
                 + dR_0dd * sig_gammand
                 + dR_0df * sig_gammanf
                 + dR_0drho * sig_gammanrho))
                 + (2 * dR_0dalpha * (dR_0da * sig_alphana
                 + dR_0db * sig_alphanb
                 + dR_0dc * sig_alphanc
                 + dR_0dd * sig_alphand
                 + dR_0df * sig_alphanf
                 + dR_0drho * sig_alphanrho))
                 + (2 * dR_0da * (dR_0db * sig_anb
                 + dR_0dc * sig_anc
                 + dR_0dd * sig_and
                 + dR_0df * sig_anf
                 + dR_0drho * sig_anrho))
                 + (2 * dR_0db * (dR_0dc * sig_bnc
                 + dR_0dd * sig_bnd
                 + dR_0df * sig_bnf
                 + dR_0drho * sig_bnrho))
                 + (2 * dR_0dc * (dR_0dd * sig_cnd
                 + dR_0df * sig_cnf
                 + dR_0drho * sig_cnrho))
                 + (2 * dR_0dd * (dR_0df * sig_dnf
                 + dR_0drho * sig_dnrho))
                 + (2 * dR_0df * (dR_0drho * sig_fnrho)))
    
    return sigma_R_0**0.5


# """ERROR PROPOGATION PLOT------------------------------------------------------
# This function computes the variance of R_0. This is a manual function.
# ----------------------------------------------------------------------------"""

def plotErrorProp(t, sigma_R_0):
    
    fig, axsR = plt.subplots()

    axsR.set_title('Error of Reproduction Number (R\u2080)')
    axsR.set_xlabel('Time (Days)')
    axsR.set_ylabel('Uncertinty of Reproduction Number (R\u2080)')

    axsR.plot(t, sigma_R_0)

    plt.show()
    plt.clf()
    


if __name__ == "__main__":
    total_con = totalNumberOfCases()
    total_deaths = totalNumberOfDeaths()
    
    # define constants
    N = totalPop()
    y0 = (N - 1, 0, 1, 0, 0)  # initial conditions: one infected, rest susceptible
    moreTimes = np.linspace(0, 365-1, 365)
    
    #MAKING AN ARRAY
    times = np.linspace(0, len(total_con)-1, len(total_con)) # time (in days)
    mod = lmfit.Model(fitter)

    #Exposed Beta -- Rate of Transmission between SUSCEPTIBLE and EXPOSED
    mod.set_param_hint('L_E', min=0, max = 6) # L is the curve's maximum value
    mod.set_param_hint('k_E', min = 0, max = 0.15) # k is the logistic growth rate or steepness of the curve
    mod.set_param_hint('t_0_E', min=1, max=365) # t_0 is the value of the sigmoid's midpoint

    #Infected Beta -- Rate of Transmission between SUSCEPTIBLE and INFECTED
    mod.set_param_hint('L_I', min=0, max = 6) # L is the curve's maximum value
    mod.set_param_hint('k_I', min = 0, max = 0.5) # k is the logistic growth rate or steepness of the curve
    mod.set_param_hint('t_0_I', min=1, max=365) # t_0 is the value of the sigmoid's midpoint

    #Gamma -- Rate of Recovery (sick for 2 weeks up to 6)
    mod.set_param_hint('gamma', min= 1/(6*7), max = 1/12)#0.02, max=0.1)

    #Alpha -- Incubation period (5-6 days up to 14 days)
    mod.set_param_hint('alpha', min = 0.0714, max=0.2)#0.01667, max=0.1)

    #Rho -- Rate people die from infection
    mod.set_param_hint('rho', min = 0)#, max = 0.5)
    
    #Delta -- Fatality Rate (2% fatality rate)
    mod.set_param_hint('a', min = 0)
    mod.set_param_hint('b', min = 0)
    mod.set_param_hint('c', min = 0)
    mod.set_param_hint('d', min = 0)
    mod.set_param_hint('f', min = 0)
    
    
    
    #Puts Total number of case and deaths in 1 array to calculate best fit
    data = []
    data = np.concatenate([total_con, total_deaths, total_deaths/total_con])

    params = mod.make_params(verbose=True)
    result = mod.fit(data,
                     params,
                     t=times,
                      L_E = 0.2,
                      k_E = 0.0045,
                      t_0_E = 56, 
                      L_I = 0.47,
                      k_I = 0.47,
                      t_0_I = 67,
                      gamma = 0.023, 
                      alpha = 0.0787, 
                      rho = 0.63,
                      a = 2.86, 
                      b = 2.86, 
                      c = 2.86, 
                      d = 0.0000216, 
                      f = 0.073)

          
    #Prints L's, k's, t_0's, gamma, alpha, rho
    print(result.fit_report())
    
    #print(result.eval_uncertainty())
    
    
    print('Maximum of Beta for Exposed: ', max(beta(times,
                                                    result.best_values['L_I'],
                                        result.best_values['k_I'],
                                        result.best_values['t_0_I'])))
                                                    
                                        # result.best_values['L_E'],
                                        # result.best_values['k_E'],
                                        # result.best_values['t_0_E'])))
        
    print('Maximum of Beta for Infected: ', max(beta(times,
                                        result.best_values['L_I'],
                                        result.best_values['k_I'],
                                        result.best_values['t_0_I'])))
    
    print('Maximum of Delta: ', max(delta(times, 
                                          result.best_values['a'],
                                          result.best_values['b'],
                                          result.best_values['c'],
                                          result.best_values['d'],
                                          result.best_values['f'])))
    

    print('Maximum R_0_NGM: ', max(calculateR_0_NGM(times,
                                                    result.best_values['L_E'],
                                                    result.best_values['k_E'],
                                                    result.best_values['t_0_E'],
                                                    result.best_values['L_I'],
                                                    result.best_values['k_I'],
                                                    result.best_values['t_0_I'],
                                                    result.best_values['gamma'],
                                                    result.best_values['alpha'],
                                                    result.best_values['a'],
                                                      result.best_values['b'],
                                                      result.best_values['c'],
                                                      result.best_values['d'],
                                                      result.best_values['f'],
                                                    result.best_values['rho'])))

    print('Minimun R_0_NGM: ', min(calculateR_0_NGM(times,
                                                    result.best_values['L_E'],
                                                    result.best_values['k_E'],
                                                    result.best_values['t_0_E'],
                                                    result.best_values['L_I'],
                                                    result.best_values['k_I'],
                                                    result.best_values['t_0_I'],
                                                    result.best_values['gamma'],
                                                    result.best_values['alpha'],
                                                    result.best_values['a'],
                                                      result.best_values['b'],
                                                      result.best_values['c'],
                                                      result.best_values['d'],
                                                      result.best_values['f'],
                                                    result.best_values['rho'])))

    print('Mean R_0_NGM: ', (max(calculateR_0_NGM(times,
                                                    result.best_values['L_E'],
                                                    result.best_values['k_E'],
                                                    result.best_values['t_0_E'],
                                                    result.best_values['L_I'],
                                                    result.best_values['k_I'],
                                                    result.best_values['t_0_I'],
                                                    result.best_values['gamma'],
                                                    result.best_values['alpha'],
                                                    result.best_values['a'],
                                                      result.best_values['b'],
                                                      result.best_values['c'],
                                                      result.best_values['d'],
                                                      result.best_values['f'],
                                                    result.best_values['rho'])) + 
                             min(calculateR_0_NGM(times,
                                                    result.best_values['L_E'],
                                                    result.best_values['k_E'],
                                                    result.best_values['t_0_E'],
                                                    result.best_values['L_I'],
                                                    result.best_values['k_I'],
                                                    result.best_values['t_0_I'],
                                                    result.best_values['gamma'],
                                                    result.best_values['alpha'],
                                                    result.best_values['a'],
                                          result.best_values['b'],
                                          result.best_values['c'],
                                          result.best_values['d'],
                                          result.best_values['f'],
                                                    result.best_values['rho'])))/2)
    
    
    #Plot Betas
    plotBeta(times, result.best_values['L_E'],
                    result.best_values['k_E'],
                    result.best_values['t_0_E'], 
                    result.best_values['L_I'],
                    result.best_values['k_I'],
                    result.best_values['t_0_I'])    
    
    #Plot R_0
    plotR_0_NGM(times,
                result.best_values['L_E'],
                result.best_values['k_E'],
                result.best_values['t_0_E'],
                result.best_values['L_I'],
                result.best_values['k_I'],
                result.best_values['t_0_I'],
                result.best_values['gamma'],
                result.best_values['alpha'],
                result.best_values['a'],
                result.best_values['b'],
                result.best_values['c'],
                result.best_values['d'],
                result.best_values['f'],
                result.best_values['rho'], 
                errorProp(times, 
                        result.best_values['L_E'],
                        result.best_values['k_E'],
                        result.best_values['t_0_E'],
                        result.best_values['L_I'],
                        result.best_values['k_I'],
                        result.best_values['t_0_I'],
                        result.best_values['gamma'],
                        result.best_values['alpha'],
                        result.best_values['a'],
                        result.best_values['b'],
                        result.best_values['c'],
                        result.best_values['d'],
                        result.best_values['f'],
                        result.best_values['rho']))
    
    


    #Integrate SEIRD -- With Data
    S, E, I, R, D = integrateEquationsOverTime(deriv,
                                               times,
                                               result.best_values['L_E'],
                                               result.best_values['k_E'],
                                               result.best_values['t_0_E'],
                                               result.best_values['L_I'],
                                               result.best_values['k_I'],
                                               result.best_values['t_0_I'],
                                               result.best_values['gamma'],
                                               result.best_values['alpha'],
                                               result.best_values['rho'],
                                               result.best_values['a'],
                                               result.best_values['b'],
                                               result.best_values['c'],
                                               result.best_values['d'],
                                               result.best_values['f'])

    #Integrate SEIRD -- Year Out
    S_y, E_y, I_y, R_y, D_y = integrateEquationsOverTime(deriv,
                                                moreTimes,
                                                result.best_values['L_E'],
                                                result.best_values['k_E'],
                                                result.best_values['t_0_E'],
                                                result.best_values['L_I'],
                                                result.best_values['k_I'],
                                                result.best_values['t_0_I'],
                                                result.best_values['gamma'],
                                                result.best_values['alpha'],
                                                result.best_values['rho'],
                                                result.best_values['a'],
                                                result.best_values['b'],
                                                result.best_values['c'],
                                                result.best_values['d'],
                                                result.best_values['f'])
     
    #Integrate SEIRD -- Without Quarentine 
    S_wo, E_wo, I_wo, R_wo, D_wo = integrateEquationsOverTimeConst(derivConst,
                                                                        times, 
                                                                        max(beta(times,
                                                                         result.best_values['L_E'],
                                                                         result.best_values['k_E'],
                                                                         result.best_values['t_0_E'])),
                                                                        max(beta(times,
                                                                         result.best_values['L_I'],
                                                                         result.best_values['k_I'],
                                                                         result.best_values['t_0_I'])), 
                                                                        result.best_values['gamma'], 
                                                                        result.best_values['alpha'], 
                                                                        max(delta(times, result.best_values['a'],
                                                                         result.best_values['b'],
                                                                         result.best_values['c'],
                                                                         result.best_values['d'],
                                                                         result.best_values['f'])), 
                                                                        result.best_values['rho'])
    
    #Integrate SEIRD -- Without Quarentine a year out
    S_woq, E_woq, I_woq, R_woq, D_woq = integrateEquationsOverTimeConst(derivConst,
                                                                        moreTimes, 
                                                                        max(beta(times,
                                                                         result.best_values['L_E'],
                                                                         result.best_values['k_E'],
                                                                         result.best_values['t_0_E'])),
                                                                        max(beta(times,
                                                                         result.best_values['L_I'],
                                                                         result.best_values['k_I'],
                                                                         result.best_values['t_0_I'])), 
                                                                        result.best_values['gamma'], 
                                                                        result.best_values['alpha'], 
                                                                        max(delta(times, 
                                                                         result.best_values['a'],
                                                                         result.best_values['b'],
                                                                         result.best_values['c'],
                                                                         result.best_values['d'],
                                                                         result.best_values['f'])), 
                                                                        result.best_values['rho'])
    
    #Create an numpy array
    residualOfIandD = result.residual
    

    #Plot Residual and Best Fit
    plotBestFitInfected(times, I, total_con, residualOfIandD[:130])
    plotBestFitDied(times, D, total_deaths, residualOfIandD[130:260])
    plotBestFitDelta(times, D, I, total_deaths, total_con, residualOfIandD[260:])
    
    plotBestFit(times, I, D, total_con, total_deaths, residualOfIandD)


    print('Population of the US:', N)
    print('Total Number of Deaths:', max(total_deaths))
    print('Total Number of Cases:', max(totalNumberOfCases()))
    print('Suspetible:', min(S))
    print('Exposed:', max(E))
    print('Infected:', max(I))
    print('Recovered:', max(R)) # should be somewhere around 300,000
    print('Dead:',max(D)) # should equal total dead

    total = S+I+E+R+D

    print('Total:', min(total)) # should equal total population

    #Plot SEIRD model -- Based on data with and without quarentine
    plotSEIRD(times, S, E, I, R, D, S_wo, E_wo, I_wo, R_wo, D_wo, 'SEIRD Model of COVID-19')
        
    #Plot SEIRD model -- A year out
    plotSEIRD(moreTimes, S_y, E_y, I_y, R_y, D_y, S_woq, E_woq, I_woq, R_woq, D_woq, 'Projected SEIRD Model of COVID-19')