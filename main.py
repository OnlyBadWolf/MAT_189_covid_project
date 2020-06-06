import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian
import numdifftools as ndt

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
This function calculates and define the derviatives for the data.
----------------------------------------------------------------------------"""
def deriv(y, t, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, rho, a, b, c, d, f):# mu, sig, phi):
    S, E, I, R, D = y

    dSdt = -beta(t,L_E, k_E, t_0_E) * S * E / N  - beta(t,L_I,k_I,t_0_I) * S * I / N

    dEdt = beta(t,L_E, k_E, t_0_E) * S * E / N  + beta(t,L_I,k_I,t_0_I) * S * I / N - alpha * E

    dIdt = alpha * E - gamma * (1 - delta(t, a, b, c, d, f))* I - rho *  delta(t, a, b, c, d, f) * I

    dRdt = gamma *(1 -  delta(t, a, b, c, d, f))* I

    dDdt = rho *  delta(t, a, b, c, d, f) * I

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




"""INTEGRATE THE SIR EQUATIONS OVER TIME---------------------------------------
This function integrates the SIR equation over time.
----------------------------------------------------------------------------"""
def integrateEquationsOverTime(deriv, t, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, rho, a, b, c, d, f):
    ret = odeint(deriv, y0, t, args=(L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, rho, a, b, c, d, f))
    S, E, I, R, D = ret.T
    return S, E, I, R, D




"""PLOT THE SEIRD MODEL--------------------------------------------------------
This function plots the SEIRD Model.
----------------------------------------------------------------------------"""
def plotSEIRD(t, S, E, I, R, D, title):
    f, ax = plt.subplots(1,1,figsize=(10,4))

    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'm', alpha=0.7, linewidth=2, label='Exposed')
    ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, D, 'r', alpha=0.7, linewidth=2, label='Died')
    ax.plot(t, S+E+I+R+D, 'c--', alpha=0.7, linewidth=2, label='Total')

    ax.set_ylim(1, N)
    ax.set_yscale('log')
    #ax.set_ylim(0, 2500000)
    #ax.set_ylim(0, 6045189)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title(title)


    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    plt.show()

    plt.clf()




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




"""GASSIAN FUNCTION------------------------------------------------------
This calculates the gaussian function.
----------------------------------------------------------------------------"""
def gaussianG(t, mu, sig, phi):
    # mu is the position of center peak
    # sig is the standard deviation 
    # phi is the height of curve's peak
    return phi * np.exp(-np.power(t - mu, 2) / (2 * np.power(sig, 2)))



"""DELTA FUNCTION--------------------------------------------------------------
This function calcuates the rate in which individuals dying,
delta. We are using a logistics function because the rate of individuals
dying should decrease over time as we get better at treating.
----------------------------------------------------------------------------"""
def  delta(t, a, b, c, d, f):#mu, sig, phi):
    return abs(((t-a) * (t-b) * (t-c) * d * np.exp(-f * t)) + a*b*c*d) * 0.2 #gaussianG(t, mu, sig, phi)


"""PLOTTING DELTA FUNCTION-----------------------------------------------------
This displays a graph of the delta function. It should have part of a bell
curve logistic function.
----------------------------------------------------------------------------"""
def plotDelta(times, cases, deaths, a, b, c , d, f):
    fig, axsG = plt.subplots()

    axsG.set_title('Function of Delta')
    axsG.set_xlabel('Time (number of days)')
    axsG.set_ylabel('delta')
    axsG.plot(times, (deaths * 2)/(10 * cases))
    plt.show()
    plt.clf()
    
    fig, axsD = plt.subplots()

    axsD.set_title('Function of Delta')
    axsD.set_xlabel('Time (number of days)')
    axsD.set_ylabel('delta')
    axsD.plot(times, delta(times, a, b, c, d, f))
    plt.show()
    plt.clf()




"""PLOTTING BETA FUNCTION------------------------------------------------------
This displays a graph of the beta function. It should look like a upside
logistic function.
----------------------------------------------------------------------------"""
def plotBeta(times, L, k, t_0):
    fig, axsB = plt.subplots()

    axsB.set_title('Function of Beta')
    axsB.set_xlabel('Time (number of days)')
    axsB.set_ylabel('beta')

    axsB.plot(times, beta(times, L, k, t_0))

    plt.show()
    plt.clf()



"""R_0 FUNCTION USING NGM------------------------------------------------------
This function calculates R_0 using the NGM.
----------------------------------------------------------------------------"""
def calculateR_0_NGM(time, L_E, k_E, t_0_E, alpha):
    R_0 = beta(time, L_E, k_E, t_0_E) / alpha 
    return R_0
 
    
"""PLOTTING R_0 FUNCTION NGM---------------------------------------------------
This displays a graph of the R0 function as modeled by the NGM. 
----------------------------------------------------------------------------"""
def plotR_0_NGM(time, L_E, k_E, t_0_E, alpha):
    fig, axsR = plt.subplots()

    axsR.set_title('Function of R_0_NGM')
    axsR.set_xlabel('Time (number of days)')
    axsR.set_ylabel('R_0')

    axsR.plot(time, calculateR_0_NGM(time, L_E, k_E, t_0_E, alpha))

    plt.show()
    plt.clf()




"""PLOT THE BEST FIT OF INFECTED-----------------------------------------------
This function plots the data of cases and the best fit for our model.
----------------------------------------------------------------------------"""
def plotBestFitInfected(t, I, total_con, residual):
    fig, axR = plt.subplots()

    axR.plot(t, residual)
    axR.axhline(0, linestyle='--')

    axR.set_title('Residual of Infected')
    axR.set_ylabel('Residual')
    axR.set_xlabel('Time (days)')

    plt.show()
    plt.clf()

    f, axI = plt.subplots()

    axI.scatter(t, total_con, s=4, label='data')
    axI.plot(t, I, 'y', label='best fit')

    #ax.set_ylim(0, 1200000)
    #ax.set_ylim(0, 60953552)
    axI.set_xlabel('Time (days)')
    axI.set_ylabel('Population')
    axI.set_title('Infected Population')

    axI.yaxis.set_tick_params(length=0)
    axI.xaxis.set_tick_params(length=0)
    axI.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = axI.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        axI.spines[spine].set_visible(False)

    plt.show()
    plt.clf()


"""PLOT THE BEST FIT OF DEAD---------------------------------------------------
This function plots the data of dead and the best fit for our model.
----------------------------------------------------------------------------"""
def plotBestFitDied(t, D, total_deaths, residual):
    fig, axR = plt.subplots()
    #axR = fig.add_subplot(411, anchor=(0, 1))
    axR.plot(t, residual)
    axR.axhline(0, linestyle='--')

    axR.set_title('Residual of Died')
    axR.set_ylabel('Residual')
    axR.set_xlabel('Time (days)')

    plt.show()
    plt.clf()

    fig, axD = plt.subplots()

    axD.scatter(t, total_deaths, s=4, label='data')
    axD.plot(t, D, 'y', label='best fit')

    #ax.set_ylim(0, 1200000)
    #ax.set_ylim(0, 60953552)
    axD.set_xlabel('Time (days)')
    axD.set_ylabel('Population')
    axD.set_title('Population Died from COVID-19')

    axD.yaxis.set_tick_params(length=0)
    axD.xaxis.set_tick_params(length=0)
    axD.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = axD.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        axD.spines[spine].set_visible(False)

    plt.show()
    plt.clf()
    


"""PLOT THE BEST FIT OF DELTA---------------------------------------------------
This function plots the data of delta and the best fit for our model.
----------------------------------------------------------------------------""" 
def plotBestFitDelta(t, D, I, total_deaths, total_con, residual):
    fig, axR = plt.subplots()
    #axR = fig.add_subplot(411, anchor=(0, 1))
    axR.plot(t, residual)
    axR.axhline(0, linestyle='--')

    axR.set_title('Residual of Fatality Rate')
    axR.set_ylabel('Fatality')
    axR.set_xlabel('Time (days)')

    plt.show()
    plt.clf()

    fig, axD = plt.subplots()

    axD.scatter(t, total_deaths/total_con, s=4, label='data')
    axD.plot(t, D/I , 'y', label='best fit')

    #ax.set_ylim(0, 1200000)
    #ax.set_ylim(0, 60953552)
    axD.set_xlabel('Time (days)')
    axD.set_ylabel('Fatality Rate')
    axD.set_title('Fatality Rate of COVID-19')

    axD.yaxis.set_tick_params(length=0)
    axD.xaxis.set_tick_params(length=0)
    axD.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = axD.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        axD.spines[spine].set_visible(False)

    plt.show()
    plt.clf()
    
    
"""ERROR PROPOGATION----------------------------------------------------------
This function computes the variance of R_0.
Problem: NEED covariance matrix to comupte
----------------------------------------------------------------------------"""
def errorProp(t, L, k, t_0, alpha, coV):
    dR_0dL = beta(t, 1, k, t_0)/ alpha
    dR_0dk = -beta(t, L, k ,t_0) * (t - t_0) * np.power(np.e, k*(t-t_0))/(alpha * (1 + np.power(np.e, k*(t-t_0))))
    dR_0dt_0 = (k * np.power(np.e, k*(t-t_0)) * beta(t, L, k, t_0))/ (alpha * (1 + np.power(np.e, k*(t-t_0))))
    dR_0da = -1 * beta(t, L, k ,t_0)/alpha**2
    
    sigma_R_0 = (abs(dR_0dL)**2 * coV[1][2]**2 + abs(dR_0dk)**2 * coV[2][3]**2 
                 + abs(dR_0dt_0)**2 * coV[3][4]**2  + abs(dR_0da)**2 * coV[8][9]**2 
                 + 2 * dR_0dL * dR_0dk * 0.467 
                 + 2 * dR_0dL * dR_0dt_0 * -0.371
                 + 2 * dR_0dL * dR_0da * 0.345
                 + 2 * dR_0dk * dR_0dt_0 * 0.323
                 + 2 * dR_0dk * dR_0da * 0.304
                 + 2 * dR_0dt_0 * dR_0da * 0.995) 
    
    
    return sigma_R_0**0.5

def plotErrorProp(t, sigma_R_0):
    fig, axsR = plt.subplots()

    axsR.set_title('Error of R_0')
    axsR.set_xlabel('Time (number of days)')
    axsR.set_ylabel('Uncertinty of R_0')

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
    evenMoreTimes = np.linspace(0, 365*2-1, 365*2)
    withoutQuarentineTime = (0, 54-1, 54)
    
    # MAKING AN ARRAY
    times = np.linspace(0, len(total_con)-1, len(total_con)) # time (in days)
    mod = lmfit.Model(fitter)

    # Exposed Beta -- Rate of Transmission between SUSCEPTIBLE and EXPOSED
    mod.set_param_hint('L_E', min=0, max = 6) # L is the curve's maximum value
    mod.set_param_hint('k_E', min = 0, max = 0.15) # k is the logistic growth rate or steepness of the curve
    mod.set_param_hint('t_0_E', min=1, max=365) # t_0 is the value of the sigmoid's midpoint

    # Infected Beta -- Rate of Transmission between SUSCEPTIBLE and INFECTED
    mod.set_param_hint('L_I', min=0, max = 6) # L is the curve's maximum value
    mod.set_param_hint('k_I', min = 0, max = 0.5) # k is the logistic growth rate or steepness of the curve
    mod.set_param_hint('t_0_I', min=1, max=365) # t_0 is the value of the sigmoid's midpoint

    # Gamma -- Rate of Recovery (sick for 2 weeks up to 6)
    mod.set_param_hint('gamma', min= 1/(6*7), max = 1/12)#0.02, max=0.1)

    # Alpha -- Incubation period (5-6 days up to 14 days)
    mod.set_param_hint('alpha', min = 0.0714, max=0.2)#0.01667, max=0.1)

    # Delta -- Fatality Rate (2% fatality rate)
    #mod.set_param_hint('delta', min=0)#, max = 0.005)

    # Rho -- Rate people die from infection
    mod.set_param_hint('rho', min = 0)#, max = 0.5)
    
    # Delta -- Fatality Rate 
    # mod.set_param_hint('mu', min = 0)  # mu is the position of center peak
    # mod.set_param_hint('sig', min = 0)  # sig is the standard deviation 
    # mod.set_param_hint('phi', min = 0) # phi is the height of curve's peak
    mod.set_param_hint('a', min = 0)#, max = 120)
    mod.set_param_hint('b', min = 0)#, max = 120)
    mod.set_param_hint('c', min = 0)#, max = 120)
    mod.set_param_hint('d', min = 0)
    #mod.set_param_hint('e', min = 0)
    mod.set_param_hint('f', min = 0)
    
    #((t-f) * (t-a) * (t-b) * c * np.exp(-d * t)) + e
    
    # Puts Total number of case and deaths in 1 array to calculate best fit
    data = []
    data = np.concatenate([total_con, total_deaths, total_deaths/total_con])

    params = mod.make_params(verbose=True)
    result = mod.fit(data,
                     params,
                     t=times,
                      L_E = 0.194, #2,
                      k_E = 0.0045,
                      t_0_E = 56, #52.100165375262584,
                      L_I = 0.47,
                      k_I = 0.47,
                      t_0_I = 67,
                      gamma = 0.023, #0.074, 1/14
                      alpha = 0.0787, #0.192 0.167
                      #delta = 0.006,
                      rho = 0.63,
                      a = 1, 
                      b = 1, 
                      c = 1, 
                      d = 1, 
                      #e = 13,
                      f = 1)
                      # mu = 25, 
                      # sig = 10, 
                      # phi = 0.03)
    # print(delta(times,  result.best_values['a'], 
    #           result.best_values['b'], 
    #           result.best_values['c'],
    #           result.best_values['d'], 
    #           result.best_values['e'], 
    #           result.best_values['f']))
    
    plotDelta(times, total_con, total_deaths, 
              result.best_values['a'], 
              result.best_values['b'], 
              result.best_values['c'],
              result.best_values['d'], 
              #result.best_values['e'], 
              result.best_values['f'])
    
                      #     L_E = 0.204, #2,
                      # k_E = 0.004,
                      # t_0_E = 100, #52.100165375262584,
                      # L_I = 0.5,
                      # k_I = 0.5,
                      # t_0_I = 67,
                      # gamma = 0.023, #0.074, 1/14
                      # alpha = 0.0714, #0.192 0.167
                      # #delta = 0.006,
                      # rho = 1.4,
                      # a = 10, 
                      # b = 82, 
                      # c = 1, 
                      # d = 0.05, 
                      # e = 0.05)
                      # # mu = 25, 
                      # # sig = 10, 
                      # # phi = 0.03)
    
    #     L_E:    0.20385978 +/- 5.74910431 (2820.13%) (init = 0.9)
    # k_E:    0.00428831 +/- 0.14247720 (3322.46%) (init = 0.0005)
    # t_0_E:  1.00000290 +/- 2393.06286 (239305.59%) (init = 100)
    # L_I:    0.50404278 +/- 19.0767367 (3784.75%) (init = 2)
    # k_I:    0.49990505 +/- 0.24682398 (49.37%) (init = 0.0005)
    # t_0_I:  67.1660201 +/- 1.47262024 (2.19%) (init = 52.1)
    # gamma:  0.02380952 +/- 0.31280482 (1313.78%) (init = 0.07142857)
    # alpha:  0.07140001 +/- 2.27057959 (3180.08%) (init = 0.167)
    # delta:  0.00146591 +/- 40.8721304 (2788169.11%) (init = 0.006)
    # rho:    1.42430280 +/- 39795.3857 (2794025.66%) (init = 0.002)
                     #     L_E = 0.9, #2,
                     # k_E = 0.0005,
                     # t_0_E = 100, #52.100165375262584,
                     # L_I = 2,
                     # k_I = 0.0005,
                     # t_0_I = 52.1,
                     # gamma = 1/14, #0.074, 1/14
                     # alpha = 0.167, #0.192 0.167
                     # delta = 0.006,
                     # rho = 0.002) #, calc_covar= True)
          
    #Prints L's, k's, t_0's, gamma, alpha, rho
    print(result.fit_report())
    
 #   plotDelta(times, total_con, total_deaths)
    
    #lmfit.report_fit(result.params)

    plotBeta(moreTimes,
             result.best_values['L_I'],
             result.best_values['k_I'],
             result.best_values['t_0_I'])
    
#    print(gaussianG(times, result.best_values['mu'], result.best_values['sig'], result.best_values['phi']))


 #Prints R_0 NGM--------------------------------------------------
    print('Maximum R_0_NGM: ', 
              max(calculateR_0_NGM(times,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['alpha'])))

    print('Minimun R_0_NGM: ', 
              min(calculateR_0_NGM(times,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['alpha'])))

    print('Mean R_0_NGM: ', 
              (max(calculateR_0_NGM(times,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['alpha'])) + 
              min(calculateR_0_NGM(times,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['alpha'])))/2)

    plotR_0_NGM(times,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['alpha'])

    #Integrate SEIRD
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
                                               #result.best_values['delta'],
                                               result.best_values['rho'],
                                               result.best_values['a'],
                                               result.best_values['b'],
                                               result.best_values['c'],
                                               result.best_values['d'],
                                               #result.best_values['e'],
                                               result.best_values['f'])
                                               # result.best_values['mu'],
                                               # result.best_values['sig'],
                                               # result.best_values['phi'])

    
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
                                                #result.best_values['delta'],
                                                result.best_values['rho'],
                                                result.best_values['a'],
                                                result.best_values['b'],
                                                result.best_values['c'],
                                                result.best_values['d'],
                                               # result.best_values['e'],
                                               result.best_values['f'])
                                               # result.best_values['mu'],
                                               # result.best_values['sig'],
                                               # result.best_values['phi'])
    
    S_2y, E_2y, I_2y, R_2y, D_2y = integrateEquationsOverTime(deriv,
                                                evenMoreTimes,
                                                result.best_values['L_E'],
                                                result.best_values['k_E'],
                                                result.best_values['t_0_E'],
                                                result.best_values['L_I'],
                                                result.best_values['k_I'],
                                                result.best_values['t_0_I'],
                                                result.best_values['gamma'],
                                                result.best_values['alpha'],
                                                #result.best_values['delta'],
                                                result.best_values['rho'],
                                               # result.best_values['mu'],
                                               # result.best_values['sig'],
                                               # result.best_values['phi'])
                                                result.best_values['a'],
                                                result.best_values['b'],
                                                result.best_values['c'],
                                                result.best_values['d'],
                                               # result.best_values['e'],
                                               result.best_values['f'])

    S_q, E_q, I_q, R_q, D_q = integrateEquationsOverTime(deriv,
                                                evenMoreTimes,
                                                result.best_values['L_E'],
                                                result.best_values['k_E'],
                                                result.best_values['t_0_E'],
                                                result.best_values['L_I'],
                                                result.best_values['k_I'],
                                                result.best_values['t_0_I'],
                                                result.best_values['gamma'],
                                                result.best_values['alpha'],
                                                #result.best_values['delta'],
                                                result.best_values['rho'],
                                               # result.best_values['mu'],
                                               # result.best_values['sig'],
                                               # result.best_values['phi'])
                                                result.best_values['a'],
                                               result.best_values['b'],
                                               result.best_values['c'],
                                               result.best_values['d'],
                                             #  result.best_values['e'],
                                               result.best_values['f'])


    
    # Create an numpy array
    residualOfIandD = result.residual

   #Print Variance of R_0
    # print(errorProp(times, 
    #                 result.best_values['L_E'], 
    #                 result.best_values['k_E'],
    #                 result.best_values['t_0_E'], 
    #                 result.best_values['alpha'], 
    #                 covariance))    
    

    # Plot Residual and Best Fit
    plotBestFitInfected(times, I, total_con, residualOfIandD[:130])
    plotBestFitDied(times, D, total_deaths, residualOfIandD[130:260])
    plotBestFitDelta(times, D, I, total_deaths, total_con, residualOfIandD[260:])
    


    print('Population of the US:', N)
    print('Total Number of Deaths:', max(total_deaths))
    print('Total Number of Cases:', max(totalNumberOfCases()))
    print('Suspetible:', min(S[:len(times)]))
    print('Exposed:', max(E[:len(times)]))
    print('Infected:', max(I[:len(times)]))
    print('Recovered:', max(R[:len(times)])) # should be somewhere around 300,000
    print('Dead:',max(D[:len(times)])) # should equal total dead

    total = S+I+E+R+D

    print('Total:', min(total)) # should equal total population

    #PLOT SEIRD MODEL
    plotSEIRD(times, S, E, I, R, D, 'SEIRD Model of COVID-19')
    
    #PLOT SEIRD MODEL (A year out)
    plotSEIRD(moreTimes, S_y, E_y, I_y, R_y, D_y, 'Projected SEIRD Model of COVID-19 for a Year')
    
    #PLOT SEIRD MODEL (Two years out)
    plotSEIRD(evenMoreTimes, S_2y, E_2y, I_2y, R_2y, D_2y, 'Projected SEIRD Model of COVID-19 for Two Years')

    #PLOT WITHOUT QUARENTINE 
    #plotSEIRD(withoutQuarentineTime, S[:53], E[:53], I[:53], R[:53], D[:53], 'SEIRD Model without Quatentine')
