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
def plotSEIRD(t, S, E, I, R, D, title):
    f, ax = plt.subplots(1,1,figsize=(10,4))

    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'm', alpha=0.7, linewidth=2, label='Exposed')
    ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, D, 'r', alpha=0.7, linewidth=2, label='Dead')
    ax.plot(t, S+E+I+R+D, 'c--', alpha=0.7, linewidth=2, label='Total')

    ax.set_ylim(1, N)
    ax.set_yscale('log')
    #ax.set_ylim(0, 2500000)
    #ax.set_ylim(0, 6045189)
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Population (Number of Individuals)')
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
    
    ax.set_title('Beta over Time')
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
def calculateR_0_NGM(time, L_E, k_E, t_0_E, alpha):
    R_0 = beta(time, L_E, k_E, t_0_E) / alpha 
    return R_0
 
    
"""PLOTTING R_0 FUNCTION NGM---------------------------------------------------
This displays a graph of the R0 function as modeled by the NGM. 
----------------------------------------------------------------------------"""
def plotR_0_NGM(time, L_E, k_E, t_0_E, alpha):
    fig, axsR = plt.subplots()

    axsR.set_title('Reproduction Number (R\u2080) Over Time')
    axsR.set_xlabel('Time (Days)')
    axsR.set_ylabel('Reproduction Number (R\u2080)')

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

    axR.set_title('Residual of Infected Population')
    axR.set_ylabel('Residual')
    axR.set_xlabel('Time (Days)')

    plt.show()
    plt.clf()

    f, axI = plt.subplots()

    axI.scatter(t, total_con, s=4, label='ata')
    axI.plot(t, I, 'y', label='best fit')

    #ax.set_ylim(0, 1200000)
    #ax.set_ylim(0, 60953552)
    axI.set_xlabel('Time (Days)')
    axI.set_ylabel('Population (Number of Infected Individuals)')
    axI.set_title('Population of Infected Individuals')

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

    axR.set_title('Residual of Dead Population')
    axR.set_ylabel('Residual')
    axR.set_xlabel('Time (Days)')

    plt.show()
    plt.clf()

    fig, axD = plt.subplots()

    axD.scatter(t, total_deaths, s=4, label='Data')
    axD.plot(t, D, 'y', label='Best Fit')

    #ax.set_ylim(0, 1200000)
    #ax.set_ylim(0, 60953552)
    axD.set_xlabel('Time (Days)')
    axD.set_ylabel('Population (Number of Dead Individuals)')
    axD.set_title('Population of Dead Individuals' )

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

    axR.set_title('Residual of the Fatality Rate')
    axR.set_ylabel('Residual')
    axR.set_xlabel('Time (Days)')

    plt.show()
    plt.clf()

    fig, axD = plt.subplots()

    axD.scatter(t, total_deaths/total_con, s=4, label='data')
    axD.plot(t, D/I , 'y', label='best fit')

    #ax.set_ylim(0, 1200000)
    #ax.set_ylim(0, 60953552)
    axD.set_xlabel('Time (Days)')
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
This function computes the variance of R_0. This is a manual function.
----------------------------------------------------------------------------"""
def errorProp(t, L, k, t_0, alpha):
    dR_0dL = beta(t, 1, k, t_0)/ alpha
    dR_0dk = -beta(t, L, k ,t_0) * (t - t_0) * np.power(np.e, k*(t-t_0))/(alpha * (1 + np.power(np.e, k*(t-t_0))))
    dR_0dt_0 = (k * np.power(np.e, k*(t-t_0)) * beta(t, L, k, t_0))/ (alpha * (1 + np.power(np.e, k*(t-t_0))))
    dR_0da = -1 * beta(t, L, k ,t_0)/alpha**2
    
    sigma_R_0 = (abs(dR_0dL)**2 * 35.7762578**2 + abs(dR_0dk)**2 * 0.89701212**2 
                 + abs(dR_0dt_0)**2 * 21785.9891**2  + abs(dR_0da)**2 * 12.9266165**2 
                 + 2 * dR_0dL * dR_0dk * -0.999 
                 + 2 * dR_0dL * dR_0dt_0 * -0.978
                 + 2 * dR_0dL * dR_0da * 0.995
                 + 2 * dR_0dk * dR_0dt_0 * 0.969
                 + 2 * dR_0dk * dR_0da * -0.998
                 + 2 * dR_0dt_0 * dR_0da * -0.953) 
    
    
    return sigma_R_0**0.5


"""ERROR PROPOGATION PLOT------------------------------------------------------
This function computes the variance of R_0. This is a manual function.
----------------------------------------------------------------------------"""
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
                                        result.best_values['L_E'],
                                        result.best_values['k_E'],
                                        result.best_values['t_0_E'])))
        
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
                                                    result.best_values['alpha'])))

    print('Minimun R_0_NGM: ', min(calculateR_0_NGM(times,
                                                    result.best_values['L_E'],
                                                    result.best_values['k_E'],
                                                    result.best_values['t_0_E'],
                                                    result.best_values['alpha'])))

    print('Mean R_0_NGM: ', (max(calculateR_0_NGM(times,
                                                  result.best_values['L_E'],
                                                  result.best_values['k_E'],
                                                  result.best_values['t_0_E'],
                                                  result.best_values['alpha'])) + 
                             min(calculateR_0_NGM(times,
                                                  result.best_values['L_E'],
                                                  result.best_values['k_E'],
                                                  result.best_values['t_0_E'],
                                                  result.best_values['alpha'])))/2)
    
    
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
                result.best_values['alpha'])
    
    #Plot the uncertainty of R_0  
    plotErrorProp(times, errorProp(times, 
                    result.best_values['L_E'], 
                    result.best_values['k_E'],
                    result.best_values['t_0_E'], 
                    result.best_values['alpha']))
    
    


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

    #Plot SEIRD model -- Based on data
    plotSEIRD(times, S, E, I, R, D, 'SEIRD Model of COVID-19')
    
    #Plot SEIRD model -- without quarentine (using the max. beta and max. delta from delta)
    plotSEIRD(times, S_wo, E_wo, I_wo, R_wo, D_wo, 'SEIRD Model of COVID-19 without Quatentine')
    
    #Plot SEIRD model -- A year out
    plotSEIRD(moreTimes, S_y, E_y, I_y, R_y, D_y, 'Projected SEIRD Model of COVID-19')
    
    #Plot SEIRD model -- without quarentine (using the max. beta and max. delta from data)
    plotSEIRD(moreTimes, S_woq, E_woq, I_woq, R_woq, D_woq, 'Projected SEIRD Model of COVID-19 without Quatentine')
