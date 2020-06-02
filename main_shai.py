## Shai added _NGM functions fot plotting and calculating R0

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian

import csv

import warnings
warnings.filterwarnings('ignore')


"""To-do List:
    - figure out how to get a better fit
    - model upto March 15th before shelter in place
    """


# Global variables
N = 0
y0 = ()





"""DERIVATIVE OF SIR-----------------------------------------------------------
This function calculates and define the derviatives for the data.
----------------------------------------------------------------------------"""
def deriv(y, t, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, delta, rho):
    S, E, I, R, D = y

    dSdt = -beta(t,L_E, k_E, t_0_E) * S * E / N  - beta(t,L_I,k_I,t_0_I) * S * I / N

    dEdt = beta(t,L_E, k_E, t_0_E) * S * E / N  + beta(t,L_I,k_I,t_0_I) * S * I / N - alpha * E

    dIdt = alpha * E - gamma * (1-delta)* I - rho * delta * I

    dRdt = gamma *(1 - delta)* I

    dDdt = rho * delta * I

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
def integrateEquationsOverTime(deriv, t, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha,  delta, rho):
    ret = odeint(deriv, y0, t, args=(L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha,  delta, rho))
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
def fitter(t, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, delta, rho):
    S, E, I, R, D = integrateEquationsOverTime(deriv, t, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, delta, rho)
    return np.concatenate([I, D])




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
    return phi * np.exp(-np.power(t - mu, 2) / (2 * np.power(sig, 2)))



"""GAMMA FUNCTION------------------------------------------------------
This function calcuates the rate in which individuals recover,
gamma. We are using a logistics function because the rate of individuals
recovering should increase over time as there are more infected
individuals.
----------------------------------------------------------------------------"""
def gamma(time, L_g, K, t_0g):
    gamma = L_g/(1+np.power(np.e, -K*(time-t_0g)))
    return gamma



"""DELTA FUNCTION--------------------------------------------------------------
This function calcuates the rate in which individuals dying,
delta. We are using a logistics function because the rate of individuals
dying should decrease over time as we get better at treating.
----------------------------------------------------------------------------"""
def delta(time, Ld, kd, t_0d):
    delta = Ld/(1+np.power(np.e, -kd*(time-t_0d)))
    return delta


"""PLOTTING DELTA FUNCTION-----------------------------------------------------
This displays a graph of the delta function. It should have part of a bell
curve logistic function.
----------------------------------------------------------------------------"""
def plotDelta(times, Ld, kd, t_0d):
    fig, axsG = plt.subplots()

    axsG.set_title('Function of Delta')
    axsG.set_xlabel('Time (number of days)')
    axsG.set_ylabel('delta')
    axsG.plot(times, delta(times, Ld, kd, t_0d))
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




"""PLOTTING GAMMA FUNCTION-----------------------------------------------------
This displays a graph of the gamma function. It should look like as logistic
function.
----------------------------------------------------------------------------"""
def plotGamma(times, L_g, K, t_0g):
    fig, axsG = plt.subplots()

    axsG.set_title('Function of Gamma')
    axsG.set_xlabel('Time (number of days)')
    axsG.set_ylabel('gamma')

    axsG.plot(times, gamma(times, L_g, K, t_0g))

    plt.show()
    plt.clf()




""" FUNCTION---------------------------------------------------------------ORIG
This function calculates R_0.
----------------------------------------------------------------------------"""
def calculateR_0(time, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma):
    R_0 = (beta(time, L_E, k_E, t_0_E)+beta(time, L_I, k_I, t_0_I))/gamma
    return R_0



""" FUNCTION----------------------------------------------------------------NGM!!!!!!!!
This function calculates R_0 using the NGM
----------------------------------------------------------------------------"""
def calculateR_0_NGM(time, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, delta, rho):
    w1 = gamma * (1-delta) + rho * delta
    R_0 = (beta(time, L_I, k_I, t_0_I) * alpha)/(w1* (alpha - beta(time, L_E, k_E, t_0_E)))
    return R_0




"""PLOTTING R_0 FUNCTION--------------------------------------------------ORIG
This displays a graph of the R0 function. It should have part of a bell
curve logistic function.
----------------------------------------------------------------------------"""
def plotR_0(time, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma):
    fig, axsR = plt.subplots()

    axsR.set_title('Function of R_0')
    axsR.set_xlabel('Time (number of days)')
    axsR.set_ylabel('R_0')

    axsR.plot(time, calculateR_0(time, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma))

    plt.show()
    plt.clf()


"""PLOTTING R_0 FUNCTION--------------------------------------------------NGM!!!!!!!!
This displays a graph of the R0 function as modeled by the NGM. 
----------------------------------------------------------------------------"""
def plotR_0_NGM(time, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, delta, rho):
    fig, axsR = plt.subplots()

    axsR.set_title('Function of R_0_NGM')
    axsR.set_xlabel('Time (number of days)')
    axsR.set_ylabel('R_0')

    axsR.plot(time, calculateR_0_NGM(time, L_E, k_E, t_0_E, L_I, k_I, t_0_I, gamma, alpha, delta, rho))

    plt.show()
    plt.clf()



"""PLOT THE BEST FIT OF INFECTED-----------------------------------------------
This function plots the data of cases and the best fit for our model.
----------------------------------------------------------------------------"""
def plotBestFitInfected(t, I, total_con, residualBitch):
    fig, axR = plt.subplots()

    axR.plot(t, residualBitch)
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
This function plots the data of cases and the best fit for our model.
----------------------------------------------------------------------------"""
def plotBestFitDied(t, D, total_deaths, residualBitch):
    fig, axR = plt.subplots()
    #axR = fig.add_subplot(411, anchor=(0, 1))
    axR.plot(t, residualBitch)
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



if __name__ == "__main__":
    total_con = totalNumberOfCases()
    total_deaths = totalNumberOfDeaths()

    # define constants
    N = totalPop()
    #D = 14 # infections lasts 2 week on average (14 days)
    y0 = (N - 2, 1, 1, 0, 0)  # initial conditions: one infected, rest susceptible
    moreTimes = np.linspace(0, 365-1, 365)
    evenMoreTimes = np.linspace(0, 365*2-1, 365*2)
    withoutQuarentineTime = (0, 54-1, 54)
    #alpha = 1/6 # incubation rate
    #gamma = 1/14 # rate of recovery

    # MAKING AN ARRAY
    times = np.linspace(0, len(total_con)-1, len(total_con)) # time (in days)

    mod = lmfit.Model(fitter)

    # Exposed Beta -- Rate of Transmission between SUSCEPTIBLE and EXPOSED
    mod.set_param_hint('L_E', min=0, max = 6)
    mod.set_param_hint('k_E', min = 0, max = 0.15)
    mod.set_param_hint('t_0_E', min=1, max=365)

    # Infected Beta -- Rate of Transmission between SUSCEPTIBLE and INFECTED
    mod.set_param_hint('L_I', min=0, max = 6)
    mod.set_param_hint('k_I', min = 0, max = 0.5)
    mod.set_param_hint('t_0_I', min=1, max=365)

    # Gamma -- Rate of Recovery (sick for 2 weeks up to 6)
    mod.set_param_hint('gamma', min= 1/(6*7), max = 1/12)#0.02, max=0.1)

    # Alpha -- Incubation period (5-6 days up to 14 days)
    mod.set_param_hint('alpha', min = 0.0714, max=0.2)#0.01667, max=0.1)

    # Delta -- Fatality Rate (2% fatality rate)
    mod.set_param_hint('delta', min=0) #, max = 0.005)

    # Rho -- Rate people die from infection
    mod.set_param_hint('rho', min = 0, max = 0.5)


    # Puts Total number of case and deaths in 1 array to calculate best fit
    data = []
    data = np.concatenate([total_con, total_deaths])

    params = mod.make_params(verbose=True)
    result = mod.fit(data,
                     params,
                     t=times,
                     L_E = 2,
                     k_E = 0.0005,
                     t_0_E = 52.100165375262584,
                     L_I = 2,
                     k_I = 0.0005,
                     t_0_I = 52.1,
                     gamma = 1/14, #0.074,
                     alpha = 0.167, #0.192
                     delta = 0.006,
                     rho = 0.002)

    print(result.covar)

    plotBeta(moreTimes,
             result.best_values['L_I'],
             result.best_values['k_I'],
             result.best_values['t_0_I'])


    """ORIGINAL R_0 CODE CALCULATIONS---------------------------------------------
    This until the next part is what you already had, I didn't want to change it 
    in case I broke something. 
    ----------------------------------------------------------------------------"""

    print('Maximum R_0: ', max(calculateR_0(moreTimes,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['L_I'],
              result.best_values['k_I'],
              result.best_values['t_0_I'],
              result.best_values['gamma'])))

    print('Minimun R_0: ', min(calculateR_0(moreTimes,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['L_I'],
              result.best_values['k_I'],
              result.best_values['t_0_I'],
              result.best_values['gamma'])))

    meanOfR_0 = (max(calculateR_0(moreTimes,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['L_I'],
              result.best_values['k_I'],
              result.best_values['t_0_I'],
              result.best_values['gamma'])) +  min(calculateR_0(moreTimes,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['L_I'],
              result.best_values['k_I'],
              result.best_values['t_0_I'],
              result.best_values['gamma'])))/2

    print('Mean R_0: ', meanOfR_0)

    plotR_0(moreTimes,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['L_I'],
              result.best_values['k_I'],
              result.best_values['t_0_I'],
              result.best_values['gamma'])

    #result.plot()

    # Prints L, k, t_0, gamma, alpha, delta
    print(result.best_values)



    """SHAI's R_0 CODE CALCULATIONS---------------------------------------------
    This part is the above but with the modified _NGM functions
    ----------------------------------------------------------------------------"""

    print('Maximum R_0_NGM: ', max(calculateR_0_NGM(moreTimes,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['L_I'],
              result.best_values['k_I'],
              result.best_values['t_0_I'],
              result.best_values['gamma'],
              result.best_values['alpha'],
              result.best_values['delta'],
              result.best_values['rho'])))

    print('Minimun R_0_NGM: ', min(calculateR_0_NGM(moreTimes,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['L_I'],
              result.best_values['k_I'],
              result.best_values['t_0_I'],
              result.best_values['gamma'],
              result.best_values['alpha'],
              result.best_values['delta'],
              result.best_values['rho'])))

    meanOfR_0_NGM = (max(calculateR_0_NGM(moreTimes,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['L_I'],
              result.best_values['k_I'],
              result.best_values['t_0_I'],
              result.best_values['gamma'],
              result.best_values['alpha'],
              result.best_values['delta'],
              result.best_values['rho'])) +  min(calculateR_0_NGM(moreTimes,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['L_I'],
              result.best_values['k_I'],
              result.best_values['t_0_I'],
              result.best_values['gamma'],
              result.best_values['alpha'],
              result.best_values['delta'],
              result.best_values['rho'])))/2

    print('Mean R_0_NGM: ', meanOfR_0_NGM)

    plotR_0_NGM(moreTimes,
              result.best_values['L_E'],
              result.best_values['k_E'],
              result.best_values['t_0_E'],
              result.best_values['L_I'],
              result.best_values['k_I'],
              result.best_values['t_0_I'],
              result.best_values['gamma']
              result.best_values['alpha'],
              result.best_values['delta'],
              result.best_values['rho'])

    #result.plot()

    # Prints L, k, t_0, gamma, alpha, delta
    print(result.best_values)

### ------- ------- ------- ------- ------- ------- ------- ------- ------- ------- ------- ------- ------- ------- ------- #####


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
                                               result.best_values['delta'],
                                               result.best_values['rho'])

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
                                               result.best_values['delta'],
                                               result.best_values['rho'])
    
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
                                               result.best_values['delta'],
                                               result.best_values['rho'])

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
                                               result.best_values['delta'],
                                               result.best_values['rho'])


    # Create an numpy array
    residualBitch = result.residual

    # Plot Residual and Best Fit
    plotBestFitInfected(times, I[:130], total_con, residualBitch[:130])
    plotBestFitDied(times, D[:130], total_deaths, residualBitch[130:])


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
