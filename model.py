import time as timer
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import root
from scipy import constants as con
from matplotlib import colormaps as cm

auger_assisted_trapping = False
scaling = 1e12 #um3 / cm3

# physical models

T = 298 # K
q = 1*con.e # C

# Perovskite material parameters

m_eff_n = 0.2 # unitless (effective electron mass)
m_eff_p = 0.2 # unitless (effective hole mass)
einf = 5*con.epsilon_0 
eps = 33.5*con.epsilon_0
wv = (0.0165*con.e/con.hbar) #  s-1 (phonon frequency)
a0 = 6.3 * 1e-10 # m (lattice constant)

Nc = 6.98e18 #cm^-3     effective densities of state at CB/VB for MAPI, from DOI:10.1021/acs.jpcc.6b10914
Nv = 2.49e18 #cm^-3


def construct_param_set(i, input_df, param_dict, param_input):

    param_values = {} # parameter, value pairs to fit measurement i

    for key in param_dict[input_df.loc[i, 'sample']].keys():

        if param_dict[input_df.loc[i, 'sample']][key][5] == 'constant':
            param_values[key] = param_dict[input_df.loc[i, 'sample']][key][1]
        
        elif param_dict[input_df.loc[i, 'sample']][key][5] == 'vary indv':
            param_values[key] = param_input[input_df.loc[i, 'sample']][key]

        elif param_dict[input_df.loc[i, 'sample']][key][5] == 'vary together':
            param_values[key] = param_input[input_df.loc[0, 'sample']][key]
    
    return param_values


def import_params(i, input_df, param_dict, param_input):

    param_values = construct_param_set(i, input_df, param_dict, param_input)
    Eg = input_df.loc[i, 'bandgap (eV)']

    # Unpack, calculate and/or rescale all params to desired units. (rescale everything to be um-3 instead of cm-3)

    Ecb = Eg *con.e # J   (bandgap)
    Evb = 0 *con.e # J
    
    Ef = param_values['p_Ef'] *con.e # J
    n0 = ( 2 * ( (m_eff_n*con.m_e*con.k*T) / (2*con.pi*(con.hbar**2)) )**(3/2) ) * np.exp( -(Ecb - Ef) / (con.k*T))*1e-6 *scaling**-1 # um-3
    p0 = ( 2 * ( (m_eff_p*con.m_e*con.k*T) / (2*con.pi*(con.hbar**2)) )**(3/2) ) * np.exp( -(Ef - Evb) / (con.k*T))*1e-6 *scaling**-1 # um-3

    krad = 10**param_values['p_krad'] *scaling          # um3 s-1 (internal radiative recombination coefficient)
    Pr = param_values['p_Pr']                          # unitless (probability of reabsorption)
    Can = 10**param_values['p_Ca'] *scaling**2          # um6 s-1 (Auger coefficient for nnp)
    Cap = 10**param_values['p_Ca'] *scaling**2          # um6 s-1 (Auger coefficient for ppn)

    N_1 = 10**param_values['p_N_1'] *scaling**-1        # um-3 (density of trap 1)
    depth_trap_1 = param_values['p_depth_trap_1']       # eV (energy level of trap 1, distance from VB)
    beta_n_1 = 10**param_values['p_bn_1'] *scaling      # um3 s-1 (phonon-assisted capture coefficient for n, trap 1)
    beta_p_1 =10**param_values['p_bp_1'] *scaling       # um3 s-1 (phonon-assisted capture coefficient for p, trap 1)
    n1_1 = Nc*np.exp((depth_trap_1 - (Ecb/con.e))/(0.0257)) *scaling**-1 # um-3 
    p1_1 = Nv*np.exp(((Evb/con.e) - depth_trap_1)/(0.0257)) *scaling**-1 # um-3
    e_n_1 = beta_n_1*n1_1               # s-1 (emission coefficient for n in trap 1)
    e_p_1 = beta_p_1*p1_1               # s-1 (emission coefficient for p in trap 1)

    N_2 = 10**param_values['p_N_2'] *scaling**-1           # um-3 (density of trap 2)
    depth_trap_2 = param_values['p_depth_trap_2']      # eV (energy level of trap 2, distance from VB)
    beta_n_2 = 10**param_values['p_bn_2'] *scaling      # um3 s-1 (phonon-assisted capture coefficient for n, trap 2)
    beta_p_2 =10**param_values['p_bp_2'] *scaling       # um3 s-1 (phonon-assisted capture coefficient for p, trap 2)
    n1_2 = Nc*np.exp((depth_trap_2 - (Ecb/con.e))/(0.0257)) *scaling**-1 # um-3 
    p1_2 = Nv*np.exp(((Evb/con.e) - depth_trap_2)/(0.0257)) *scaling**-1 # um-3
    e_n_2 = beta_n_2*n1_2               # s-1 (emission coefficient for n in trap 2)
    e_p_2 = beta_p_2*p1_2               # s-1 (emission coefficient for p in trap 2)

    if auger_assisted_trapping == True:
        T1_1, T2_1, T3_1, T4_1 = k_cap_auger(depth_trap_1, Eg)
        T1_1 = T1_1 *scaling**2         # um6 s-1 (Auger coefficient for n-assisted n trapping)
        T2_1 = T2_1 *scaling**2         # um6 s-1 (Auger coefficient for p-assisted n trapping)
        T3_1 = T3_1 *scaling**2         # um6 s-1 (Auger coefficient for n-assisted p trapping)
        T4_1 = T4_1 *scaling**2         # um6 s-1 (Auger coefficient for p-assisted p trapping)
        T1_2, T2_2, T3_2, T4_2  = k_cap_auger(1.25-depth_trap_2, Eg)
        # need to double check the descriptions for these
        T1_2 = T1_2 *scaling**2         # um6 s-1 (Auger coefficient for n-assisted n trapping)
        T2_2 = T2_2 *scaling**2         # um6 s-1 (Auger coefficient for p-assisted n trapping)
        T3_2 = T3_2 *scaling**2         # um6 s-1 (Auger coefficient for n-assisted p trapping)
        T4_2 = T4_2 *scaling**2         # um6 s-1 (Auger coefficient for p-assisted p trapping)
    else:
        T1_1, T2_1, T3_1, T4_1 = [float(0), float(0), float(0), float(0)]
        T1_2, T2_2, T3_2, T4_2 = [float(0), float(0), float(0), float(0)]
    
    A_ct = param_values['p_A_ct']

    return  np.array([n0, p0, krad, Pr, Can, Cap, 
            N_1, depth_trap_1, beta_n_1, beta_p_1, e_n_1, e_p_1, n1_1, p1_1, 
            N_2, depth_trap_2, beta_n_2, beta_p_2, e_n_2, e_p_2, n1_2, p1_2, 
            T1_1, T2_1, T3_1, T4_1, T1_2, T2_2, T3_2, T4_2, A_ct])


# Function to calculate Auger-assisted trapping coefficients T1, T2, T3, T4. ONLY for n traps for now
# from Staub et al. 2018 https://pubs.acs.org/doi/10.1021/acsomega.8b00962
def k_cap_auger(Etrap, Eg): # ONLY for n traps for now

    Ecb = Eg *con.e # J   (bandgap)
    Evb = 0 *con.e # J

    if Etrap < 0.625:
        print("Not a hole trap - trap-assisted auger is not supported")
    deltaE_n = Ecb - (Etrap*con.e) # J ('depth' of defect from CB)
    #deltaE_p = (deltaE*con.e) - Evb # J ('depth' of defect from VB)

    o_n = m_eff_p/m_eff_n
    oL_n = 1/o_n
    #o_p = m_eff_n/m_eff_p
    #oL_p = 1/o_p

    w_n = (Ecb-deltaE_n)/(Ecb-(1+oL_n)*deltaE_n)
    N1 = 1/(16*deltaE_n**3) # J-3
    N2 = (oL_n**(9/2))/(((1+oL_n)**4)*(deltaE_n**3))
    N3 = (deltaE_n**(5/2))/((Ecb-deltaE_n)**(3/2)*(Ecb**4))
    N4 = (oL_n**(9/2)*deltaE_n**(5/2))/((Ecb-deltaE_n)**(3/2)*(deltaE_n*oL_n+Ecb-deltaE_n)**4)

    d11 = 13
    d21 = 0.5*(33-15*oL_n)
    d31 = - 2 - 6*o_n + (1 - deltaE_n/Ecb)*(20*(1+o_n) - 3/2)
    d41 = - 8 - 40.5 * w_n

    d12 = -260
    d22 = 0.5*(33-15*oL_n)
    d32 = 30*(1+o_n)*(5*o_n-3) + 15/((1-deltaE_n/Ecb)**2) - 1200*(1-deltaE_n/Ecb)*(1+o_n)**2 + 1680*(1-deltaE_n/Ecb)**2
    d42 = 40*(11 + 120*w_n + 168*w_n**2) + 64/w_n - (27/(w_n**2))

    b1 = 2*deltaE_n/(con.k*T)
    b2 = (1+oL_n)*deltaE_n/(con.k*T)
    b3 = Ecb/(con.k*T)
    b4 = (Ecb-(1-oL_n)*deltaE_n)/(con.k*T)

    T1 = ( 8*(q**4)*(con.hbar**3)/(m_eff_n*con.m_e*eps)**2 ) * N1*(1+(d11/b1)+(d12/(4*b1**2))) 
    T2 = ( 8*(q**4)*(con.hbar**3)/(m_eff_n*con.m_e*eps)**2 ) * N2*(1+(d21/b2)+(d22/(4*b2**2)))
    T3 = ( 8*(q**4)*(con.hbar**3)/(m_eff_p*con.m_e*eps)**2 ) * N3*(1+(d31/b3)+(d32/(4*b3**2)))
    T4 = ( 8*(q**4)*(con.hbar**3)/(m_eff_p*con.m_e*eps)**2 ) * N4*(1+(d41/b4)+(d42/(4*b4**2)))
    # C4 J3 kg-2 C-2 V2 m2 J-3 = m6 s-4 ? need to check these units

    return T1*1e12, T2*1e12, T3*1e12, T4*1e12

# ODEs adapted from Yuan et al. 2024 https://www.nature.com/articles/s41563-023-01771-2
def calc_TRPL(time, i, input_df, param_dict, param_input, exc_density, show_carrier_densities = False):

    # Define the ODEs for the changes in electron, hole, and trapped electron density
    
    param_list = import_params(i, input_df, param_dict, param_input)
    n0, p0, krad, Pr, Can, Cap, N_1, depth_trap_1, beta_n_1, beta_p_1, e_n_1, e_p_1, n1_1, p1_1, N_2, depth_trap_2, beta_n_2, beta_p_2, e_n_2, e_p_2, n1_2, p1_2, T1_1, T2_1, T3_1, T4_1, T1_2, T2_2, T3_2, T4_2, A_ct = param_list
    
    exc_density_scaled = exc_density *scaling**-1

    '''#figure out dark steady-state occupation of traps (not currently in use)
    n_t1_0_a = -beta_n_1
    n_t1_0_b = n0 * beta_n_1 + p0 * beta_p_1 + e_n_1 + e_p_1 + beta_n_1 * N_1
    n_t1_0_c = -N_1*(n0 * beta_n_1 + e_p_1)
    n_t1_0 = (-n_t1_0_b + np.sqrt(n_t1_0_b**2-4*n_t1_0_a*n_t1_0_c))/(2*n_t1_0_a)

    p_t2_0_a = -beta_p_2
    p_t2_0_b = p0 * beta_p_2 + n0 * beta_n_2 + e_p_2 + e_n_2 + beta_p_2 * N_2
    p_t2_0_c = -N_2*(p0 * beta_p_2 + e_n_2)
    p_t2_0 = (-p_t2_0_b + np.sqrt(p_t2_0_b**2-4*p_t2_0_a*p_t2_0_c))/(2*p_t2_0_a)
    #print('n0, p0', n0, p0)
    #print('n_t1_0, p_t2_0:', n_t1_0, p_t2_0)'''
    
    # define coupled ODEs
    def eq_time_resolved(c, t, n_pulse): 

        n, p, n_t1, p_t2 = c # n and p are the TOTAL carrier densities, including n0/p0
    
        # generation pulse
        pulse_fwhm = 2e-9
        sig = pulse_fwhm / (2 * np.sqrt(2 * np.log(2)))
        gauss_dat = 1 / (sig * np.sqrt(2 * np.pi)) * np.exp(-(1 / 2) * ((t) / sig) ** 2)
        
        # Auger and radiative recombination
        R_aug = - Can*n*n*p - Cap*n*p*p
        R_rad = - (1-Pr)*krad*(n*p)

        # for dn/dt
        R_n_srh_ntrap =  - beta_n_1*n*(N_1-n_t1) + e_n_1*n_t1
        R_n_srh_ptrap =  - beta_n_2*n*p_t2 + e_n_2*(N_2-p_t2)
        if auger_assisted_trapping == True:
            R_n_aug_ntrap = - T1_1 * ( (n*n)*(N_1-n_t1) - n1_1*n*n_t1 ) - T2_1 * ( (n*p)*(N_1-n_t1) - n1_1*p*n_t1 )  
            R_n_aug_ptrap = - T3_2 * ( (n*p)*p_t2 - p1_2*n*(N_2-p_t2) ) - T4_2 * ( (p*p)*p_t2 - p1_2*p*(N_2-p_t2) )
        
            dn_dt = R_aug + R_rad + R_n_srh_ntrap +  R_n_srh_ptrap + R_n_aug_ntrap + R_n_aug_ptrap 
        else:
            dn_dt = R_aug + R_rad + R_n_srh_ntrap +  R_n_srh_ptrap
        
        # for dp/dt
        R_p_srh_ntrap =  - beta_p_1*p*n_t1 + e_p_1*(N_1-n_t1)
        R_p_srh_ptrap =  - beta_p_2*p*(N_2-p_t2) + e_p_2*p_t2
        if auger_assisted_trapping == True:
            R_p_aug_ntrap = - T3_1 * ( (n*p)*n_t1 - p1_1*n*(N_1-n_t1) ) - T4_1 * ( (p*p)*n_t1 - p1_1*p*(N_1-n_t1) )
            R_p_aug_ptrap = - T1_2 * ( (n*n)*(N_2-p_t2) - n1_2*n*p_t2 ) - T2_2 * ( (n*p)*(N_2-p_t2) - n1_2*p*p_t2 )

            dp_dt = R_aug + R_rad + R_p_srh_ntrap + R_p_srh_ptrap + R_p_aug_ntrap + R_p_aug_ptrap
        else:
            dp_dt = R_aug + R_rad + R_p_srh_ntrap + R_p_srh_ptrap

        # for dn_t1/dt
        R_nt1_srh_ntrap =  beta_n_1*n*(N_1-n_t1) - e_n_1*n_t1 - beta_p_1*p*n_t1 + e_p_1*(N_1-n_t1)
        if auger_assisted_trapping == True:
            R_nt1_aug_ntrap = ( T1_1 * ( (n*n)*(N_1-n_t1) - n1_1*n*n_t1 ) + T2_1 * ( (n*p)*(N_1-n_t1) - n1_1*p*n_t1 )  
                                - T3_1 * ( (n*p)*n_t1 - p1_1*n*(N_1-n_t1) ) - T4_1 * ( (p*p)*n_t1 - p1_1*p*(N_1-n_t1) )) 

            dn_t1_dt = R_nt1_srh_ntrap + R_nt1_aug_ntrap
        else:
            dn_t1_dt = R_nt1_srh_ntrap
    
        # for dp_t2/dt
        R_pt2_srh_ptrap =  beta_p_2*p*(N_2-p_t2) - e_p_2*p_t2 - beta_n_2*n*p_t2 + e_n_2*(N_2-p_t2)
        if auger_assisted_trapping == True:
            R_pt2_aug_ptrap = ( T4_2 * ( (p*p)*(N_2-p_t2) - p1_2*p*p_t2 ) + T3_2 * ( (n*p)*(N_2-p_t2) - p1_2*n*p_t2 )  
                            - T2_2 * ( (n*p)*p_t2 - n1_2*p*(N_2-p_t2) ) - T1_1 * ( (n*n)*p_t2 - n1_2*n*(N_2-p_t2) )) 

            dp_t2_dt = R_pt2_srh_ptrap + R_pt2_aug_ptrap
        else:
            dp_t2_dt = R_pt2_srh_ptrap
        
        # needs to be phrased as below for jit to work (i think)
        func =    np.array([(gauss_dat*n_pulse + dn_dt),    # dn/dt
                            (gauss_dat*n_pulse + dp_dt),    # dp/dt
                            dn_t1_dt,
                            dp_t2_dt])

        return func

    # Solve the ODEs for electron, hole, trapped electron density.

    x = 50 # factor to help ode integration, increase if ode problems
    time_expanded = np.linspace(time[0]*1e-9, time[-1]*1e-9, len(time)*x)

    carriers_t0 = np.array([n0, p0, 0, 0]) # initial densities (if all traps empty during decay)

    k=0 # counter for integration to be repeated with various starting densities until start (before pulse) = end
    while k < 6:

        # solve ODEs over time range
        result = odeint(eq_time_resolved, carriers_t0, time_expanded, args=(exc_density_scaled,))
    
        # resulting densities are filtered for negative values (comes from numerical errors when densities very small)
        result_n = np.where(result[:, 0][0::x]>0, result[:, 0][0::x], result[:, 0][0]) # total carrier density (with n0)
        result_p = np.where(result[:, 1][0::x]>0, result[:, 1][0::x], result[:, 1][0]) # total carrier density (with p0)
        result_nt1 = np.where(result[:, 2][0::x]>0, result[:, 2][0::x], result[:, 2][0]) # electrons in trap 1
        result_pt2 = np.where(result[:, 3][0::x]>0, result[:, 3][0::x], result[:, 3][0]) # holes in trap 2
        
        # term describing magnitude in difference between start and end values (with some averaging for numerical errors)
        diffs = np.array( [ np.max( [ np.nanmean(result_n[-50:-1])/result_n[0], result_n[0]/np.nanmean(result_n[-50:-1]) ] ),
                            np.max( [ np.nanmean(result_p[-50:-1])/result_p[0], result_p[0]/np.nanmean(result_p[-50:-1])] ) ] )
        
        if (diffs < 2).all(): # stopping criterion
            break
        else:
            n_in_new = np.nanmean(result_n[-50:-1])
            p_in_new = np.nanmean(result_p[-50:-1])
            nt1_in_new = np.nanmean(result_nt1[-50:-1])
            pt2_in_new = np.nanmean(result_pt2[-50:-1])
            carriers_t0 = [n_in_new, p_in_new, nt1_in_new, pt2_in_new] # define new starting point and solve ODEs again
            k+=1
            #if k==6: # stop after 5 tries, normally this is enough
                #print('trpl did not converge, remaining difference is ', diffs)
            continue

    if show_carrier_densities == True:
        plt.show()
        plt.plot(time, result_n*scaling, label='n(t)')
        plt.plot(time, result_p*scaling, label='p(t)')
        plt.plot(time, result_nt1*scaling, label='n_trap_1(t)')
        plt.plot(time, result_pt2*scaling, label='p_trap_2(t)')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(1, time[-1])
        plt.legend()
        plt.show()
        plt.clf()

    # Calculate normalized PL from the determined charge carrier densities.
    PL_norm = ( (result_n*result_p)/np.nanmax(result_n*result_p) ) * (1-A_ct)

    return PL_norm, result*scaling


def PLQE_function(generation_rates, i, input_df, param_dict, param_input, print_carrier_densities = False, print_fitting_info = False):
    
    param_list = import_params(i, input_df, param_dict, param_input)
    n0, p0, krad, Pr, Can, Cap, N_1, depth_trap_1, beta_n_1, beta_p_1, e_n_1, e_p_1, n1_1, p1_1, N_2, depth_trap_2, beta_n_2, beta_p_2, e_n_2, e_p_2, n1_2, p1_2, T1_1, T2_1, T3_1, T4_1, T1_2, T2_2, T3_2, T4_2, A_ct = param_list

    # Define the system of equations for steady state
    def steady_state_exp(carriers, G_ext):

        n = carriers[0] # um-3 -  this is the TOTAL free carrier density (WITH n0/p0)
        p = carriers[1] # um-3

        # Calculate steady-state trapped carrier density for given n,p
        if auger_assisted_trapping == True:
            n_t1 = N_1 * ((n * beta_n_1 + e_p_1 + T1_1*n**2 + T2_1*n*p + T3_1*p1_1*n + T4_1*p1_1*p ) / (n * beta_n_1 + p * beta_p_1 + e_n_1 + e_p_1 + T1_1*n**2 + T1_1*n*n1_1 + T2_1*n*p + T2_1*n1_1*p + T3_1*p1_1*n + T3_1*p*n + T4_1*p1_1*p + T4_1*p**2))
            p_t2 = N_2 * ((p * beta_p_2 + e_n_2 + T4_2*p**2 + T3_2*n*p + T2_2*n1_2*p + T1_2*n1_2*n ) / (p * beta_p_2 + n * beta_n_2 + e_p_2 + e_n_2 + T4_2*p**2 + T4_2*p*p1_2 + T3_2*n*p + T3_2*p1_2*n + T2_2*n1_2*p + T2_2*p*n + T1_2*n1_2*n + T1_1*n**2))
        else:
            n_t1 = N_1 * ((n * beta_n_1 + e_p_1 ) / (n * beta_n_1 + p * beta_p_1 + e_n_1 + e_p_1 ))
            p_t2 = N_2 * ((p * beta_p_2 + e_n_2 ) / (p * beta_p_2 + n * beta_n_2 + e_p_2 + e_n_2))

        #Calculate recombination rates for given n,p (all um-3 s-1)
        R_aug = - Can*n*n*p - Cap*n*p*p 
        R_rad = - (1-Pr)*krad*(n*p)

        Rsrh_t1 = - (N_1 * beta_n_1 * beta_p_1 * n * p) / (n * beta_n_1 + p * beta_p_1 + e_n_1 + e_p_1)
        Rsrh_t2 = - (N_2 * beta_p_2 * beta_n_2 * n * p) / (n * beta_n_2 + p * beta_p_2 + e_n_2 + e_p_2)
        
        if auger_assisted_trapping == True:
            # need to double check that these are correct
            Rtaug_t1 =  T1_1 * ( (n*n)*(N_1-n_t1) - n1_1*n*n_t1 ) + T2_1 * ( (n*p)*(N_1-n_t1) - n1_1*p*n_t1 ) 
            Rtaug_t2 =  T4_2 * ( (p*p)*(N_2-p_t2) - p1_2*p*p_t2 ) + T3_2 * ( (n*p)*(N_2-p_t2) - p1_2*n*p_t2 ) 

            # first function to minimize: dn/dt (equivalent to dp/dt, can pick either)
            dndt = G_ext + R_aug + R_rad + Rsrh_t1 + Rsrh_t2 + Rtaug_t1 + Rtaug_t2
        else:
            dndt = G_ext + R_aug + R_rad + Rsrh_t1 + Rsrh_t2

        # second function to minimize: n(total) - n0 = p(total) - p0 (this enforces the Fermi level position)
        charge_balance = (n - n0 + n_t1) - (p - p0 + p_t2)
        
        return dndt, charge_balance

    # Solve for total carrier densities n, p for each fluence by finding the root of the function above. 
    n, p, n_t1, p_t2 = [[],[],[],[]]

    generation_rates_scaled = generation_rates *scaling**-1 # photons s-1 um-3

    for G in generation_rates_scaled:
        
        if np.isnan(G):
            n.append(np.nan)
            p.append(np.nan)
            n_t1.append(np.nan)
            p_t2.append(np.nan)
            continue

        # initial guess for n,p - may want to adjust this if solver is struggling
        carriers_ss_initialguess = np.array([(1e14 *scaling**-1), (1e14 *scaling**-1)])

        # plot to diagnose issues if solver is struggling
        if print_fitting_info == True:
            print('initial:', steady_state_exp(carriers_ss_initialguess, G))
            n_ex = np.linspace(10 *scaling**-1, 18 *scaling**-1, num=100)
            p_ex = np.linspace(10 *scaling**-1, 18 *scaling**-1, num=100)      
            X, Y = np.meshgrid(n_ex, p_ex)

            Z = np.abs(steady_state_exp([X, Y], G)[0])
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=cm['plasma'])
            ax.view_init(elev=10., azim=120)
            plt.show()

            Z = np.abs(steady_state_exp([X, Y], G)[1])
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=cm['plasma'])
            ax.view_init(elev=10., azim=120)
            plt.show()

        # solve for steady-state n,p at fluence G using L-M algorithm
        
        carriers_i = root(steady_state_exp, carriers_ss_initialguess, args=(G), method='lm', options={'maxiter': 200000})
        
        error_square = (steady_state_exp(carriers_i.x, G)[0])**2 + (steady_state_exp(carriers_i.x, G)[1])**2
            
        if auger_assisted_trapping == True:
            n_t1_out = N_1 * ((carriers_i.x[0] * beta_n_1 + e_p_1 + T1_1*carriers_i.x[0]**2 + T2_1*carriers_i.x[0]*carriers_i.x[1] + T3_1*p1_1*carriers_i.x[0] + T4_1*p1_1*carriers_i.x[1] ) / (carriers_i.x[0] * beta_n_1 + carriers_i.x[1] * beta_p_1 + e_n_1 + e_p_1 + T1_1*carriers_i.x[0]**2 + T1_1*carriers_i.x[0]*n1_1 + T2_1*carriers_i.x[0]*carriers_i.x[1] + T2_1*n1_1*carriers_i.x[1] + T3_1*p1_1*carriers_i.x[0] + T3_1*carriers_i.x[1]*carriers_i.x[0] + T4_1*p1_1*carriers_i.x[1] + T4_1*carriers_i.x[1]**2))
            p_t2_out = N_2 * ((carriers_i.x[1] * beta_p_2 + e_n_2 + T4_2*carriers_i.x[1]**2 + T3_2*carriers_i.x[0]*carriers_i.x[1] + T2_2*n1_2*carriers_i.x[1] + T1_2*n1_2*carriers_i.x[0] ) / (carriers_i.x[1] * beta_p_2 + carriers_i.x[0] * beta_n_2 + e_p_2 + e_n_2 + T4_2*carriers_i.x[1]**2 + T4_2*carriers_i.x[1]*p1_2 + T3_2*carriers_i.x[0]*carriers_i.x[1] + T3_2*p1_2*carriers_i.x[0] + T2_2*n1_2*carriers_i.x[1] + T2_2*carriers_i.x[1]*carriers_i.x[0] + T1_2*n1_2*carriers_i.x[0] + T1_1*carriers_i.x[0]**2))
        else:
            n_t1_out = N_1 * ((carriers_i.x[0] * beta_n_1 + e_p_1) / (carriers_i.x[0] * beta_n_1 + carriers_i.x[1] * beta_p_1 + e_n_1 + e_p_1 ))
            p_t2_out = N_2 * ((carriers_i.x[1] * beta_p_2 + e_n_2) / (carriers_i.x[1] * beta_p_2 + carriers_i.x[0] * beta_n_2 + e_p_2 + e_n_2 ))
        
        if print_fitting_info == True:
            # print solver results
            print(carriers_i.message, f'nfev={getattr(carriers_i, "nfev", None)} nit={getattr(carriers_i, "nit", None)}')
            print('Estimated n, p, nt, pt:')
            print("{:e}".format(carriers_i.x[0]*scaling), "{:e}".format(carriers_i.x[1]*scaling),"{:e}".format(n_t1_out*scaling), "{:e}".format(p_t2_out*scaling))
            print((carriers_i.x[0] - n0 + n_t1_out)*scaling, (carriers_i.x[1] - p0 + p_t2_out)*scaling)
            print('Estimated error:')
            print(steady_state_exp(carriers_i.x, G))
            print(error_square)

        solver_failed = []
        if error_square <1e-5: # i just picked a suitably low number, may need adjustment
            solver_failed.append(False)
            n.append(carriers_i.x[0])
            p.append(carriers_i.x[1])
            n_t1.append(n_t1_out)
            p_t2.append(p_t2_out)
        else: # if solver error is too large (failed), carrier densities are set to a dummy value of 0
            solver_failed.append(True)
            n.append(0)
            p.append(0)
            n_t1.append(0)
            p_t2.append(0)

    n = np.array(n)
    p = np.array(p)
    n_t1 = np.array(n_t1)
    p_t2 = np.array(p_t2)
    solver_failed = np.array(solver_failed)

    # calculate steady-state recombination rates
    Raug = Can*n*n*p + Cap*n*p*p
    Rrad = (1-Pr)*krad*(n*p)
    Rsrh_1 = (N_1 * beta_n_1 * beta_p_1 * n * p) / (n * beta_n_1 + p * beta_p_1 + e_n_1 + e_p_1)
    Rsrh_2 = (N_2 * beta_p_2 * beta_n_2 * n * p) / (n * beta_n_2 + p * beta_p_2 + e_n_2 + e_p_2)
    if auger_assisted_trapping == True:
        # need to double check that these are correct
        Rtaug_1 =  T1_1 * ( (n*n)*(N_1-n_t1) - n1_1*n*n_t1 ) + T2_1 * ( (n*p)*(N_1-n_t1) - n1_1*p*n_t1 ) 
        Rtaug_2 =  T4_2 * ( (p*p)*(N_2-p_t2) - p1_2*p*p_t2 ) + T3_2 * ( (n*p)*(N_2-p_t2) - p1_2*n*p_t2 ) 

        Rtot = Raug + Rrad + Rsrh_1 + Rsrh_2 + Rtaug_1 + Rtaug_2
    else:
        Rtot = Raug + Rrad + Rsrh_1 + Rsrh_2 # all rates are um-3 s-1

    Rtot[Rtot == 0] = 1e-3  # low dummy value if solver failed, to avoid division by zero error

    if print_carrier_densities == True:
        print('densities n, p, n_t1, p_t2',n, p, n_t1, p_t2)
        print('fraction n_t/n_tot is ', (n_t1)/(n_t1+n))
        print('fraction p_t/p_tot is ', (p_t2)/(p+p_t2))
        print('fraction traps 1 occupied ', n_t1/N_1)
        print('fraction traps 2 occupied ', p_t2/N_2)

        plt.plot(p, Raug, label = 'Raug')
        plt.plot(p, Rrad, label = 'Rrad')
        plt.plot(p, Rsrh_1, label = 'Rsrh 1')
        plt.plot(p, Rsrh_2, label = 'Rsrh 2')
        if auger_assisted_trapping == True:
            plt.plot(p, Rtaug_1, label = 'Rtaug 1')
            plt.plot(p, Rtaug_2, label = 'Rtaug 2')
        plt.plot(p, Rtot, label = 'Rtotal')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
        plt.clf()

    PLQE_calc = np.where(solver_failed, 100 * np.ones_like(Rtot), Rrad/Rtot)

    # export recombination rates in cm-3 s-1 (scaling is undone)
    if auger_assisted_trapping == True:
        recomb_rates = (Rsrh_1 *scaling, Rsrh_2 *scaling, Rrad *scaling, Raug *scaling, Rtaug_1 *scaling, Rtaug_2 *scaling)
    else:
        recomb_rates = (Rsrh_1 *scaling, Rsrh_2 *scaling, Rrad *scaling, Raug *scaling, np.nan, np.nan)

    return PLQE_calc, n*scaling, p*scaling, recomb_rates