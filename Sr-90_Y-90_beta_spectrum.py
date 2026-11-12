from scipy import constants as p
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integral
from scipy import special as sp
from scipy import optimize as opt
import scipy.interpolate as ip

import streamlit as st
st.header('Beta Spectrum of Sr-90 and Y-90')


rho_air = 1.205e-3

#Energy in SI
def fermi_sr(E_1):
    E_1 = np.maximum(E_1, 1e-30)
    E = E_1/(p.m_e*p.c**2)
    return np.sqrt(8.6994+4.7116/E)

#Calculated in SI
M_fi_sr = 0.693*(2*p.pi**3*p.h**7)/(p.g**2*p.m_e**5*p.c**4*10**9.4)

#Energy in SI
def shape_sr(E_n):
    C_1 = 1.0965*(1-0.0054*(1+E_n/(p.m_e*p.c**2)))
    return C_1*(1/p.c)**4*(E_n**2+2*p.m_e*p.c**2*E_n)*(0.54*p.e*10**6-E_n)**2

#energy in MEV
def en_sr(E_n):
    if E_n<0.54:
        E = E_n*10**6*p.e
        return np.sqrt(E**2+2*E*p.m_e*p.c**2)*(E+p.m_e*p.c**2)*(0.54*p.e*10**6-E)**2*fermi_sr(E)*M_fi_sr*shape_sr(E)
    if E_n>0.54 or E_n<0:
        return 0
    
def norm_sr(E_n):
    return en_sr(E_n)/integral.quad(en_sr, 0, 0.54)[0]


#Energy in SI
def fermi_y(E_1):
    E_1 = np.maximum(E_1, 1e-30)
    E = E_1/(p.m_e*p.c**2)
    return np.sqrt(9.3330+5.2025/E)

#Calculated in SI
M_fi_y = 0.693*(2*p.pi**3*p.h**7)/(p.g**2*p.m_e**5*p.c**4*10**9.224)

#Energy in SI
def shape_y(E_n):
    C_1 = 1.0270*(1-0.0072*(1+E_n/(p.m_e*p.c**2)))
    return C_1*(1/p.c)**4*(E_n**2+2*p.m_e*p.c**2*E_n)*(2.28*p.e*10**6-E_n)**2

#energy in MEV
def en_y(E_n):
    if E_n<=2.28:
        E = E_n*10**6*p.e
        return np.sqrt(E**2+2*E*p.m_e*p.c**2)*(E+p.m_e*p.c**2)*(2.28*p.e*10**6-E)**2*fermi_y(E)*M_fi_sr*shape_y(E)
    if E_n>2.28 or E_n<0:
        return 0
    
def norm_y(E_n):
    return en_y(E_n)/integral.quad(en_y, 0, 2.28)[0]

def norm_tot(E_n):
    return 0.5*(norm_sr(E_n)+norm_y(E_n))

E_min = 0.0          # MeV
E_max = 2.28        # Sr-90 endpoint energy in MeV
num_points = 2000
E_vals = np.linspace(E_min, E_max, num_points)

y_vals = np.zeros(len(E_vals))
for i in range(0,len(E_vals)):
    y_vals[i] = 0.5*(np.nan_to_num(norm_sr(E_vals[i]))+np.nan_to_num(norm_y(E_vals[i])))

# ---------- Plot ----------
plt.figure(figsize=(8,5))
plt.plot(E_vals, y_vals, color='darkblue', lw=2, label=r'Total $\beta$ spectrum')
plt.fill_between(E_vals, 0, y_vals, color='lightblue', alpha=0.3)

plt.title(r'$\beta^-$ Energy Spectrum of $^{90}$Sr → $^{90}$Y and $^{90}$Y → $^{90}$Zr', fontsize=14)
plt.xlabel('Kinetic energy $E$ / MeV', fontsize=12)
plt.ylabel('Probability density / MeV$^{-1}$', fontsize=12)
plt.xlim(E_min, E_max)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
#plt.savefig('energy_distribution.png')
plt.show()

def prod(E_n):
    return E_n*0.5*(norm_y(E_n)+norm_sr(E_n))
y_vals_maxnorm = np.max(y_vals)
def act_prod(E_n):
    return E_n*0.5*(norm_y(E_n)+norm_sr(E_n))/y_vals_maxnorm

print(integral.quad(prod, 0, 2.28), integral.quad(act_prod, 0, 2.28))

energies, stopping_p = np.loadtxt('/Users/chinmayib/Downloads/energies.txt', unpack=True)

#undiscretize the energy stopping power relation

stop_pow = ip.interp1d(energies, stopping_p)

def CSDA(E):
    total = 0
    if 0<=E<=2.28:
        idx = np.argmin(np.abs(E-energies))
        for i in range(0, idx+1):
            total = total+0.01/stop_pow(energies[i])
    else:
        return 0
    return total+(E-energies[idx])*1/stop_pow(E)

#discrete CSDA range array
distances = np.zeros(len(energies))
for i in range(0, len(distances)):
    distances[i] = CSDA(energies[i])/rho_air

#undiscretizing CSDA range Energy relation

CSDA_En = ip.interp1d(energies, distances)

#energy from CSDA range

En_ran = ip.interp1d(distances, energies)

#range straggling NOT implemented using erfc(x)
def frac_hit(d):
    E_0 = En_ran(d)      
    return integral.quad(norm_tot, E_0, 2.28)[0]


#Multiple scattering estimate
def theta_highland(d, E_n):
    def psq(E_a):
        E = E_a*10**6*p.e
        return (1.871*10**21)**2*(E**2+2*p.m_e*p.c**2*E)/p.c**2
    def betasq(E_b):
        return psq(E_b)/(psq(E_b)+p.m_e**2*p.c**2*(1.871*10**21)**2)
    sum=0
    idx = np.abs(CSDA(E_n)/rho_air-d-distances).argmin()
    idx2 = np.abs(E_n-energies).argmin()
    i_s = min(idx, idx2)
    i_e = max(idx, idx2)
    for i in range(i_s, i_e+1):
        if i_e>=1:    
            sum = sum+0.01*(14.1)**2/(psq(energies[i])*betasq(energies[i]))*(1/36.62)*(1/stopping_p[i])
        elif i_e == 0:
            sum = 0
    return (1+1/9*np.log10((d*rho_air)/(36.62)))*np.sqrt(sum+(E_n-energies[i_e])*(14.1)**2/(psq(E_n)*betasq(E_n))*(1/36.62)*(1/stop_pow(E_n)))

def angle_dist(theta, E_n, d):
    if theta_highland(d, E_n)<20*np.pi/180:
        return 2*np.exp(-theta**2/(theta_highland(d, E_n)**2))/(theta_highland(d, E_n))**2
    elif theta_highland(d, E_n)>20*np.pi/180:
        return np.cos(theta)**2/np.pi