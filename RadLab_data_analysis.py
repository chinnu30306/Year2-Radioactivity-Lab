import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from scipy.stats import chi2
from scipy.special import lambertw


def exponential(x, a, b):
    return a * np.exp(-b * x)

def linear(x, m, c):
    return m * x + c

def intersect(x, m1, c1, m2, c2):
    return (c2 - c1) / (m1 - m2)

def straightLine(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return m, c

def beerLambert(x, I1, mu1, I2, mu2):
    return I1 * np.exp(-mu1 * x) + I2 * np.exp(-mu2 * x)

def E_max_calc(R, a, b):
    return np.sqrt(((R/a + 1)**2 - 1)/b)

def stopping_power_incorrect(KE, beta, Z=13, A=26.98, I=166):
    return (0.307075 * Z / A) * (1/beta**2) * np.log(1.16*KE)

def stopping_power(KE, C, k):

    me = 0.511
    gamma = 1 + KE/me
    beta = np.sqrt(1 - 1/gamma**2)

    Z = 13
    A = 26.98
    I = 166e-6

    collisional = C*(0.307075 * Z/A) * (1/beta**2) * (
        np.log((2*me*beta**2*gamma**2)/I) - beta**2
    )

    radiative = k*KE

    return collisional + radiative

def saturation(x, a, b):
    return a - (a - 22.2) * np.exp(-x/b)

'''
file_name = os.listdir("/Users/chinmayib/Downloads/RadLab")
file_name = [f for f in file_name if f.endswith('.txt')]

for i in file_name:
    n = np.loadtxt("/Users/chinmayib/Downloads/RadLab/" + i, skiprows=1, delimiter=',', usecols=0)
    hist, bin_edges = np.histogram(n, bins=10)
    bincentres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.errorbar(bincentres, hist, yerr=np.sqrt(hist), fmt='.', capsize=2)
    plt.hist(n, bins=10)
    plt.xlabel('counts per cycle')
    plt.ylabel('number of cycles')
    plt.title(i)
    plt.grid()
    plt.show()


task_15 = os.listdir("/Users/chinmayib/Downloads/task_15")
task_15 = [f for f in task_15 if f.endswith('.txt')]

n = np.array([])
d = np.array([])
for i in task_15:
    d = np.append(d, float(i[:-4]))
    data = np.loadtxt("/Users/chinmayib/Downloads/task_15/" + i, skiprows=1, delimiter=',', usecols=0)
    n = np.append(n, np.mean(data))

# 10sec per cycle
plt.plot(d, n, 'o')
plt.xlabel('distance (cm)')
plt.ylabel('mean counts per cycle')
plt.title('Mean counts per cycle vs distance')
plt.grid()
plt.show()

d = d/100
x = n/10 * d**2
plt.plot(d, x, 'o')
error = np.sqrt((d**2/10 * np.sqrt(n))**2 + (2e4*n*d)**2)/x
mean = np.mean(x[1:])
print("mean:", mean)
plt.errorbar(d, x, yerr=error, fmt='.', capsize=2)
plt.xlabel('distance (m)')
plt.ylabel('mean counts per sec * distance^2')
plt.title('Mean counts per sec * distance^2 vs distance')
plt.grid()
plt.show()
'''

'''
#extension - 1mm Al
thickness_mm, countRate, sampleTime = np.loadtxt("/Users/chinmayib/Downloads/Al_1mm.txt", skiprows=1, usecols=(0,1,2), unpack=True)
thickness_cm = thickness_mm / 10
countRate_perSec = countRate / sampleTime
error = np.sqrt(countRate) / sampleTime
countRate_background = np.mean(countRate_perSec[4:])

popt_exp, pcov_exp = curve_fit(exponential, thickness_cm, countRate_perSec)
plt.plot(thickness_cm, countRate_perSec, 'x')
plt.errorbar(thickness_cm, countRate_perSec, yerr=error, fmt='.', capsize=2)
plt.plot(thickness_cm, exponential(thickness_cm, *popt_exp))
plt.xlabel('thickness of Al (cm)')
plt.ylabel('count rate (per sec)')
plt.title('Count rate vs thickness of Al')
plt.grid()
plt.show()

log_countRate = np.log(countRate_perSec)
log_error = 1/countRate_perSec
m1, c1 = straightLine(thickness_cm[0], log_countRate[0], thickness_cm[1], log_countRate[1])
m2, c2 = straightLine(thickness_cm[1], log_countRate[1], thickness_cm[2], log_countRate[2])
m3, c3 = straightLine(thickness_cm[3], log_countRate[3], thickness_cm[4], log_countRate[4])
popt1 = [m1, c1]
popt2 = [m2, c2]
popt3 = [m3, c3]

R = intersect(thickness_cm, *popt2, *popt3) * 2.7
E_max_1mm = np.sqrt(((R/0.11 + 1)**2 - 1)/22.4)
print("E_max for 1mm Al sheets:", E_max_1mm, "MeV")

plt.plot(thickness_cm, log_countRate, 'x')
plt.errorbar(thickness_cm, log_countRate, yerr=log_error, fmt='.', capsize=2)
plt.plot(thickness_cm[:2], linear(thickness_cm[:2], *popt1))
plt.plot(thickness_cm[1:3], linear(thickness_cm[1:3], *popt2))
plt.plot(thickness_cm[3:], linear(thickness_cm[3:], *popt3))
plt.axhline(np.log(countRate_background), linestyle='--')
plt.xlabel('thickness of Al (cm)')
plt.ylabel('ln(count rate) (per sec)')
plt.title('Natural log of count rate vs thickness of 1mm Al sheets')
plt.grid()
plt.show()


#extension - 0.5mm Al
thickness_mm, countRate, sampleTime = np.loadtxt("/Users/chinmayib/Downloads/Al_05mm.txt", skiprows=1, usecols=(0,1,2), unpack=True)
thickness_cm = thickness_mm / 10
countRate_perSec = countRate / sampleTime
error = np.sqrt(countRate) / sampleTime
countRate_background = np.mean(countRate_perSec[7:])

popt_exp, pcov_exp = curve_fit(exponential, thickness_cm, countRate_perSec)
plt.plot(thickness_cm, countRate_perSec, 'x')
plt.errorbar(thickness_cm, countRate_perSec, yerr=error, fmt='.', capsize=2)
plt.plot(thickness_cm, exponential(thickness_cm, *popt_exp))
plt.xlabel('thickness of Al (cm)')
plt.ylabel('count rate (per sec)')
plt.title('Count rate vs thickness of 0.5mm Al sheets')
plt.grid()
plt.show()

log_countRate = np.log(countRate_perSec)
log_error = 1/countRate_perSec
popt1, pcov1 = curve_fit(linear, thickness_cm[:3], log_countRate[:3])
popt2, pcov2 = curve_fit(linear, thickness_cm[3:6], log_countRate[3:6])
popt3, pcov3 = curve_fit(linear, thickness_cm[7:], log_countRate[7:])

R = intersect(thickness_cm, *popt2, *popt3) * 2.7
E_max_05mm = np.sqrt(((R/0.11 + 1)**2 - 1)/22.4)
print("E_max for 0.5mm Al sheets:", E_max_05mm, "MeV")

plt.plot(thickness_cm, log_countRate, 'x')
plt.errorbar(thickness_cm, log_countRate, yerr=log_error, fmt='.', capsize=2)
plt.plot(thickness_cm[:3], linear(thickness_cm[:3], *popt1))
plt.plot(thickness_cm[3:6], linear(thickness_cm[3:6], *popt2))
plt.plot(thickness_cm[7:], linear(thickness_cm[7:], *popt3))
plt.axhline(np.log(countRate_background), linestyle='--')
plt.xlabel('thickness of Al (cm)')
plt.ylabel('ln(count rate) (per sec)')
plt.title('Natural log of count rate vs thickness of 0.5mm Al sheets')
plt.grid()
plt.show()
'''

#backscatter
countRate_backscatter = np.array([22.2, 26.3, 29.2, 31.5, 31.9, 32.3, 32.3])
thickness = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
error_backscatter = np.sqrt(countRate_backscatter)
plt.errorbar(thickness, countRate_backscatter, yerr=error_backscatter, fmt='x', capsize=2, label='Data points')
p0 = [87, 0.13]
popt_backscatter, pcov_backscatter = curve_fit(saturation, thickness, countRate_backscatter)
thickness_smooth = np.linspace(0, 0.4, 1000)
plt.plot(thickness, countRate_backscatter, 'x')
#plt.plot(thickness_smooth, 22.2 + 10.2 * (1-np.exp(-thickness_smooth/0.025)), label='Backscatter fit')
plt.plot(thickness_smooth, saturation(thickness_smooth, *popt_backscatter), label='Saturation fit')
plt.xlabel('Thickness of Al (cm)')
plt.ylabel('Count rate (per sec)')
plt.title('Backscatter count rate vs thickness of Al')
plt.grid()
plt.show()

#extension - 0.1mm Al
thickness_mm, countRate, sampleTime = np.loadtxt("/Users/chinmayib/Downloads/extension_0.1mm.txt", skiprows=1, usecols=(0,1,2), unpack=True)
thickness_cm = thickness_mm / 10
'''
for i in range(len(thickness_mm)):
    if i > 15:
        thickness_mm[i] += 0.1
thickness_cm = np.insert(thickness_cm, 16, 0.17)
countRate = np.insert(countRate, 16,31773)
sampleTime = np.insert(sampleTime, 16, 17)
'''
areal_density_Al = thickness_cm * 2.7
countRate_perSec_Al_incorrect = countRate / sampleTime

tau = 2e-6   # example dead time (seconds)
Rm = countRate_perSec_Al_incorrect  # observed count rate (counts per second)

countRate_perSec_Al = -(1/tau)*lambertw(-Rm*tau).real / 0.29

error = np.sqrt(countRate) / sampleTime
countRate_background_Al = np.mean(countRate_perSec_Al[-3:])

plt.plot(areal_density_Al, countRate_perSec_Al, 'x')
plt.errorbar(areal_density_Al, countRate_perSec_Al, yerr=error, fmt='.', capsize=2)
plt.xlabel('Areal density of Al (g/cm^2)')
plt.ylabel('count rate (per sec)')
plt.title('Count rate vs areal density of 0.1mm Al sheets')
plt.grid()
plt.show()

log_countRate = np.log(countRate_perSec_Al)
log_error = 1/np.sqrt(countRate)
log_error_upper = np.log(countRate_perSec_Al + error)
log_error_lower = np.log(countRate_perSec_Al - error)
log_error_total_Al = [log_error_lower, log_error_upper]
log_error_average = (log_error_upper - log_error_lower) / 2
popt1, pcov1 = curve_fit(linear, areal_density_Al[:11], log_countRate[:11])
exp1_Al = np.exp(popt1[0] * areal_density_Al + popt1[1])
popt2, pcov2 = curve_fit(linear, areal_density_Al[23:32], log_countRate[23:32])
exp2_Al = np.exp(popt2[0] * areal_density_Al + popt2[1])
popt3 = [0, np.log(countRate_background_Al)]
exp3_Al = np.exp(popt3[0] * areal_density_Al + popt3[1])

R_mid = intersect(areal_density_Al, *popt1, *popt2)
E_max_mid = np.sqrt(((R_mid/0.11 + 1)**2 - 1)/22.4)
print("Midpoint E_max for 0.1mm Al sheets:", E_max_mid, "MeV")

R_Al = intersect(areal_density_Al, *popt2, *popt3) + 0.005*2.7 + 0.1*1.18
print("R for 0.1mm Al sheets:", R_Al, "g/cm^2")
E_max = np.sqrt(((R_Al/0.11 + 1)**2 - 1)/22.4)
print("E_max for 0.1mm Al sheets:", E_max, "MeV")

chi2_1 = np.sum((log_countRate[:15] - linear(areal_density_Al[:15], *popt1))**2 / log_error_average[:15]**2)
dof1 = len(areal_density_Al[:15]) - 2
pval_01mm1 = 1 - chi2.cdf(chi2_1, dof1)
print("p-value for 0.1mm Al sheets in region 1:", pval_01mm1)
chi2_2 = np.sum((log_countRate[23:32] - linear(areal_density_Al[23:32], *popt2))**2 / log_error_average[23:32]**2)
dof2 = len(areal_density_Al[23:32]) - 2
pval_01mm2 = 1 - chi2.cdf(chi2_2, dof2)
print("p-value for 0.1mm Al sheets in region 2:", pval_01mm2)
chi2_3 = np.sum((log_countRate[-3:] - linear(areal_density_Al[-3:], *popt3))**2 / log_error_average[-3:]**2)
dof3 = len(areal_density_Al[-3:]) - 2
pval_01mm3 = 1 - chi2.cdf(chi2_3, dof3)
print("p-value for 0.1mm Al sheets in region 3:", pval_01mm3)

plt.yscale('log')
plt.plot(areal_density_Al, countRate_perSec_Al, '.')
plt.errorbar(areal_density_Al, countRate_perSec_Al, yerr=log_error_total_Al, fmt='.', capsize=2, label='Data points')
#plt.plot(areal_density_Al, exp1_Al, label='Region 1')
plt.plot(areal_density_Al, exp2_Al, label='Beta attenuation fit')
#plt.plot(areal_density_Al, exp3_Al)
plt.axhline(countRate_background_Al, label='Background gamma count rate')
plt.legend()
plt.xlabel('Areal density of Al (g/cm^2)')
plt.ylabel('Count rate (per sec)')
plt.grid()
plt.show()

'''
#extension - 0.1mm Cu
thickness_mm, countRate, sampleTime = np.loadtxt("/Users/chinmayib/Downloads/Cu_01mm_2.txt", skiprows=1, usecols=(0,1,2), unpack=True)
thickness_cm = thickness_mm / 10
areal_density = thickness_cm * 8.96
countRate_perSec = countRate / sampleTime
error = np.sqrt(countRate) / sampleTime
countRate_background = np.mean(countRate_perSec[-3:])

plt.plot(areal_density, countRate_perSec, 'x')
plt.errorbar(areal_density, countRate_perSec, yerr=error, fmt='.', capsize=2)
plt.xlabel('Areal density of Cu (g/cm^2)')
plt.ylabel('count rate (per sec)')
plt.title('Count rate vs areal density of 0.1mm Cu sheets')
plt.grid()
plt.show()

log_countRate = np.log(countRate_perSec)
log_error = 1/np.sqrt(countRate)
log_error_upper = np.log(countRate_perSec + error)
log_error_lower = np.log(countRate_perSec - error)
log_error_total = [log_error_lower, log_error_upper]
log_error_average = (log_error_upper - log_error_lower) / 2
popt1, pcov1 = curve_fit(linear, areal_density[:3], log_countRate[:3])
exp1 = np.exp(popt1[0] * areal_density + popt1[1])
popt2, pcov2 = curve_fit(linear, areal_density[6:10], log_countRate[6:10])
exp2 = np.exp(popt2[0] * areal_density + popt2[1])
popt3 = [0, np.log(countRate_background)]

R_mid = intersect(areal_density, *popt1, *popt2)
E_max_mid = np.sqrt(((R_mid/0.11 + 1)**2 - 1)/22.4)
print("Midpoint E_max for 0.1mm Cu sheets:", E_max_mid, "MeV")

R = intersect(areal_density, *popt2, *popt3) + 0.005*8.96 + 0.1*1.18
E_max = np.sqrt(((R/0.11 + 1)**2 - 1)/22.4)
print("E_max for 0.1mm Cu sheets:", E_max, "MeV")

chi2_1 = np.sum((log_countRate[:3] - linear(areal_density[:3], *popt1))**2 / log_error_average[:3]**2)
dof1 = len(areal_density[:3]) - 2
pval_01mm1 = 1 - chi2.cdf(chi2_1, dof1)
print("p-value for 0.1mm Cu sheets in region 1:", pval_01mm1)
chi2_2 = np.sum((log_countRate[6:10] - linear(areal_density[6:10], *popt2))**2 / log_error_average[6:10]**2)
dof2 = len(areal_density[6:10]) - 2
pval_01mm2 = 1 - chi2.cdf(chi2_2, dof2)
print("p-value for 0.1mm Cu sheets in region 2:", pval_01mm2)
chi2_3 = np.sum((log_countRate[-3:] - linear(areal_density[-3:], *popt3))**2 / log_error_average[-3:]**2)
dof3 = len(areal_density[-3:]) - 2
pval_01mm3 = 1 - chi2.cdf(chi2_3, dof3)
print("p-value for 0.1mm Cu sheets in region 3:", pval_01mm3)

plt.yscale('log')
plt.plot(areal_density, countRate_perSec, '.')
plt.errorbar(areal_density, countRate_perSec, yerr=log_error_total, fmt='.', capsize=2)
plt.plot(areal_density, exp1)
plt.plot(areal_density, exp2)
plt.axhline(countRate_background, linestyle='--')
plt.xlabel('Areal density (g/cm^2)')
plt.ylabel('Count rate (per sec)')
plt.title('Count rate vs areal density of 0.1mm Cu sheets')
plt.grid()
plt.show()
'''

#energy vs range theoretical
R_test = np.linspace(0, 1.45, num=1000)
E_max_test = np.sqrt(((R_test/0.11 + 1)**2 - 1)/22.4)
plt.plot(R_test, E_max_test, label=r'$R = 0.11(\sqrt{1+22.4E_{max}^2}+1)$ for $0<E_{max}<3MeV$')

KE_MeV_Al, R_gcm2_Al = np.loadtxt("/Users/chinmayib/Downloads/CDSA data Al.txt", unpack=True)
plt.plot(R_gcm2_Al, KE_MeV_Al, '.')
plt.errorbar(R_gcm2_Al, KE_MeV_Al, xerr=0.05*R_gcm2_Al, fmt='.', capsize=2)
start = 32
stop = 39
popt_straight, pcov_straight = curve_fit(linear, R_gcm2_Al[start:stop], KE_MeV_Al[start:stop], p0=[0.11, 22.4])
plt.plot(R_gcm2_Al[start:stop], linear(R_gcm2_Al[start:stop], *popt_straight), label=r'Linear fit for $1<E_{max}<3MeV$')
'''
p0 = [0.0967, 24.83]
bounds = ([0.08, 15], [0.16, 30])
popt_Al, pcov_Al = curve_fit(E_max_calc, R_gcm2_Al, KE_MeV_Al, p0=p0, bounds=bounds, maxfev=2000)
plt.plot(R_gcm2_Al, E_max_calc(R_gcm2_Al, *popt_Al), label='Aluminium fit')
'''
plt.legend()
plt.xlim(-0.1, 2.5)
plt.ylim(-0.1, 4)
plt.xlabel('Areal density R (g/cm^2)')
plt.ylabel('Maximum energy E_max (MeV)')
plt.grid()
plt.show()

chi2_Al = np.sum((KE_MeV_Al[start:stop] - linear(R_gcm2_Al[start:stop], *popt_straight))**2 / 0.05*R_gcm2_Al[start:stop]**2)
dof = len(R_gcm2_Al[start:stop]) - 2
pval = 1 - chi2.cdf(chi2_Al, dof)
print("p-value for range vs energy:", pval)

corrected_E_max_Al = linear(R_Al, *popt_straight)
print("Corrected E_max values for Al:", corrected_E_max_Al)

'''
#bethe-bloch
KE_MeV, Stopping_Power_MeVcm2g = np.loadtxt("/Users/chinmayib/Downloads/CDSA Al stopping power.txt", unpack=True)
popt, pcov = curve_fit(
    stopping_power,
    KE_MeV,
    Stopping_Power_MeVcm2g,
    p0=[1,0.03]
)
plt.plot(KE_MeV, stopping_power(KE_MeV, *popt))
plt.plot(KE_MeV, Stopping_Power_MeVcm2g, 'x')
plt.xlabel('Kinetic energy (MeV)')
plt.ylabel('Stopping power (MeV cm^2/g)')
plt.title('Stopping power of aluminium for electrons from NIST ESTAR data')
plt.grid()
plt.show()
'''