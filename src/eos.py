import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

Tc = 160 # MeV

def s(T, Gamma = 8):
    return 1 / 2 * ((1 - np.tanh((T - Tc) / Gamma)) * 2 / 15 * np.pi ** 2 * T ** 3 + (1 + np.tanh((T - Tc) / Gamma)) * 53 / 45 * np.pi ** 2 * T ** 3)

def sT3(T, Gamma):
    return s(T, Gamma) / T ** 3

# plot S(T) for Gamma = 1, 10, 100 MeV
T = np.linspace(60, 260, 1000)

for Gamma in [1e0, 1e1, 1e2, 1e3]:
    plt.plot(T, s(T, Gamma), label = r'$\Gamma = {}$ MeV'.format(Gamma))
plt.xlabel(r'$T$ [MeV]')
plt.ylabel(r'$S(T)$')
plt.xscale('log')
plt.yscale('log')
plt.title(r'Entropy Density vs. Temperature for different $\Gamma$')
plt.legend()
plt.show()


for Gamma in [1e0, 1e1, 1e2, 1e3]:
    plt.plot(T, sT3(T, Gamma), label = r'$\Gamma = {}$ MeV'.format(Gamma))
plt.xlabel(r'$T$ [MeV]')
plt.ylabel(r'$S(T) / T^3$')
plt.xscale('log')
plt.yscale('log')
plt.title(r'Entropy Density vs. Temperature / $T^3$ for different $\Gamma$s')
plt.legend()
plt.show()

def P(T):
    value = []
    for i, t in enumerate(T):
        val, err = scipy.integrate.quad(s, 0, t)
        value.append(val)
    return value

plt.plot(T, P(T))
plt.xlabel(r'$T$ [MeV]')
plt.ylabel(r'$P(T)$')
plt.title(r'Pressure vs. Temperature')
plt.xscale('log')
plt.yscale('log')
plt.show()

def e(T):
    return T * s(T) - P(T)

plt.plot(T, e(T))
plt.xlabel(r'$T$ [MeV]')
plt.ylabel(r'$\epsilon(T)$')
plt.title(r'Energy Density vs. Temperature')
plt.xscale('log')
plt.yscale('log')
plt.show()

def e_idealgas(T):
    value = []
    for i, t in enumerate(T):
        val, err = scipy.integrate.quad(s, 0, t)
        value.append(3 * val)
    return value

def Delta(T):
    return (e(T) - e_idealgas(T)) / T ** 4

plt.plot(T, e(T) / T ** 4, label = r'$\epsilon / T^4$')
#plt.plot(T, P(T) / T ** 4, label = r'$P(T) / T^4$')
#plt.plot(T, s(T) / T ** 3, label = r'$s(T) / T^3$')
plt.plot(T, e_idealgas(T) / T ** 4, label = r'$\epsilon_{ideal gas} / T^4$')
plt.plot(T, Delta(T), label = r'$\Delta$')
plt.xlabel(r'T [MeV]')
plt.ylabel(r'Energy Density / $ T^4$')
plt.title(r'Energy Densities vs. Temperature / $T^4$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

def cs2(T):
    return (np.diff(np.log(T))  / np.diff(np.log(s(T))))

plt.plot(T, cs2(T))
plt.xlabel(r'$T [MeV]$')
plt.ylabel(r'$c_s^2$')
plt.xscale('log')
plt.yscale('log')
plt.title(r'Speed of Sound Squared vs. Temperature')
plt.show()