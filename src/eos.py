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
plt.title(r'Entropy Density vs. Temperature for different $\Gamma$')
plt.legend()
plt.savefig("..\\results\\1_entropy_density_vs_temperature.png")
plt.clf()

for Gamma in [1e0, 1e1, 1e2, 1e3]:
    plt.plot(T, s(T, Gamma), label = r'$\Gamma = {}$ MeV'.format(Gamma))
plt.xlabel(r'$T$ [MeV]')
plt.ylabel(r'$S(T)$')
plt.xscale('log')
plt.yscale('log')
plt.title(r'Entropy Density vs. Temperature for different $\Gamma$')
plt.legend()
plt.savefig("..\\results\\1_entropy_density_vs_temperature_log.png")
plt.clf()


for Gamma in [1e0, 1e1, 1e2, 1e3]:
    plt.plot(T, sT3(T, Gamma), label = r'$\Gamma = {}$ MeV'.format(Gamma))
plt.xlabel(r'$T$ [MeV]')
plt.ylabel(r'$S(T) / T^3$')
plt.title(r'Entropy Density vs. Temperature / $T^3$ for different $\Gamma$s')
plt.legend()
plt.savefig("..\\results\\2_entropy_density_vs_temperature_over_T3.png")
plt.clf()

for Gamma in [1e0, 1e1, 1e2, 1e3]:
    plt.plot(T, sT3(T, Gamma), label = r'$\Gamma = {}$ MeV'.format(Gamma))
plt.xlabel(r'$ T$ [MeV]')
plt.ylabel(r'$ S(T) / T^3$')
plt.xscale('log')
plt.yscale('log')
plt.title(r'Entropy Density vs. Temperature / $T^3$ for different $\Gamma$s')
plt.legend()
plt.savefig("..\\results\\2_entropy_density_vs_temperature_over_T3_log.png")
plt.clf()

def P(T):
    value = []
    for i, t in enumerate(T):
        val, err = scipy.integrate.quad(s, 0, t)
        value.append(val)
    return value

def e(T):
    return T * s(T) - P(T)

plt.plot(T, P(T), label = r'$P(T)$')
plt.plot(T, e(T), label = r'$\epsilon(T)$')
plt.xlabel(r'$T$ [MeV]')
plt.title(r'Pressure and Energy Density vs. Temperature')
plt.legend()
plt.savefig("..\\results\\3_pressure_energy_vs_temperature.png")
plt.clf()

plt.plot(T, P(T), label = r'$ P(T)$')
plt.plot(T, e(T), label = r'$ \epsilon(T)$')
plt.xlabel(r'$ T$ [MeV]')
plt.title(r'Pressure and Energy Density vs. Temperature')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig("..\\results\\3_pressure_energy_vs_temperature_log.png")
plt.clf()

plt.plot(T, P(T) / T ** 4, label = r'$P(T) / T^4$')
plt.plot(T, e(T) / T ** 4, label = r'$\epsilon(T) / T^4$')
plt.xlabel(r'$T$ [MeV]')
plt.title(r'Pressure and Energy Density over $T^4$ vs. Temperature')
plt.legend()
plt.savefig("..\\results\\4_pressure_energy_over_T4_vs_temperature.png")
plt.clf()

plt.plot(T, P(T) / T ** 4, label = r'$ P(T) / T^4$')
plt.plot(T, e(T) / T ** 4, label = r'$ \epsilon(T) / T^4$')
plt.xlabel(r'$ T$ [MeV]')
plt.title(r'Pressure and Energy Density over $T^4$ vs. Temperature')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig("..\\results\\4_pressure_energy_over_T4_vs_temperature_log.png")
plt.clf()

def e_idealgas(T):
    value = []
    for i, t in enumerate(T):
        val, err = scipy.integrate.quad(s, 0, t)
        value.append(3 * val)
    return value

def Delta(T):
    return (e(T) - e_idealgas(T)) / T ** 4

T = np.linspace(60, 600, 2000)

plt.plot(T, e(T) / T ** 4, label = r'$\epsilon / T^4$')
plt.plot(T, e_idealgas(T) / T ** 4, label = r'$\epsilon_{ideal gas} / T^4$')
plt.plot(T, Delta(T), label = r'$\Delta$')
plt.xlabel(r'$T$ [MeV]')
plt.ylabel(r'Energy Density / $ T^4$')
plt.title(r'Energy Densities vs. Temperature / $T^4$')
plt.legend()
plt.savefig("..\\results\\5_energy_density_vs_temperature_over_T4.png")
plt.clf()

plt.plot(T, e(T) / T ** 4, label = r'$\epsilon / T^4$')
plt.plot(T, e_idealgas(T) / T ** 4, label = r'$\epsilon_{ideal gas} / T^4$')
plt.plot(T, Delta(T), label = r'$\Delta$')
plt.xlabel(r'$ T$ [MeV]')
plt.ylabel(r'Energy Density /$ T^4$')
plt.title(r'Energy Densities vs. Temperature / $T^4$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("..\\results\\5_energy_density_vs_temperature_over_T4_log.png")
plt.clf()

def cs2(T):
    return (np.diff(np.log(T))  / np.diff(np.log(s(T))))

T = np.linspace(60, 260, 1000)

cs2T = cs2(T)
T = T[:-1]

plt.plot(T, cs2T)
plt.xlabel(r'$T [MeV]$')
plt.ylabel(r'$c_s^2$')
plt.title(r'Speed of Sound Squared vs. Temperature')
# plot a vertical line at Tc
plt.vlines(x = Tc, ymin = 0, ymax = 0.6, color = 'r', linestyle = '--')
plt.savefig("..\\results\\6_speed_of_sound_squared_vs_temperature.png")
plt.clf()

plt.plot(T, cs2T)
plt.xlabel(r'$ T [MeV]$')
plt.ylabel(r'$ c_s^2$')
plt.xscale('log')
plt.yscale('log')
plt.title(r'Speed of Sound Squared vs. Temperature')
plt.vlines(x = Tc, ymin = 0, ymax = 1e1, color = 'r', linestyle = '--')
plt.savefig("..\\results\\6_speed_of_sound_squared_vs_temperature_log.png")
plt.clf()