"""This script compares numerical and analytical result of one spin after one precession
in an external magnetic field of 0.1 T. This comparison is done for different time steps
in order to recreate figure 10 of the paper "The design and verification of MuMax3".
However with different algorithms.
https://doi.org/10.1063/1.4899186 """


import matplotlib.pyplot as plt
import numpy as np

from mumaxplus import *
from mumaxplus.util import *


def single_system(method, dt):
    """This function simulates a single spin in a magnetic field of 0.1 T without damping.

    Returns the absolute error between the simulation and the exact solution.

    Parameters:
    method -- The used simulation method
    dt     -- The time step
    """
    # --- Setup ---
    world = World(cellsize=(1e-9, 1e-9, 1e-9))
    
    hfield_z = 0.1  # External field strength
    duration = 2*np.pi/(GAMMALL * hfield_z)  # Time of one precession

    magnet = Ferromagnet(world, grid=Grid((1, 1, 1)))
    magnet.enable_demag = False
    magnet.magnetization = (1/np.sqrt(2), 0, 1/np.sqrt(2))
    magnet.alpha = 0.
    magnet.aex = 10e-12
    magnet.msat = 1/MU0
    world.bias_magnetic_field = (0, 0, hfield_z)

    # --- Run the simulation ---
    world.timesolver.set_method(method)
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = dt
    
    world.timesolver.run(duration)
    output = magnet.magnetization.average()

    # --- Compare with exact solution ---
    exact = np.array([1/np.sqrt(2), 0., 1/np.sqrt(2)])
    error = np.linalg.norm(exact - output)

    return error


method_names = ["Heun", "BogackiShampine", "CashKarp", "Fehlberg", "DormandPrince"]

exact_names = {"Heun": "Heun",
               "BogackiShampine": "Bogacki-Shampine",
               "CashKarp": "Cash-Karp",
               "Fehlberg": "Fehlberg",
               "DormandPrince": "Dormand-Prince"
               }

RK_names = {"Heun": "RK12",
            "BogackiShampine": "RK32",
            "CashKarp": "RKCK45",
            "Fehlberg": "RKF45",
            "DormandPrince": "RK45"
            }

exact_order = {"Heun": 2,
               "BogackiShampine": 3,
               "CashKarp": 5,
               "Fehlberg": 5,
               "DormandPrince": 6
               }

N_dens = 30  # Amount of datapoints between two powers of 10
dts = {"Heun": np.logspace(-12, np.log10(0.4e-10), int(N_dens * (np.log10(0.4e-10) + 12))), 
       "BogackiShampine": np.logspace(np.log10(0.2e-11), np.log10(0.4e-10), int(N_dens * (np.log10(0.4e-10) - np.log10(0.2e-11)))),
       "CashKarp": np.logspace(np.log10(1.5e-11), np.log10(0.4e-10), int(N_dens * (np.log10(0.4e-10) - np.log10(1.5e-11)))),
       "Fehlberg": np.logspace(np.log10(1.2e-11), np.log10(0.4e-10), int(N_dens * (np.log10(0.4e-10) - np.log10(1.2e-11)))),
       "DormandPrince": np.logspace(np.log10(1.4e-11), np.log10(0.4e-10), int(N_dens * (np.log10(0.4e-10) - np.log10(1.4e-11))))
       }

# --- Plotting ---
plt.xscale('log')
plt.yscale('log')
plt.xlim((0.9e-12, 0.5e-10))
plt.ylim((1e-7, 1))
plt.xlabel("time step (s)")
plt.ylabel("absolute error after 1 precession")

# --- Simulation Loops ---
orders = {}
for method in method_names:
    error = np.zeros(shape=dts[method].shape)
    for i, dt in enumerate(dts[method]):
        err = single_system(method, dt)
        error[i] = err
    
    # Find the order
    log_dts, log_error = np.log10(dts[method]), np.log10(error)
    order = np.polyfit(log_dts, log_error, 1)[0]
    orders[exact_names[method]] = order

    plt.scatter(dts[method], error, marker="o", zorder=2)

    intercept = np.polyfit(log_dts, log_error - log_dts * exact_order[method], 0)
    plt.plot(np.array([1e-14, 1e-9]), (10**intercept)*np.array([1e-14, 1e-9])**exact_order[method], label=f"{RK_names[method]} {exact_names[method]}")

#print(orders)  # Uncomment if you want to see the estimated orders
plt.legend()
plt.show()