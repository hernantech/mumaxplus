import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mumaxplus import Antiferromagnet, Grid, World
from mumaxplus.util.constants import GAMMALL, MU0
import os.path

# Antiferromagnetic spinwave dispersion relation

# Numerical parameters
fmax = 5E13            # maximum frequency (in Hz) of the sinc pulse
T = 2E-12              # simulation time (longer -> better frequency resolution)
dt = 1 / (2 * fmax)    # the sample time
nt = 1 + int(T / dt)
dx = 0.5E-9            # cellsize
nx = 512*5             # number of cells

# Material/system parameters
a = dx              # lattice constant
Bdc = 2             # bias field along the z direction
A = 10E-12          # exchange constant
A_nn = -5E-12
A_c = -400E-12
Ms = 400e3          # saturation magnetization
alpha = 1e-3        # damping parameter
K = 611e3

# Create the world
grid_size = (nx, 1, 1)
cell_size = (dx, dx, dx)

m_filename = "FM_disp.npy"

world = World(cell_size)
world.bias_magnetic_field = (0, 0, Bdc)

def simulate():
    # Create a ferromagnet
    magnet = Antiferromagnet(world, Grid(size=grid_size))
    for sub in magnet.sublattices:
        sub.msat = Ms
        sub.aex = A
        sub.alpha = alpha
        sub.ku1 = K
        sub.anisU = (0, 0, 1)

    magnet.afmex_nn = A_nn
    magnet.afmex_cell = A_c
    magnet.latcon = a

    Bt = lambda t: (1e2 * np.sinc(2 * fmax * (t - T / 2)), 0, 0)
    mask = np.zeros(shape=(1, 1, nx))
    # Put signal at the center of the simulation box
    mask[:, :, nx // 2 - 1:nx // 2 + 1] = 1
    for sub in magnet.sublattices:
        sub.bias_magnetic_field.add_time_term(Bt, mask)

    magnet.sub1.magnetization = (0, 0, 1)
    magnet.sub2.magnetization = (0, 0, -1)

    # Run solver
    m = np.zeros(shape=(nt, 3, 1, 1, nx))
    m[0,...] = magnet.sub1.magnetization.eval()
    for i in tqdm(range(nt)):
        world.timesolver.run(dt)
        m[i,...] = magnet.sub1.magnetization.eval()
    np.save(m_filename, m)
    return m

# check if the files already exist
if os.path.isfile(m_filename):
    m = np.load(m_filename)
else:
    m = simulate()

# extent of k values and frequencies, compensated for cell-width
xmin, xmax = -np.pi/dx, np.pi/dx  # rad/m
ymin, ymax = 0, fmax  # Hz

extent = [-(2 * np.pi) / (2 * dx) * (nx+1)/nx,
          (2 * np.pi) / (2 * dx) * (nx-1)/nx,
          -1 / (2 * dt) * nt/(nt-1),
          1 / (2 * dt) * nt/(nt-1)]

# Fourier transform in time and x-direction of magnetization
m_FT = np.abs(np.fft.fftshift(np.fft.fft2(m[:,1,0,0,:])))

# normalize the transform in the relevant area, so it is visible in the plot
x_start = int((xmin - extent[0]) / (extent[1] - extent[0]) * nx)
x_end = int((xmax - extent[0]) / (extent[1] - extent[0]) * nx)
y_start = int((ymin - extent[2]) / (extent[3] - extent[2]) * nt)
y_end = int((ymax - extent[2]) / (extent[3] - extent[2]) * nt)

m_max = np.max(m_FT[y_start:y_end, x_start:x_end])

FT_tot = m_FT/m_max

fig, ax = plt.subplots()
linewidth = 2

# Plot the analytical derived dispersion relation
k = np.linspace(xmin, xmax, 2500)
wext = Bdc
wani = 2 * K / Ms
wex = 2*A/Ms * k**2
wc = 4*A_c/(a*a*Ms)
wnn = A_nn/Ms * k**2
wmagnon = np.sqrt((wani + wex - wnn) * (wani + wex - 2*wc + wnn))
w1 = GAMMALL * (wmagnon + wext)
w2 = GAMMALL * (wmagnon - wext)
w3 = GAMMALL * (-wmagnon + wext)
w4 = GAMMALL * (-wmagnon - wext)

ax.plot(k, w1/(2*np.pi), '--', lw=1)
ax.plot(k, w2/(2*np.pi), '--', lw=1)
ax.plot(k, w3/(2*np.pi), '--', lw=1)
ax.plot(k, w4/(2*np.pi), '--', lw=1)

# plot numerical result
ax.imshow(FT_tot, aspect='auto', origin='lower', extent=extent, cmap="inferno")

ax.axhline(GAMMALL * Bdc / (2 * np.pi), c='g', ls='--', lw=1)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.vlines([-1/dx, 1/dx], 0, fmax, 'r', '--', lw=1)
ax.set_ylabel("$f$ (Hz)")
ax.set_xlabel("$k$ (1/m)")
ax.set_title('Antiferromagnetic Spinwave Dispersion Relation')
plt.show()
