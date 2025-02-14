from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import neelskyrmion, show_field
import matplotlib.pyplot as plt

# create the world
cellsize = (1e-9, 1e-9, 0.4e-9)
world = World(cellsize)

# create the ferromagnet
magnet = Ferromagnet(world, Grid(size=(128, 64, 1)))
magnet.enable_demag = False
magnet.msat = 580e3
magnet.aex = 15e-12
magnet.ku1 = 0.8e6
magnet.anisU = (0, 0, 1)
magnet.alpha = 0.2
magnet.dmi_tensor.set_interfacial_dmi(3.2e-3)

# set and relax the initial magnetization
magnet.magnetization = neelskyrmion(
    position=(64e-9, 32e-9, 0), radius=10e-9, charge=-1, polarization=1
)
magnet.minimize()

# add a current
magnet.xi = 0.3
magnet.jcur = (1e12, 0, 0)
magnet.pol = 0.4

world.timesolver.run(2e-10)

show_field(magnet.magnetization)    
