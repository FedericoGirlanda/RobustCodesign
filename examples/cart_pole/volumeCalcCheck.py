import numpy as np
from time import time

from cart_pole.model.parameters import Cartpole
from cart_pole.controllers.lqr.RoAest.utils import vol_ellipsoid, storeEllipse, ellipseVolume_convexHull
from TrajOpt_TrajStab_CMAES import roaVolComputation

sys = Cartpole("short")
Q_opt = np.diag([10., 100., .1, .1])  
R_opt = 1
RoA_path = "data/cart_pole/RoA/RoA_CMAES.csv"
roa_options = {"QN": Q_opt,
               "R": R_opt,
               "urdf": "data/cart_pole/urdfs/cartpole.urdf",
               "xG": [0,0,0,0]}

ti = time()
volume = roaVolComputation(sys, roa_options)
tf = time()
volume_convex = roaVolComputation(sys, roa_options, convex_vol_calc=True)
tf_c = time()

print("seconds that volume takes: ", (tf-ti))
print("value: ", volume)
print("seconds that volume takes:", (tf_c-tf))
print("value: ", volume_convex)