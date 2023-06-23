import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('WebAgg')
import time 
import yaml

from acrobot.utils.model_parameters import model_parameters

from acrobot.roaEstimation.obj_fcts import caprr_coopt_interface
from acrobot.roaEstimation.vis import TIrhoVerification, getEllipsePatch

robot = "acrobot"
yaml_paths = ["data/acrobot/designParams/CMA-ES_design1st.yml"]
designLabels = ["CMA-ES_design1st"]

stateIndeces = [0,1,2,3] # pos1, pos2, vel1, vel2
labels = ["pos1 [rad]", "pos2 [rad]", "vel1 [rad/s]", "vel2 [rad/s]"]
stateSlice = [stateIndeces[0], stateIndeces[2]] # projecting the state space in 2D 
goal = [np.pi,0,0,0]

for i,yaml_path in enumerate(yaml_paths):
  fig, ax = plt.subplots(figsize=(18, 8))

  ## Model parameters from yaml file for Symbolic plant
  mpar = model_parameters(filepath=yaml_path)
  design_params = {"m": mpar.m,
                  "l": mpar.l,
                  "lc": mpar.r,
                  "b": mpar.b,
                  "fc": mpar.cf,
                  "g": mpar.g,
                  "I": mpar.I,
                  "tau_max": mpar.tl}

  ## Controller parameters
  par_dict = yaml.safe_load(open(yaml_path, 'r'))
  Q = np.diag([par_dict['q11'], par_dict['q22'], par_dict['q33'], par_dict['q44']])
  R = np.eye(2)*par_dict['r11']
  Qf = np.copy(Q)

  roa_backends = ["najafi", "sos_eq", "sos_con"]
  linestyles = ["-",":","--"]
  p_buffer =[]

  for j,be in enumerate(roa_backends):
    ## RoA calculation
    najafi_evals = 100000
    roa_calc = caprr_coopt_interface(design_params=design_params,
                                    Q=Q,
                                    R=R,
                                    backend=be,
                                    najafi_evals=najafi_evals,
                                    robot = robot)
    roa_calc._update_lqr(Q=Q, R=R)
    start = time.time()
    vol, rho_f, S = roa_calc._estimate()
    stop = time.time()
    exeT = stop-start

    print("------")
    print(be + " :")
    print("rho: ", rho_f)
    print("volume: ", vol)
    print("estimation time: ", exeT)

    p = getEllipsePatch(goal[stateSlice[0]],goal[stateSlice[1]],stateSlice[0], stateSlice[1],rho_f,S, linest=linestyles[j])
    ax.add_patch(p)
    p_buffer = np.append(p_buffer,p)

  x_g = ax.scatter(goal[stateSlice[0]],goal[stateSlice[1]],color="black",marker="x", linewidths=3)
  ax.set_xlabel(labels[stateSlice[0]])
  ax.set_ylabel(labels[stateSlice[1]])
  ax.legend(handles = [x_g,p_buffer[0],p_buffer[1],p_buffer[2]], 
              labels = ["Goal state", "najafi-based sampling method", "SOS method with equality-constrained formulation", "SOS method with line search"])
  plt.suptitle("Comparison between different RoA estimation methods")
  plt.title(designLabels[i])
  plt.grid(True)
  plt.show()