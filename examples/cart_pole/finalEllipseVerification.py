import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch 
import pandas
from tqdm import tqdm

from pydrake.all import Linearize, \
                        LinearQuadraticRegulator, \
                        DiagramBuilder, \
                        AddMultibodyPlantSceneGraph, \
                        Parser

from cart_pole.model.parameters import Cartpole
from cart_pole.utilities.process_data import prepare_trajectory
from cart_pole.simulation.simulator import StepSimulator
from cart_pole.controllers.tvlqr.RoAest.utils import getEllipseFromCsv
from cart_pole.controllers.lqr.RoAest.plots import plot_ellipse, get_ellipse_patch
from cart_pole.controllers.lqr.RoAest.utils import sample_from_ellipsoid

from generateUrdf import generateUrdf

def arrow(x,y,ax,n):
    d = len(x)//(n+1)    
    ind = np.arange(d,len(x),d)
    for i in ind:
        ar = FancyArrowPatch ((x[i-1],y[i-1]),(x[i],y[i]), 
                              arrowstyle='->', mutation_scale=20)
        ax.add_patch(ar)

# Optimized simulation environment
traj_path = "results/cart_pole/optCMAES_167332/trajectoryOptimal_CMAES.csv" 
funnel_path = "results/cart_pole/optCMAES_167332/RoA_CMAES.csv"
label = "RTC"
sys = Cartpole("short")
sys.fl = 6
old_Mp = sys.Mp
old_lp = sys.lp
sys.Mp = 0.227
l_ratio = (sys.lp/old_lp)**2
sys.Jp = sys.Jp*l_ratio + (sys.Mp-old_Mp)*(sys.lp**2)
urdf_path = generateUrdf(sys.Mp, sys.lp, sys.Jp)
Q_f = np.diag([100,100,100,100])
R_f = np.array([5.01])
xG = [0,0,0,0]
(rho_f, S_f) = getEllipseFromCsv(funnel_path, -1)

# Create plant and drake lqr controller from urdf
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0)
Parser(plant).AddModelFromFile(urdf_path)
plant.Finalize()
# Compute lqr controller
tilqr_context = plant.CreateDefaultContext()
input_i = plant.get_actuation_input_port().get_index()
output_i = plant.get_state_output_port().get_index()
plant.get_actuation_input_port().FixValue(tilqr_context, [0])
tilqr_context.SetContinuousState(xG)
linearized_cartpole = Linearize(plant, tilqr_context, input_i, output_i,
                                equilibrium_check_tolerance=1e-3) 
(Kf, Sf) = LinearQuadraticRegulator(linearized_cartpole.A(), linearized_cartpole.B(), Q_f, R_f)


# Verification of the TI-RoA
print("Verification...")
indexes = (0,2) # Meaningful values (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
n_verifications = 100
dt = 0.01
t = 8

N = int(t/dt)
X = np.zeros((N,4))
U = np.zeros((N,1))
T = np.zeros((N,1))
labels = [r"$x_{cart}$ [m]",r"$\theta$ [rad]",r"$\dot x_{cart}$ [m/s]",r"$\dot \theta$ [rad/s]"]
p = get_ellipse_patch(indexes[0], indexes[1], xG,rho_f,S_f,linec="green")  
fig, ax = plt.subplots(figsize=(18,18))
ax.add_patch(p)
goal = ax.scatter(X[-1][indexes[0]],X[-1][indexes[1]],color="black",marker="x", s=150, linewidths=20)
for j in tqdm(range(n_verifications)):  
    x_0 = sample_from_ellipsoid(indexes,S_f,rho_f,r_i = 0.99)
    X[0][indexes[0]] = x_0[0]
    X[0][indexes[1]] = x_0[1]
    for i in range(N-1):
        U[i+1] = np.clip(-Kf.dot(X[i]), -sys.fl, sys.fl)
        f,df = sys.continuous_dynamics(X[i], U[i+1])
        X[i+1] = X[i] + dt*f
        T[i+1] = T[i] + dt

    # coloring the checked initial states depending on the result    
    if (round(np.asarray(X).T[0][-1],2) == 0.00 and round(np.asarray(X).T[1][-1],2) == 0.00 and round(np.asarray(X).T[2][-1],2) == 0.00 and round(np.asarray(X).T[3][-1],2) == 0.00):
        greenDot = ax.scatter(X[0][indexes[0]],X[0][indexes[1]],color="green",marker="o", s=100)
        ax.plot(X.T[indexes[0]],X.T[indexes[1]],color="green")
        redDot = None
    else:
        redDot = ax.scatter(X[0][indexes[0]],X[0][indexes[1]],color="red",marker="o", s=100)
        ax.plot(X.T[indexes[0]],X.T[indexes[1]],color="red")
ticksSize = 30
fontSize = 30
ax.tick_params(axis='both', which='major', labelsize=ticksSize)
ax.set_xlabel(labels[indexes[0]], fontsize = fontSize)
ax.set_ylabel(labels[indexes[1]], fontsize = fontSize)
if (not redDot == None):
    ax.legend(handles = [goal,greenDot,redDot], 
                labels = ["goal state","successfull initial state","failing initial state"], fontsize = fontSize, loc = "upper left")
else: 
    ax.legend(handles = [goal,greenDot], 
                labels = ["goal state","successfull initial state"], fontsize = fontSize, loc = "upper left")
#plt.title("Verification of RoA guarantee certificate")
plt.grid(True)
plt.show()