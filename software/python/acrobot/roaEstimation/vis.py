import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches

"""
Visualization functions used for RoA estimation
"""

def getEllipseParamsFromQuad(s0Idx,s1Idx,rho,S):
    """
    Returns ellipses in the plane defined by the states matching the indices s0Idx and s1Idx for funnel plotting.
    """

    ellipse_mat=np.array([  [S[s0Idx][s0Idx],S[s0Idx][s1Idx]],
                            [S[s1Idx][s0Idx],S[s1Idx][s1Idx]]])*(1/rho)
    
    #eigenvalue decomposition to get the axes
    w,v=np.linalg.eigh(ellipse_mat) 

    try:
        #let the smaller eigenvalue define the width (major axis*2!)
        width=2/float(np.sqrt(w[0]))
        height=2/float(np.sqrt(w[1]))
        #the angle of the ellipse is defined by the eigenvector assigned to the smallest eigenvalue (because this defines the major axis (width of the ellipse))
        angle=np.rad2deg(np.arctan2(v[:,0][1],v[:,0][0]))

    except:
        print("paramters do not represent an ellipse.")

    return width,height,angle

def getEllipsePatch(x0,x1,s0Idx,s1Idx,rho,S, linest = None):
    """
    just return the patches object. I.e. for more involved plots...
    x0 and x1 -> centerpoint
    """
    w,h,a=getEllipseParamsFromQuad(s0Idx,s1Idx,rho,S)
    if not linest == None:
        return patches.Ellipse((x0,x1), w, h, a, alpha=1,ec="red",facecolor="none", linestyle = linest)
    else:
        return patches.Ellipse((x0,x1), w, h, a, alpha=1,ec="red",facecolor="none")

def getEllipsePatches(x0,x1,s0Idx,s1Idx,rhoHist,S):
    p=[]
    for rhoVal in rhoHist:
        p.append(getEllipsePatch(x0,x1,s0Idx,s1Idx,rhoVal,S))
        
    return p

def plotEllipse(x0,x1,s0Idx,s1Idx,rho,S, save_to=None, show=True):
    p=getEllipsePatch(x0,x1,s0Idx,s1Idx,rho,S)
    
    fig, ax = plt.subplots()
    ax.add_patch(p)
    l=np.max([p.width,p.height])
    ax.set_xlim(x0-l/2,x0+l/2)
    ax.set_ylim(x1-l/2,x1+l/2)
    ax.grid(True)
    if not (save_to is None):
        plt.savefig(save_to)
    if show:
        plt.show()

###############################
# Funnels saving and plotting #
###############################

import os
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.spatial import ConvexHull
from matplotlib.collections import LineCollection

from acrobot.roaEstimation.probabilistic import projectedEllipseFromCostToGo, sampleFromEllipsoid, quadForm
from acrobot.simulation.simulation import Simulator
from acrobot.utils.csv_trajectory import load_trajectory

from acrobot.roaEstimation.drake_sim import DrakeStepSimulator

def TIrhoVerification(pendulum, controller, rho_i, nSimulations):
    '''
    Function to verify the time-invariant RoA estimation.

    Parameters
    ----------
    pendulum: simple_pendulum.model.pendulum_plant
        configured pendulum plant object
    controller: simple_pendulum.controllers.tvlqr.tvlqr
        configured tvlqr controller object
    rho_i: float
        rho value from the time-invariant estimation
    nSimulations: int
        number of simulations for the verification
    '''

    # Figure settings
    nStates = 4
    fig, ax = plt.subplots(nStates,
                           1,
                           figsize=(18, nStates*3),
                           sharex="all")
    fig.suptitle("Verification of RoA guarantee certificate")
    labels=["theta1 [rad]","theta2 [rad]","theta_dot1 [rad/s]","theta_dot2 [rad/s]"]

    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24

    mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
    mpl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Sampling parameters
    S_i = np.array(controller.S)
    x_goal = [np.pi,0,0,0]
    dt = 0.01
    t_final = 4

    T = np.linspace(0,t_final,num = int(t_final/dt))

    # Goals plotting
    for i in range(nStates):
        ax[i].set_xlabel("time [s]")
        ax[i].set_ylabel(labels[i])
        ax[i].grid(True)
        x_g = np.ones(len(T))*x_goal[i]
        ax[i].plot(T,x_g, color = "grey", linestyle= "dashed")

    # Sampling, simulating and verifying
    one_green = False
    one_red = False
    for j in range(1,nSimulations+1):                                                                                                              

        xBar0=sampleFromEllipsoid(S_i,rho_i) # sample new initial state inside the estimated RoA
        x_i = xBar0 + x_goal

        sim = Simulator(plant=pendulum) # init the simulation

        T, X, U = sim.simulate(t0 = 0, x0 = x_i, tf = t_final, dt = dt, controller = controller) # simulating this interval 

        # plotting the checked initial states and resulting trajectories, the color depends on the result
        xBar_f = np.array(X[-1]) - np.array(x_goal)
        finalJ = quadForm(S_i, xBar_f)

        if (np.round(finalJ,2) == 0):
            for j in range(nStates):
                succInit = ax[j].plot(T,np.array(X).T[j], color = "green")
            one_green = True
        else:
            for j in range(nStates):
                failInit = ax[j].plot(T,np.array(X).T[j], color = "red")
            one_red = True

    # managing the dynamic legend of the plot
    if (one_green and one_red):
        plt.legend(handles = [succInit,failInit], 
                    labels = ["successfull initial state","failing initial state"])
    elif ((not one_red) and one_green): 
        plt.legend(handles = [succInit], 
                    labels = ["successfull initial state"])
    else:
        plt.legend(handles = [failInit], 
                    labels = ["failing initial state"])

def getEllipseContour(S,rho,xg):
    """
    Returns a certain number(nSamples) of sampled states from the contour of a given ellipse.

    Parameters
    ----------
    S : np.array
        Matrix S that define one ellipse
    rho : np.array
        rho value that define one ellipse
    xg : np.array
        center of the ellipse

    Returns
    -------
    c : np.array
        random vector of states from the contour of the ellipse
    """
    nSamples = 1000 
    c = sampleFromEllipsoid(S,rho,rInner = 0.99) +xg
    for i in range(nSamples-1):
        xBar = sampleFromEllipsoid(S,rho,rInner = 0.99)
        c = np.vstack((c,xBar+xg))
    return c

def saveFunnel(rho, S_t, time, estMethod = "", robot = "acrobot"):

    N = len(time)
    S = S_t.value(time[0]).flatten()
    for i in range(1,N):
        S = np.vstack((S,S_t.value(time[i]).flatten()))

    log_dir = "data/"+robot+"/funnels"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir) 

    rho = np.reshape(rho, (N,1))
    csv_data = np.hstack((rho, S))

    csv_path = os.path.join(log_dir, estMethod + f"funnel-{N}.csv")
    np.savetxt(csv_path, csv_data, delimiter=',',
            header="rho,S_t", comments="")

    return csv_path

def     getEllipseFromCsv(csv_path, index):

    data = np.loadtxt(csv_path, skiprows=1, delimiter=",")

    rho = data.T[0].T[index]

    S_t = data.T[1:len(data)].T[index]
    state_dim = int(np.sqrt(len(data.T)-1))
    S_t = np.reshape(S_t,(state_dim,state_dim))

    return rho, S_t

def plotRhoEvolution(funnel_path, traj_path):
    # load trajectory data
    # trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
    # time = trajectory.T[0].T
    # x0_traj = [trajectory.T[1].T, trajectory.T[2].T]
    time, x0_traj, _ = load_trajectory(csv_path=traj_path, with_tau=True)
    N = len(time)

    # load funnel data
    funnel_data = np.loadtxt(funnel_path, skiprows=1, delimiter=",")
    rho = funnel_data.T[0]

    # plots
    fig = plt.figure()
    plt.title("rho evolution")
    ax = fig.add_subplot()
    ax.set_xlabel("Number of steps")
    ax.plot(np.arange(N),rho, color = "yellow", label = "final rho")
    ax2=ax.twinx()
    ax2.plot(np.arange(N),x0_traj.T[0],color="blue", label = "nominal traj, angle")
    ax2.plot(np.arange(N),x0_traj.T[1],color="red", label = "nominal traj, velocity")
    ax.legend(loc = "upper left")
    ax2.legend(loc = "upper right")

def plotFunnel(funnel_path, traj_path, plot_idx0=0, plot_idx1=2, ax = None):
    '''
    Function to draw a continue 2d funnel plot. This implementation makes use of the convex hull concept
    as done in the MATLAB code of the Robot Locomotion Group (https://groups.csail.mit.edu/locomotion/software.html).
    Parameters
    ----------
    rho : np.array
        array that contains the estimated rho value for all the knot points
    S: np.array
        array of matrices that define ellipses in all the knot points, from tvlqr controller.
    x0: np.array 
        pre-computed nominal trajectory
    time: np.array
        time array related to the nominal trajectory
    '''

    # load trajectory data
    trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
    time = trajectory.T[0].T
    x0 = [trajectory.T[plot_idx0 + 1].T, trajectory.T[plot_idx1 +1].T]

    # figure initialization
    zorder = 2
    funnel_color = 'red'
    if (ax == None):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(x0[0],x0[1], zorder = 3) # plot of the nominal trajectory
        zorder = 1
        funnel_color = 'green'
    
    plt.title("2d resulting Funnel")
    plt.grid(True)
    labels=["theta1 [rad]","theta2 [rad]","theta_dot1 [rad/s]","theta_dot2 [rad/s]"]
    ax.set_xlabel(labels[plot_idx0])
    ax.set_ylabel(labels[plot_idx1])
    # ax.set_xlim(-3, 4)
    # ax.set_ylim(-20, 20)

    for i in range(len(time)-1):
        (rho_i, S_i) = getEllipseFromCsv(funnel_path,i)
        (rho_iplus1, S_iplus1) = getEllipseFromCsv(funnel_path,i+1)
        S_sliced = np.array([[S_i[plot_idx0][plot_idx0], S_i[plot_idx0][plot_idx1]],[S_i[plot_idx1][plot_idx0], S_i[plot_idx1][plot_idx1]]])
        S_slicedplus1 = np.array([[S_iplus1[plot_idx0][plot_idx0], S_iplus1[plot_idx0][plot_idx1]],[S_iplus1[plot_idx1][plot_idx0], S_iplus1[plot_idx1][plot_idx1]]])
        c_prev = getEllipseContour(S_sliced,rho_i, np.array(x0).T[i]) # get the contour of the previous ellipse
        c_next = getEllipseContour(S_slicedplus1,rho_iplus1, np.array(x0).T[i+1]) # get the contour of the next ellipse
        points = np.vstack((c_prev,c_next))

        # plot the convex hull of the two contours
        hull = ConvexHull(points) 
        line_segments = [hull.points[simplex] for simplex in hull.simplices]
        ax.add_collection(LineCollection(line_segments,
                                     colors=funnel_color,
                                     linestyle='solid', zorder = zorder))
    return ax

def TVrhoVerification(funnel_path, traj_path, nSimulations, ver_idx, roaConf, plot_idx0=0, plot_idx1=2, ax_funnel = None):
    '''
    Function to verify the time-variant RoA estimation. This implementation permitts also to choose
    which knot has to be tested. Furthermore the a 3d funnel plot has been implemented.

    Parameters
    ----------
    pendulum: simple_pendulum.model.pendulum_plant
        configured pendulum plant object
    controller: simple_pendulum.controllers.tvlqr.tvlqr
        configured tvlqr controller object
    x0_t: np.array 
        pre-computed nominal trajectory
    time: np.array
        time array related to the nominal trajectory
    nSimulations: int
        number of simulations for the verification
    ver_idx: int
        knot point to be verified
    '''

    # load trajectory data
    trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
    time = trajectory.T[0].T
    x0_t = [trajectory.T[1].T, trajectory.T[2].T, trajectory.T[3].T, trajectory.T[4].T]

    # simulation init
    dt_log = time[1]-time[0]
    simulator = DrakeStepSimulator( roaConf["traj_csv"],roaConf["Q"], roaConf["R"], roaConf["Qf"], roaConf["mpar"].tl,
                                    roaConf["urdf"], roaConf["max_dt"], dt_log = dt_log, robot = roaConf["robot"])
    simulator.init_simulation()
    S_sim = simulator.tvlqr_S

    # load funnel data
    funnel_data = np.loadtxt(funnel_path, skiprows=1, delimiter=",")
    rho_t = funnel_data.T[0]
    rho_ver, S_ver = getEllipseFromCsv(funnel_path, ver_idx)
    S_ver = S_sim.value(time[ver_idx])

    # figure initialization
    fig = plt.figure()
    fig.suptitle("Verification of RoA estimation")
    ax = fig.add_subplot()
    ax.set_xlabel("time [s]")

    # plot rho evolution
    ax.plot(time[ver_idx:],rho_t[ver_idx:], color = "yellow", label = "rho evolution")
    
    first_green = True
    first_red = True
    for j in range(1,nSimulations+1):   

        print(f"Simulation n {j}...")   

        xBar0=sampleFromEllipsoid(S_ver,rho_ver) # sample new initial state inside the estimated RoA
        x_0= xBar0 + np.array(x0_t).T[ver_idx]    

        # let's simulate it
        T,X,U = simulator.simulate(x_0,ver_idx)  

        # plotting the simulated trajectories in the funnel 
        if ax_funnel != None:
            ax_funnel.plot(X[plot_idx0],X[plot_idx1], zorder = 3, color = "c")                                                                                               

        # plotting the checked initial states and resulting trajectories, the color depends on the result
        error = False
        ctg_t = []
        for i in range(ver_idx, len(time)-1):
            rho_i, S_i = getEllipseFromCsv(funnel_path, i)
            S_i = S_sim.value(time[i])
            ctg_i = quadForm(S_i,X.T[i-ver_idx] - np.array(x0_t).T[i])
            ctg_t = np.append(ctg_t,ctg_i)  
            if ctg_i > rho_i:
                error = True

        if (not error):
            if first_green:
                ax.plot(time[ver_idx:(len(time)-1)],ctg_t,color="green", label = "successfull initial state")
            else:
                ax.plot(time[ver_idx:(len(time)-1)],ctg_t,color="green")
            first_green = False
        else:
            if first_red:
                ax.plot(time[ver_idx:(len(time)-1)],ctg_t,color="red", label = "failing initial state")
            else:
                ax.plot(time[ver_idx:(len(time)-1)],ctg_t,color="red")
            first_red = False

    ax.legend()

# def funnel2DComparison(csv_pathFunnelSos, csv_pathFunnelProb, traj_path):
#     ax = plotFunnel(csv_pathFunnelProb, traj_path)
#     plotFunnel(csv_pathFunnelSos, traj_path, ax)

# def rhoComparison(csv_pathFunnelSos, csv_pathFunnelProb):
#     # load funnel data
#     funnel_data = np.loadtxt(csv_pathFunnelSos, skiprows=1, delimiter=",")
#     rho_sos = funnel_data[0].T
#     funnel_data = np.loadtxt(csv_pathFunnelProb, skiprows=1, delimiter=",")
#     rho_prob = funnel_data[0].T
#     N = len(rho_sos)

#     # plots
#     fig = plt.figure()
#     plt.title("rho evolution comparison")
#     ax = fig.add_subplot()
#     ax.set_xlabel("Number of steps")
#     ax.plot(np.arange(N),rho_sos,color = "red", label = "SOS method")
#     ax.plot(np.arange(N),rho_prob,color = "green", label = "Probabilistic Method")
#     ax.plot(np.arange(N),rho_prob-rho_sos,color = "yellow", label = "Difference")
#     ax.legend()

# def plotFunnel3d(csv_path, traj_path, ax):
#     '''
#     Function to draw a discrete 3d funnel plot. Basically we are plotting a 3d ellipse patch in each 
#     knot point.
#     Parameters
#     ----------
#     rho : np.array
#         array that contains the estimated rho value for all the knot points
#     S: np.array
#         array of matrices that define ellipses in all the knot points, from tvlqr controller.
#     x0: np.array 
#         pre-computed nominal trajectory
#     time: np.array
#         time array related to the nominal trajectory
#     ax: matplotlib.axes
#         axes of the plot where we want to add the 3d funnel plot, useful in the verification function.
#     '''

#     # choice of which funnel to plot
#     pos1_idx = 0 # pos1
#     pos2_idx = 1 #pos2
#     vel1_idx = 2 #vel1
#     vel2_idx = 3 #vel2
    
#     plot_idx0 = pos1_idx
#     plot_idx1 = vel1_idx

#     # load trajectory data
#     trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
#     time = trajectory.T[0].T
#     x0 = [trajectory.T[1].T, trajectory.T[2].T,trajectory.T[3].T, trajectory.T[4].T]

#     for i in range(len(time)):
#         (rho_i, S_i) = getEllipseFromCsv(csv_path, i)
#         # Drawing the main ellipse
#         ctg=np.asarray(S_i)
#         labels=["theta1 [rad]","theta2 [rad]","theta_dot1 [rad/s]","theta_dot2 [rad/s]"]
#         s0= plot_idx0
#         s1= plot_idx1

#         w,h,a=projectedEllipseFromCostToGo(s0,s1,[rho_i],[ctg])

#         elliIn=patches.Ellipse((x0[s0][i],x0[s1][i]), 
#                                 w[0], 
#                                 h[0],
#                                 a[0],ec="black",linewidth=1.25, color = "green", alpha = 0.1)
#         ax.add_patch(elliIn)
#         art3d.pathpatch_2d_to_3d(elliIn, z=time[i], zdir="x") # 3d plot of a patch

#     plt.title("3d resulting Funnel")
#     ax.set_xlabel("time [s]")
#     ax.set_ylabel(labels[s0])
#     ax.set_zlabel(labels[s1])
#     ax.set_xlim(0, time[-1])
#     ax.set_ylim(-6, 6)
#     ax.set_zlim(-6, 6)