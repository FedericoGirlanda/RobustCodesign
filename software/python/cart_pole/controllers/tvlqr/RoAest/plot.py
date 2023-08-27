import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
from matplotlib import patches
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.spatial import ConvexHull
from matplotlib.collections import LineCollection

from cart_pole.controllers.tvlqr.RoAest.utils import getEllipseFromCsv, getEllipseContour, quad_form, sampleFromEllipsoid, projectedEllipseFromCostToGo


def plotRhoEvolution(funnel_path, traj_path, indeces):
    #load trajectory data
    trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
    time = trajectory.T[0].T
    x0_traj = np.array([trajectory.T[1].T, trajectory.T[2].T, trajectory.T[3].T, trajectory.T[4].T])
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
    ax2.plot(np.arange(N),x0_traj[indeces[0]],color="blue", label = f"nominal traj, idx {indeces[0]}")
    ax2.plot(np.arange(N),x0_traj[indeces[1]],color="red", label = f"nominal traj, idx {indeces[1]}")
    ax.legend(loc = "upper left")
    ax2.legend(loc = "upper right")

def plotFunnel(funnel_path, traj_path, indexes, ax = None):
    # plot indexes
    plot_idx0 = indexes[0]
    plot_idx1 = indexes[1]

    #load trajectory data
    trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
    time = trajectory.T[0].T
    x0 = [trajectory.T[plot_idx0].T, trajectory.T[plot_idx1].T]
    
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
    labels=["theta [rad]","x_cart [m]","theta_dot [rad/s]","x_cart_dot [m/s]"]
    ax.set_xlabel(labels[plot_idx0])
    ax.set_ylabel(labels[plot_idx1])
    # ax.set_xlim(-3, 4)
    # ax.set_ylim(-20, 20)

    for i in range(len(time)-1):
        (rho_i, S_i) = getEllipseFromCsv(funnel_path,i)
        (rho_iplus1, S_iplus1) = getEllipseFromCsv(funnel_path,i+1)
        S_sliced = np.array([[S_i[plot_idx0][plot_idx0], S_i[plot_idx0][plot_idx1]],[S_i[plot_idx1][plot_idx0], S_i[plot_idx1][plot_idx1]]])
        S_slicedplus1 = np.array([[S_iplus1[plot_idx0][plot_idx0], S_iplus1[plot_idx0][plot_idx1]],[S_iplus1[plot_idx1][plot_idx0], S_iplus1[plot_idx1][plot_idx1]]])
        c_prev = getEllipseContour(rho_i,S_sliced, np.array(x0).T[i]) # get the contour of the previous ellipse
        c_next = getEllipseContour(rho_iplus1,S_slicedplus1, np.array(x0).T[i+1]) # get the contour of the next ellipse
        points = np.vstack((c_prev,c_next))

        # plot the convex hull of the two contours
        hull = ConvexHull(points) 
        line_segments = [hull.points[simplex] for simplex in hull.simplices]
        ax.add_collection(LineCollection(line_segments,
                                     colors=funnel_color,
                                     linestyle='solid', zorder = zorder))
    return ax

def plotFunnel3d(csv_path, traj_path, indexes, ax, fontSize = 18, ticksSize = 16):
    '''
    Function to draw a discrete 3d funnel plot. Basically we are plotting a 3d ellipse patch in each 
    knot point.
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
    ax: matplotlib.axes
        axes of the plot where we want to add the 3d funnel plot, useful in the verification function.
    '''
    # plot indexes
    plot_idx0 = indexes[0]
    plot_idx1 = indexes[1]

    # load trajectory data
    trajectory = np.loadtxt(traj_path, skiprows=1, delimiter=",")
    time = trajectory.T[0].T
    x0 = [trajectory.T[1].T, trajectory.T[2].T]

    for i in range(len(time)):
        (rho_i, S_i) = getEllipseFromCsv(csv_path, i)
        S_i = np.array([[S_i[plot_idx0][plot_idx0], S_i[plot_idx0][plot_idx1]],[S_i[plot_idx1][plot_idx0], S_i[plot_idx1][plot_idx1]]])

        # Drawing the main ellipse
        ctg=np.asarray(S_i)
        labels=[r"$\theta$"+" [rad]",r'$\dot \theta$'+" [rad/s]"]
        s0=0
        s1=1

        w,h,a=projectedEllipseFromCostToGo(s0,s1,[rho_i],[ctg])

        elliIn=patches.Ellipse((x0[s0][i],x0[s1][i]), 
                                w[0], 
                                h[0],
                                a[0],ec="black",linewidth=1.25, color = "green", alpha = 0.1)
        ax.add_patch(elliIn)
        art3d.pathpatch_2d_to_3d(elliIn, z=time[i], zdir="x") # 3d plot of a patch

    #plt.title("3d resulting Funnel", fontsize = fontSize)
    ax.set_xlabel("time [s]", fontsize = fontSize)
    ax.set_ylabel(labels[s0], fontsize = fontSize)
    ax.set_zlabel(labels[s1], fontsize = fontSize)
    ax.tick_params('x', labelsize=ticksSize)
    ax.tick_params('y', labelsize=ticksSize)
    ax.tick_params('z', labelsize=ticksSize)
    ax.set_xlim(0, time[-1])
    ax.set_ylim(-1, 5)
    ax.set_zlim(-6, 6)
    ax.xaxis.labelpad=20
    ax.yaxis.labelpad=20
    ax.zaxis.labelpad=20

def TVfunnelVerification(sim,funnel_path, n_sim, ver_knot, ax_funnel = None, dt_sim = 0.01):
    init_knot = ver_knot
    final_knot = len(sim.T_nom)-1
    fig_test, ax_test = plt.subplots(2,2, figsize = (9, 8))
    fig_test.suptitle(f"Dynamics trajectory stabilization: simulated(blue) vs desired(orange)")
    ax_test[0][0].set_ylabel("x0(x_cart)")
    ax_test[0][1].set_ylabel("x1(theta)")
    ax_test[1][0].set_ylabel("x2(x_cart_dot)")
    ax_test[1][1].set_ylabel("x3(theta_dot)")
    ax_test[0][0].hlines(np.vstack((np.ones((len(sim.T_nom),1)),-np.ones((len(sim.T_nom),1))))*0.35,sim.T_nom[0], sim.T_nom[-1])
    fig_test1, ax_test1 = plt.subplots(1,1, figsize = (9, 8))
    ax_test1.hlines(np.vstack((np.ones((len(sim.T_nom),1)),-np.ones((len(sim.T_nom),1))))*sim.sys.fl,sim.T_nom[0], sim.T_nom[-1])
    ax_test1.set_ylabel("u(force)")
    (rho, S) = getEllipseFromCsv(funnel_path,ver_knot)
    for i in range(n_sim):
        x0 = sampleFromEllipsoid(S,rho,sim.X_nom.T[init_knot]) + sim.X_nom.T[init_knot] 
        sim.init_simulation(x0 = x0, init_knot = init_knot, dt_sim = dt_sim) 
        T_sim,X_sim,U_sim = sim.simulate()

        # Plot the results 
        ax_test[0][0].plot(T_sim, X_sim[0])
        ax_test[0][1].plot(T_sim, X_sim[1])
        ax_test[1][0].plot(T_sim, X_sim[2])
        ax_test[1][1].plot(T_sim, X_sim[3])
        ax_test1.plot(T_sim,U_sim.T)
        
    return 0