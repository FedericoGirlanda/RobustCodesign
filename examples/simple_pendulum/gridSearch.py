import numpy as np
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

from simple_pendulum.controllers.tvlqr.roa.utils import funnelVolume_convexHull
from simple_pendulum.controllers.tvlqr.roa.plot import plotFunnel, rhoComparison
from simple_pendulum.utilities.process_data import prepare_trajectory, saveFunnel
from simple_pendulum.controllers.tvlqr.roa.sos import TVsosRhoComputation
from simple_pendulum.controllers.lqr.roa.sos import SOSequalityConstrained

from simple_pendulum.trajectory_optimization.dirtrel.dirtrelTrajOpt import RobustDirtranTrajectoryOptimization, pendulum, tvlqr_controller, lqr_controller
from TrajOpt_TrajStab_CMAES import roaVolComputation

class cost_container():
    def __init__(self):
        self.max_vol = 0.001
        self.optimal_traj_path = "data/simple_pendulum/dirtrel/trajectoryOptimal_gridSearch.csv"

    def objectiveFunction(self,q11,q22,r,idx1,idx2,idx3, lengths_list,Id, volume_storage):
        
        if Id == "Controller":
            # pendulum parameters
            mpar = {"l": 0.5, 
                    "m": 0.67,
                    "b": 0.4,
                    "g": 9.81,
                    "cf": 0.0,
                    "tl": 2.5}

            # robust direct transcription parameters
            options = {"N": 51,
                    "R": r,
                    "Rl": r,
                    "Q": np.diag([q11,q22]),
                    "Ql": np.diag([q11,q22]),
                    "QN": np.eye(2)*100,
                    "QNl": np.eye(2)*100,
                    "D": 0.01*0.01, 
                    "E1": np.zeros((2,2)),
                    "x0": [0.0,0.0],
                    "xG": [np.pi, 0.0],
                    "tf0": 5,
                    "speed_limit": 7,
                    "theta_limit": 2*np.pi,
                    "time_penalization": 0, 
                    "hBounds": [0.05, 0.1]}
        # elif Id == "Design": # TODO: manage RoA failing
        #     # pendulum parameters
        #     mpar = {"l": params[1], 
        #             "m": params[0],
        #             "b": 0.35,
        #             "g": 9.81,
        #             "cf": 0.0,
        #             "tl": 3}

        #     # robust direct transcription parameters
        #     options = {"N": 51,
        #             "R": .1,
        #             "Rl": .1,
        #             "Q": np.diag([10,1]),
        #             "Ql": np.diag([10,1]),
        #             "QN": np.eye(2)*100,
        #             "QNl": np.eye(2)*100,
        #             "D": 0.2*0.2, 
        #             "E1": np.eye(2)*0.001, #np.zeros((2,2)),
        #             "x0": [0.0,0.0],
        #             "xG": [np.pi, 0.0],
        #             "tf0": 3,
        #             "speed_limit": 7,
        #             "theta_limit": 2*np.pi,
        #             "time_penalization": 0, 
        #             "hBounds": [0.05, 0.05]}

        dirtrel = RobustDirtranTrajectoryOptimization(mpar, options)
        try:
            T, X, U = dirtrel.ComputeTrajectory()
            #funnel_volume = dirtrel.l_w
            log_dir = "data/simple_pendulum/dirtrel"
            traj_data = np.vstack((T, X[0], X[1], U)).T
            traj_path = os.path.join(log_dir, "trajectory_gridSearch.csv" )
            np.savetxt(traj_path, traj_data, delimiter=',',
                        header="time,pos,vel,torque", comments="")
        except:
            funnel_volume = 0.001 # 100000
            print("DIRTREL error")
            return funnel_volume
        
        funnel_path = "data/simple_pendulum/funnels/Sosfunnel_gridSearch.csv" 
        try:
            funnel_volume = -roaVolComputation(mpar, options, traj_path, funnel_path)
        except:
            print("RoA estimation ERROR") 
            return self.max_vol

        volume_storage[(idx1*lengths_list[1] +idx2)*lengths_list[2] +idx3] = funnel_volume  

        if funnel_volume > self.max_vol:
            np.savetxt(self.optimal_traj_path, traj_data, delimiter=',',
                        header="time,pos,vel,torque", comments="")
            self.max_vol = funnel_volume

if __name__ == "__main__":  

    save_dir = "results/simple_pendulum/Design3.1Opt /gridSearch/"
    Id = "Controller" #"Design"
    cube_l = 10

    volumeObj = cost_container() 

    if Id == "Controller":
        q11_values = np.linspace(1,10,cube_l)
        q22_values = np.linspace(1,10,cube_l)
        r_values = np.linspace(0.1,1,cube_l) 

        start_time = time.time() 

        X, Y, Z = np.meshgrid(q11_values, q22_values, r_values)

        N_PROC = 3
        par_storage = []
        for idx1, q11 in enumerate(q11_values):
            for idx2, q22 in enumerate(q22_values):
                for idx3, r in enumerate(r_values):
                    par_storage.append([q11, q22, r, idx1, idx2, idx3])

        par_storage2 = []
        pp = []
        for i, p in enumerate(par_storage):
            if i > 0 and i % N_PROC == 0:
                par_storage2.append(pp)
                pp = []
            pp.append(p)
        if len(pp) > 0:
            par_storage2.append(pp)

        i = 0
        init = time.time()
        vol_array = mp.Array('d', [0]*len(q11_values)*len(q22_values)*len(r_values))
        for p in par_storage2:
            manager = mp.Manager()
            jobs = []
            for pp in p:
                process = mp.Process(target=volumeObj.objectiveFunction, args=(pp[0],
                                                            pp[1],
                                                            pp[2],
                                                            pp[3],
                                                            pp[4],
                                                            pp[5],
                                                            [len(q11_values),len(q22_values),len(r_values)],
                                                            Id,
                                                            vol_array))
                jobs.append(process)
                process.start()
            for proc in jobs:
                proc.join()
                i = i+1
                print(f"{i} jobs done...")
        end = time.time()

        search_storage = np.zeros((len(r_values)*len(q11_values)*len(q22_values),4))
        vol_storage = np.zeros((len(q11_values),len(q22_values),len(r_values)))
        for idx1, q11 in enumerate(q11_values):
            for idx2, q22 in enumerate(q22_values):
                for idx3, r in enumerate(r_values):
                    idx = (idx1*len(q22_values) +idx2)*len(r_values) +idx3
                    vol_storage[idx1,idx2,idx3] = vol_array[idx]
                    search_storage[idx,:] = [par_storage[idx][0],par_storage[idx][1],par_storage[idx][2],vol_array[idx]]

        total_time = int((time.time() - start_time)/60) # mins
        print(f"The process took: {total_time} mins")

        # Save the obtained data
        data_path = os.path.join(save_dir, "data"+Id+".csv" )
        np.savetxt(data_path, search_storage, delimiter=',',
                header="q11,q22,r,Vol", comments="")

        # Read the saved data
        data_readed = np.array(pd.read_csv(data_path))

        # Find the maximum values
        max_idx = np.where(data_readed[:,3] == data_readed[:,3].max())[0][0]
        Q_opt = np.diag([data_readed[max_idx,0],data_readed[max_idx,1]])
        R_opt = [data_readed[max_idx,2]]
        print("The optimal Q is: ", Q_opt)
        print("The optimal R is: ", R_opt)

        # Save the obtained data
        result_storage = [data_readed[max_idx,0],data_readed[max_idx,1],data_readed[max_idx,2],data_readed[:,3].max(), total_time]
        result_path = os.path.join(save_dir, "result"+Id+".csv" )
        np.savetxt(result_path, np.array([result_storage]), delimiter=',',
                header="q11_opt,q22_opt,r_opt,Vol_opt, time(mins)", comments="")

        # Volume computation for saving the funnel
        # mpar = {"l": 0.5, 
        # "m": 0.67,
        # "b": 0.4,
        # "g": 9.81,
        # "cf": 0.0,
        # "tl": 2.5}
        # funnel_path = "results/Design3Opt/gridSearch/Sosfunnel_GS.csv"
        # traj_path = volumeObj.optimal_traj_path
        # roa_options = {"N": 51,
        #             "Q": np.diag([data_readed[max_idx,0],data_readed[max_idx,1]]),
        #             "R": data_readed[max_idx,2]}
        # volume = roaVolComputation(mpar, roa_options, traj_path, funnel_path)
        # print("The optimal volume is: ", volume)