from scipy.spatial.transform import Rotation as R
from scipy import linalg
from scipy.special import gamma, factorial
import numpy as np

def directSphere(d,r_i=0,r_o=1):
    """
    Implementation: Krauth, Werner. Statistical Mechanics: Algorithms and Computations. Oxford Master Series in Physics 13. Oxford: Oxford University Press, 2006. page 42
    """
    # vector of univariate gaussians:
    rand=np.random.normal(size=d)
    # get its euclidean distance:
    dist=np.linalg.norm(rand,ord=2)
    # divide by norm
    normed=rand/dist
    
    # sample the radius uniformly from 0 to 1 
    rad=np.random.uniform(r_i,r_o**d)**(1/d)
    # the r**d part was not there in the original implementation.
    # I added it in order to be able to change the radius of the sphere
    # multiply with vect and return
    return normed*rad

def sampleFromEllipsoid(S,rho,rInner=0,rOuter=1):
    lamb,eigV=np.linalg.eigh(S/rho) 
    d=len(S)
    xy=directSphere(d,r_i=rInner,r_o=rOuter) #sample from outer shells
    T=np.linalg.inv(np.dot(np.diag(np.sqrt(lamb)),eigV.T)) #transform sphere to ellipsoid (refer to e.g. boyd lectures on linear algebra)
    return np.dot(T,xy.T).T

def quadForm(M,x):
    """
    Helper function to compute quadratic forms such as x^TMx
    """
    return np.dot(x,np.dot(M,x))

def projectedEllipseFromCostToGo(s0Idx,s1Idx,rho,M):
    """
    Returns ellipses in the plane defined by the states matching the indices s0Idx and s1Idx for funnel plotting.
    """
    print("moved this to vis.py -> use the function defined there.")
    ellipse_widths=[]
    ellipse_heights=[]
    ellipse_angles=[]
    
    #loop over all values of rho
    for idx, rho in enumerate(rho):
        #extract 2x2 matrix from S
        S=M[idx]
        ellipse_mat=np.array([[S[s0Idx][s0Idx],S[s0Idx][s1Idx]],
                              [S[s1Idx][s0Idx],S[s1Idx][s1Idx]]])*(1/rho)
        
        #eigenvalue decomposition to get the axes
        w,v=np.linalg.eigh(ellipse_mat) 

        try:
            #let the smaller eigenvalue define the width (major axis*2!)
            ellipse_widths.append(2/float(np.sqrt(w[0])))
            ellipse_heights.append(2/float(np.sqrt(w[1])))

            #the angle of the ellipse is defined by the eigenvector assigned to the smallest eigenvalue (because this defines the major axis (width of the ellipse))
            ellipse_angles.append(np.rad2deg(np.arctan2(v[:,0][1],v[:,0][0])))
        except:
            continue
    return ellipse_widths,ellipse_heights,ellipse_angles

def volEllipsoid(rho,M):
    """
    Calculate the Volume of a Hyperellipsoid
    Volume of the Hyperllipsoid according to https://math.stackexchange.com/questions/332391/volume-of-hyperellipsoid/332434
    Intuition: https://textbooks.math.gatech.edu/ila/determinants-volumes.html
    Volume of n-Ball https://en.wikipedia.org/wiki/Volume_of_an_n-ball
    """
    
    # For a given hyperellipsoid, find the transformation that when applied to the n Ball yields the hyperellipsoid
    lamb,eigV=np.linalg.eigh(M/rho) 
    A=np.dot(np.diag(np.sqrt(lamb)),eigV.T) #transform ellipsoid to sphere
    detA=np.linalg.det(A)
    
    # Volume of n Ball (d dimensions)
    d=M.shape[0] # dimension 
    volC=(np.pi**(d/2))/(gamma((d/2)+1))

    # Volume of Ellipse
    volE=volC/detA
    return volE

class probTIROA:
    """
    class for probabilistic RoA estimation for linear (or linearized) systems under (infinite horizon) TILQR control.

    Takes a configuration dict and requires passing a callback function in which the simulation is done.
    The callback function returns the result of the simulation as a boolean (True = Success)
    
    the conf dict has the following structure

        roaConf={   "x0Star": <goal state to stabilize around>,
                    "xBar0Max": <bounds that define the first (over) estimate of the RoA to sample from>,
                    "S": <cost to go matrix TODO change this to V for other stabilizing controllers>
                    "nSimulations": <number of simulations>
                    }

    TODO: generalize for non LQR systems -> V instead of S
    """
    def __init__(self,roaConf,simFct):
        self.x0Star=roaConf["x0Star"]
        self.xBar0Max=roaConf["xBar0Max"]
        self.S=roaConf["S"]
        self.nSims=roaConf["nSimulations"]
        self.simClbk=simFct

        self.rhoHist=[]
        self.simSuccessHist=[]

        rho0=quadForm(self.S,self.xBar0Max)
        self.rhoHist.append(rho0)

    def doEstimate(self):
        for sim in range(self.nSims):
            x0Bar=sampleFromEllipsoid(self.S,self.rhoHist[-1])  # sample initial state from previously estimated RoA
            JStar0=quadForm(self.S,x0Bar)                       # calculate cost to go
            x0=self.x0Star+x0Bar                                # error to absolute coords

            simSuccess=self.simClbk(x0)

            if not simSuccess:
                self.rhoHist.append(JStar0)
            else:
                self.rhoHist.append(self.rhoHist[-1])

            self.simSuccessHist.append(simSuccess)

        return self.rhoHist,self.simSuccessHist

class probTVROA:
    """
    class for probabilistic RoA estimation of linear time variant systems under (finite horizon) TVLQR control
    takes a configuration dict and a step simulation function. 

    the roaConf dict has the following structure:
    
        roaConf={nSimulations:<int>,nKnotPoints:<int>,rhof:<float>,rho00:<float>,S:<np.array>}

    the step function should be a callback in which a piecewise simulation of the closed loop dynamics is implemented.
    it takes as an input the knot point until which to simulate.
    it should return a short dict with a simulation result and the deviation from the nominal trajectory in error coordinates xBar
        
        e.g.:

            def step(knotPoint):

                <Propagate dynamics to timestep of next knotpoint. only consider x passed above when launching new sim >

                if simSuccess:
                    return True,xBark
                else
                    return False,xBark

    Additionally a callback function has to be implemented to start a new simulation:

        e.g.:

            def newSim(xBar):
                < prepare new simulation with x=x00+xBar >


    Would probably make sense to put all simulation related information into a class and have the step and newSim functions as member functions.
    
    Then the entire process of RoA estimation could look like this:

        1. create simulation class object. 
        2. initialize tvroa estimation routine. pass relevant information from simulation class (roaConf) and name of callback (step) here.
        3. Do Roa estimation. for model evaluation /simulation, call the callback function previously defined
    """

    def __init__(self,roaConf, simulator):
        self.nSimulations = roaConf["nSimulations"]   # number of simulations
        self.rho00 = roaConf["rho00"]                 # big initial guess for rho00
        self.rho_f = roaConf["rho_f"]                 # fixed size of rhof that determines the size of the last ellipsoid around the end of the trajectory
        self.simulator = simulator
        self.timeStark= self.simulator.T_nom          # nominal t's
        self.xStark= self.simulator.X_nom             # nominal x's
        self.nEvalPoints = len(self.timeStark)        # n of knot points

        # array to store the evolution of rho defined by failed simulations
        self.rhoHist=np.ones(self.nEvalPoints)*self.rho00 # init of rhoHist and set initial guess
        self.rhoHist[-1]=self.rho_f                        # set final rho to const value

        # also store the cost to go evolution for those simulations that were successfull.
        self.ctgHist=np.ones(self.nEvalPoints)*np.inf 

        # Max successfull simulations
        self.maxSuccSimulations = self.nSimulations/2

    def doEstimate(self):
        self.simulator.init_simulation()                      # simulation initialization
        self.S = self.simulator.tvlqr_S                       # S matrix from the tvlqr controller

        for l in range(1,self.nEvalPoints):  # the trajectory has nKnotpoints-1 intervals or piecewise simulations
            k = (self.nEvalPoints-l-1)       # going backward, from the last interval to the first
            kPlus1 = (self.nEvalPoints-l)
            Sk = self.S.value(self.timeStark[k])
            SkPlus1 = self.S.value(self.timeStark[kPlus1])

            for j in range(self.nSimulations): 
                xBark=sampleFromEllipsoid(Sk,self.rhoHist[k])   # sample new initial state
                xk = xBark + self.xStark[k]
                
                self.ctgHist[k]= quadForm(Sk,xBark)
                termReason=0 # suppose a successful result for the simulation

                T_sim, X_sim, U_sim =self.simulator.simulate(xk,k,kPlus1) # simulation of the desired interval
                xkPlus1 = X_sim.T[-1]

                xBarkPlus1=xkPlus1-self.xStark[kPlus1]               
                self.ctgHist[kPlus1]= quadForm(SkPlus1,xBarkPlus1) # final cost to go calculation

                # is it inside the next ellipse?
                if self.ctgHist[kPlus1] > self.rhoHist[kPlus1]: # no, shrinking
                    termReason=1
                    self.rhoHist[k] = min(self.ctgHist[k], self.rhoHist[k])
                    self.maxSuccSimulations = self.nSimulations/2
                else:
                    self.maxSuccSimulations = self.maxSuccSimulations-1
                    
                if self.maxSuccSimulations == 0: # enough successes
                    break

                print(f"knot point {k}, simulation {j}")
                print("rhoHist :")
                print(self.rhoHist)
                print("termination reason:"+str(termReason))
                print("---")

        return self.rhoHist, self.ctgHist