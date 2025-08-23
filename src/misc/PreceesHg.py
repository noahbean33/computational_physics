""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright Guangliang He, 2024
    Please respect copyright & acknowledge our work."""

# PrecessHg.py: Hg Precession; written by Guangliang He

from typing import Tuple, Union
import numpy as np
import math
from scipy.special import ellipkinc
import matplotlib.pyplot as plt

# noinspection PyPep8Naming
class Schwarzschild:
    """A class for Schwarzschild space-time in GR"""

    def __init__(self, radius: float):
        """
        initialize a Schwarzschild space-time with its radius
        The unit of input should be consistent with the speed
        of light is 1.
        """
        self.r_s = radius

    def g(self, x: np.ndarray) -> np.ndarray:
        """
        take an array of N coordinates
        x[:,n]  = (t[n], r[n], theta[n], phi[n])
        return the metric as an array of 4x4xN
        The convention is g_tt is positive
        """
        D, N = x.shape
        metric = np.zeros((4, 4, N))  # 4x4xN array of np.float64
        r = x[1, :]
        theta = x[2, :]
        st = np.sin(theta)
        metric[0, 0, :] = 1-self.r_s*np.reciprocal(r)
        metric[1, 1, :] = -np.reciprocal(metric[0, 0, :])
        metric[2, 2, :] = -r*r
        metric[3, 3, :] = metric[2, 2]*st*st

        return metric

    def trajectory(self, l_over_e) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        compute light-like trajectory(ies) for the given
        angular momentum to energy ratio
        """
        a = self.r_s/l_over_e
        a2 = a*a

        if a2-4/27 < -2*np.finfo(float).eps:
            return self.__trajectory_A(a)

        if a2-4/27 > 2*np.finfo(float).eps:
            return self.__trajectory_B(a)

        # a is fixed at sqrt(4/27)
        return (np.hstack((np.full(300, self.r_s*3/2).reshape(-1, 1),
                           np.linspace(0, 2*math.pi, 300).reshape(-1, 1))),
                self.__trajectory_C_1(),
                self.__trajectory_C_2())

    def __trajectory_C_2(self):
        """
        internal method for calculate type C-2 light-like trajectory
        """
        N = 1000
        eps = 100*np.finfo(float).eps
        # u from 0 to 2/3, r from infinity to 3r_s/2
        # we cut off at 10r_s
        rVec = np.linspace(10, 3/2+eps, N).reshape(-1, 1)
        uVec = 1/rVec
        temp = np.sqrt(uVec+1/3)
        phiVec = np.log((1-temp)/(temp+1))

        uVec = np.vstack((uVec, np.flip(uVec[:-1])))
        phiVec = np.vstack((phiVec, 2*phiVec[-1]-np.flip(phiVec[:-1])))

        return np.hstack((self.r_s/uVec, phiVec))

    def __trajectory_C_1(self):
        """
        internal method for calculate type C-1 light-like trajectory
        """
        N = 1000
        eps = 10*np.finfo(float).eps
        # u from 2/3 to infinity, r from 3r_s/2 to 0
        rVec = np.linspace(0+eps, 3/2-eps, N).reshape(-1, 1)
        uVec = 1/rVec
        temp = np.sqrt(uVec+1/3)
        phiVec = np.log((temp-1)/(temp+1))

        uVec = np.vstack((np.flip(uVec[1:]), uVec))
        phiVec = np.vstack((2*phiVec[0]-np.flip(phiVec[1:]), phiVec))

        return np.hstack((self.r_s/uVec, phiVec))

    def __trajectory_B(self, a):
        """
        internal method for calculate type B light-like trajectory
        """

        # auxiliary variable
        alpha = 2*math.acosh(math.sqrt(27/4)*a)

        # the real root
        u1 = (1-2*math.cosh(alpha/3))/3

        # parameters
        n = (u1*(3*u1-2))**(1/4)
        n2 = n*n
        m = (1-(3*u1-1)/(2*n2))/2

        # number of steps
        N = 1000

        # since the only real root of p(u) is u1 < 0, then
        # p(u) > 0 for u in [0, inf)
        # we will  calculate the trajectory from r = 0 to r = 10rs
        rVec = np.linspace(0, 10, N).reshape(-1, 1)
        uVec = 1/rVec[1:]
        thetaVec = np.arccos((n2-(uVec-u1))/(n2+(uVec-u1)))
        phiVec = ellipkinc(thetaVec, m)/n

        return np.hstack((self.r_s/uVec, phiVec))

    def __trajectory_A(self, a):
        """
        internal method for calculate type A light-like trajectories
        """

        if a > math.sqrt(4/27):
            raise ValueError('Bad input a for type A trajectory')

        # auxiliary variable
        psi = 2*math.asin(math.sqrt(27/4)*a)

        # roots, u1 > u2 > 0 > u3
        u1 = (1+2*math.cos(psi/3))/3
        u2 = (1+2*math.cos((psi-2*math.pi)/3))/3
        u3 = (1+2*math.cos((psi+2*math.pi)/3))/3

        return (self.__trajectory_A_1(np.array([u1, u2, u3])),
                self.__trajectory_A_2(np.array([u1, u2, u3])))

    def __trajectory_A_1(self, uRoots):
        """
        internal method for calculate light-like trajectories of
        type A 1, scattering trajectory with u in (0, u2]
        return Tx2 array of [r, phi]

        The input is a 3x1 array of roots for p(u), in descending order
        """

        m = (uRoots[1]-uRoots[2])/(uRoots[0]-uRoots[2])
        n = 2/math.sqrt(uRoots[0]-uRoots[2])

        # number of steps
        N = 10000

        # first segment from uMin to u2
        uMin = uRoots[1]/10
        uVec = np.linspace(uMin, uRoots[1], N).reshape(-1, 1)
        thetaVec = np.arcsin(np.sqrt((uVec-uRoots[2])/(uRoots[1]-uRoots[2])))
        phiVec = n*ellipkinc(thetaVec, m)

        # second segment from u2 to uMin
        # from symmetry, it is flipped image of the first segment
        # remove the last point to avoid doubling it
        # uVec[0] = uMin
        # uVec[N-1] = u2
        # uVec[N+i] = uVec[N-i-2]
        # uVec[2*N-2] = uVec[0]
        # phiVec[0] = phi0
        # phiVec[N-1] = phi1
        # phiVec[N+i] = phi1 + (phi1-phiVec[N-i-2])
        uVec = np.vstack((uVec, np.flip(uVec[:-1])))
        phiVec = np.vstack((phiVec, 2*phiVec[-1]-np.flip(phiVec[:-1])))

        return np.hstack((self.r_s/uVec, phiVec))

    def __trajectory_A_2(self, uRoots):
        """
        internal method for calculate light-like trajectories of
        type A 2, scattering trajectory with u in [u1, infinity)
        return Tx2 array of [r, phi]

        The input is a 3x1 array of roots for p(u), in descending order
        """

        m = (uRoots[1]-uRoots[2])/(uRoots[0]-uRoots[2])
        n = 2/math.sqrt(uRoots[0]-uRoots[2])

        # number of steps
        N = 10000

        # we do the second segment first, from u1 to uMax
        uMax = uRoots[0]*10
        uVec = np.linspace(uRoots[0], uMax, N).reshape(-1, 1)
        thetaVec = np.arctan(np.sqrt((uVec-uRoots[0])/(uRoots[0]-uRoots[1])))
        phiVec = n*ellipkinc(thetaVec, m)

        # second segment from uMax to u1
        # from symmetry, it is flipped image of the first segment
        # remove the last point to avoid doubling it
        # uVec[0] = uMax
        # uVec[N-1] = u1
        # uVec[N+i] = uVec[N-i-2]
        # uVec[2*N-2] = uVec[0]
        # phiVec[0] = phi0
        # phiVec[N-1] = phi1
        # phiVec[N+i] = phi1 + (phi1-phiVec[N-i-2])
        uVec = np.vstack((np.flip(uVec[1:]), uVec))
        phiVec = np.vstack((2*phiVec[0]-np.flip(phiVec[1:]), phiVec))

        return np.hstack((self.r_s/uVec, phiVec))


def plot_polar_trajectory(traj: np.ndarray, figure, grid_spec=None) -> None:
    if grid_spec is None:
        ax = figure.add_subplot(projection='polar')
    else:
        ax = figure.add_subplot(grid_spec, projection='polar')
    ax.plot(traj[:, 1], traj[:, 0])


if __name__ == "__main__":
    r_s = 1  # what's the unit?
    black_hole = Schwarzschild(r_s)
    a_param = math.sqrt(4/27)*(1-0.001)

    trajectory = black_hole.trajectory(r_s/a_param)
    fig = plt.figure()

    if isinstance(trajectory, Tuple):
        num_traj = len(trajectory)
        gs = fig.add_gridspec(num_traj, 1)
        for i in range(len(trajectory)):
            plot_polar_trajectory(trajectory[i], fig, gs[i, 0])
    else:
        plot_polar_trajectory(trajectory, fig)

    plt.show()
