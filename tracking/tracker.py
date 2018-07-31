import pandas as pd
import numpy as np
from numpy.linalg import inv
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt


'''
2D Kalman Filter, Tracking Position, Velocity, Acceleration & Jitter
'''

def tracker_4dof(noise=0.02, time=1.0):
    q = noise
    dt = time
    tracker = KalmanFilter(dim_x=8, dim_z=2)
    tracker.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tracker.F = np.array([[1., 0., dt, 0., 1 / 2 * (dt ** 2), 0., 1 / 6 * (dt ** 3), 0],
                          [0., 1., 0., dt, 0., 1 / 2 * (dt ** 2), 0., 1 / 6 * (dt ** 3)],
                          [0., 0., 1., 0., dt, 0., 1 / 2 * (dt ** 2), 0],
                          [0., 0., 0., 1., 0, dt, 0., 1 / 2 * (dt ** 2)],
                          [0., 0., 0., 0., 1., 0., dt, 0.],
                          [0., 0., 0., 0., 0, 1., 0., dt],
                          [0., 0., 0., 0., 0, 0., 1., 0.],
                          [0., 0., 0., 0., 0, 0., 0., 1.]])
    tracker.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0., 0., 0., 0.],])
    tracker.R = np.array([[1.0, 0],
                          [0, 1.0]])
    tracker.P = np.eye(8) * 1000.
    tracker.Q = np.array([[0., 0., q, 0., q, 0., q, 0.],
                          [0., 0., 0., q, 0., q, 0., q],
                          [q, 0., q, 0., q, 0., q, 0.],
                          [0., q, 0., q, 0., q, 0., q],
                          [q, 0., q, 0., q, 0., q, 0.],
                          [0., q, 0., q, 0., q, 0., q],
                          [q, 0., q, 0., q, 0., q, 0.],
                          [0., q, 0., q, 0., q, 0., q]])
    return tracker


def tracker_3dof(noise=0.02, time=1.0):
    q = noise
    dt = time
    tracker = KalmanFilter(dim_x=6, dim_z=2)
    tracker.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tracker.F = np.array([[1., 0., dt, 0., 1 / 2 * (dt ** 2), 0.],
                          [0., 1., 0., dt, 0., 1 / 2 * (dt ** 2)],
                          [0., 0., 1., 0., dt, 0.],
                          [0., 0., 0., 1., 0, dt],
                          [0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0, 1.]])
    tracker.H = np.array([[1., 0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0., 0.],])
    tracker.R = np.array([[1., 0],
                          [0, 1.]])
    tracker.P = np.eye(6) * 1000.
    tracker.Q = np.array([[0., 0., q, 0., q, 0.],
                          [0., 0., 0., q, 0., q],
                          [q, 0., q, 0., q, 0.],
                          [0., q, 0., q, 0., q],
                          [q, 0., q, 0., q, 0.],
                          [0., q, 0., q, 0., q]])
    return tracker

def tracker_2dof(noise=0.01, time=1.0):
    q = noise
    dt = time
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    tracker.x = np.array([0.0, 0.0, 0.0, 0.0])
    tracker.F = np.array([[1., 0., dt, 0.],
                          [0.,1., 0., dt],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]])
    tracker.H = np.array([[1., 0., 0., 0.],
                          [0., 1., 0., 0.]])
    tracker.R = np.array([[1., 0],
                          [0, 1.]])
    tracker.P = np.eye(4) * 1000.
    tracker.Q = np.array([[0, 0., q, 0.],
                          [q, 0., q, 0.],
                          [0., 0., 0., q],
                          [0., q, 0., q]])
    return tracker


