import numpy as np
from bs_localization_problem import Point2d, BaseStation, BSLocProblem
from visualizer import Visualizer

# base station info
base_station1 = BaseStation(1, Point2d(2, 6))
base_station2 = BaseStation(1, Point2d(0, 0))
base_station3 = BaseStation(1, Point2d(10, 2))
base_station4 = BaseStation(1, Point2d(7, 8))
bs1_measure_cov = 1.0
bs2_measure_cov = 0.1
bs3_measure_cov = 0.8
bs4_measure_cov = 0.5

# vehicle position
true_vehicle_state = Point2d(5, 3)

# noise free measurements
truth_measure1 = true_vehicle_state.distance_to(base_station1.position_)
truth_measure2 = true_vehicle_state.distance_to(base_station2.position_)
truth_measure3 = true_vehicle_state.distance_to(base_station3.position_)
truth_measure4 = true_vehicle_state.distance_to(base_station4.position_)

# generate measurements with noise
measure1 = truth_measure1 + np.random.normal(0, np.sqrt(bs1_measure_cov))
measure2 = truth_measure2 + np.random.normal(0, np.sqrt(bs2_measure_cov))
measure3 = truth_measure3 + np.random.normal(0, np.sqrt(bs3_measure_cov))
measure4 = truth_measure4 + np.random.normal(0, np.sqrt(bs4_measure_cov))

# construct problem
problem = BSLocProblem()
problem.SetInitState(Point2d(0, 0))
problem.AddObservation((base_station1, measure1, bs1_measure_cov))
problem.AddObservation((base_station2, measure2, bs2_measure_cov))
problem.AddObservation((base_station3, measure3, bs3_measure_cov))
problem.AddObservation((base_station4, measure4, bs4_measure_cov))

# calculate PEB
peb = problem.CalPEB(true_vehicle_state)

# initialize a visualizer
vis = Visualizer()
vis.AddBaseStation((base_station1, truth_measure1, measure1, bs1_measure_cov))
vis.AddBaseStation((base_station2, truth_measure2, measure2, bs2_measure_cov))
vis.AddBaseStation((base_station3, truth_measure3, measure3, bs3_measure_cov))
vis.AddBaseStation((base_station4, truth_measure4, measure4, bs4_measure_cov))
vis.SetTruthVehicleState(true_vehicle_state)
vis.SetEstimationPEB(peb)
vis.Show()

# state optimization
for step in range(100):
    problem.StateUpdateOneStep()
    vis.SetVehicleState(problem.vehicle_state_)
    vis.Show()

    # print log
    print("step {}, state {},{}".format(
        step, problem.vehicle_state_.x, problem.vehicle_state_.y))
    res = problem.CalResiduals(problem.vehicle_state_)
    print("residuals", res.T)

    if problem.IsConvergent(1e-4):
        break
