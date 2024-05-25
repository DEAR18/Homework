import numpy as np
from bs_localization_problem import Point2d, BaseStation, BSLocProblem

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

# generate measurements
measure1 = true_vehicle_state.distance_to(
    base_station1.position_) + np.random.normal(0, np.sqrt(bs1_measure_cov))
measure2 = true_vehicle_state.distance_to(
    base_station2.position_) + np.random.normal(0, np.sqrt(bs2_measure_cov))
measure3 = true_vehicle_state.distance_to(
    base_station3.position_) + np.random.normal(0, np.sqrt(bs3_measure_cov))
measure4 = true_vehicle_state.distance_to(
    base_station4.position_) + np.random.normal(0, np.sqrt(bs4_measure_cov))

# construct problem
problem = BSLocProblem()
problem.SetInitState(Point2d(0, 0))
problem.AddObservation((base_station1, measure1, bs1_measure_cov))
problem.AddObservation((base_station2, measure2, bs2_measure_cov))
problem.AddObservation((base_station3, measure3, bs3_measure_cov))
problem.AddObservation((base_station4, measure4, bs4_measure_cov))

# state optimization
print("init state {},{}".format(problem.vehicle_state_.x, problem.vehicle_state_.y))
for step in range(100):
    problem.StateUpdateOneStep()
    print("step {}, state {},{}".format(
        step, problem.vehicle_state_.x, problem.vehicle_state_.y))
    res = problem.CalResiduals(problem.vehicle_state_)
    print("residuals", res.T)
    if problem.IsConvergent(1e-4):
        break
