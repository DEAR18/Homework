import numpy as np
from typing import Tuple


class Point2d:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __add__(self, other):
        new_x = self.x + other.x
        new_y = self.y + other.y
        return Point2d(new_x, new_y)

    def __sub__(self, other):
        new_x = self.x - other.x
        new_y = self.y - other.y
        return Point2d(new_x, new_y)

    def distance_to(self, other) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx**2 + dy**2)


class BaseStation:
    def __init__(self, id: int, position: Point2d) -> None:
        self.id_ = id
        self.position_ = position


class BSLocProblem:
    def __init__(self) -> None:
        self.vehicle_state_ = Point2d(0, 0)
        self.observations_ = []
        self.state_update_history_ = []

    def Print(self) -> None:
        print(self.vehicle_state_)
        print(self.vehicle_state_.shape)
        print(type(self.vehicle_state_))

    def SetInitState(self, init_state: Point2d) -> None:
        self.vehicle_state_ = init_state

    def AddObservation(self, observation: Tuple[BaseStation, float, float]) -> None:
        """
        observation: (BaseStation, measured distance, measurement covariance)
        """
        self.observations_.append(observation)

    def GetMeasureCov(self) -> np.ndarray:
        if len(self.observations_) == 0:
            return None
        obs_num = len(self.observations_)
        cov = np.zeros((obs_num, obs_num))
        for i in range(len(self.observations_)):
            cov[i][i] = self.observations_[i][2]
        return cov

    def CalResiduals(self, state: Point2d) -> np.ndarray:
        if len(self.observations_) == 0:
            return None
        obs_num = len(self.observations_)
        res = np.zeros((obs_num, 1))
        for i in range(len(self.observations_)):
            bs = self.observations_[i][0]
            meature_dis = self.observations_[i][1]
            predict_dis = state.distance_to(bs.position_)
            res[i][0] = meature_dis - predict_dis
        return res

    def CalJacobian(self, state: Point2d) -> np.ndarray:
        if len(self.observations_) == 0:
            return None
        obs_num = len(self.observations_)
        jacobian = np.zeros((obs_num, 2))
        for i in range(len(self.observations_)):
            bs = self.observations_[i][0]
            distance = state.distance_to(bs.position_)
            jx = (state.x - bs.position_.x) / (distance + 1e-10)
            jy = (state.y - bs.position_.y) / (distance + 1e-10)
            jacobian[i] = [jx, jy]
        return jacobian

    def StateUpdateOneStep(self) -> None:
        jacobian = self.CalJacobian(self.vehicle_state_)
        residual = self.CalResiduals(self.vehicle_state_)
        measure_cov = self.GetMeasureCov()
        measure_cov_inv = np.linalg.inv(measure_cov)
        hessian = jacobian.T @ measure_cov_inv @ jacobian
        hessian_inv = np.linalg.inv(hessian)
        delta_state = hessian_inv @ jacobian.T @ measure_cov_inv @ residual
        self.vehicle_state_ = self.vehicle_state_ + \
            Point2d(delta_state[0][0], delta_state[1][0])

        self.state_update_history_.append(
            Point2d(delta_state[0][0], delta_state[1][0]))
        if (len(self.state_update_history_) > 3):
            self.state_update_history_ = self.state_update_history_[1:]

    def IsConvergent(self, threshold: float) -> bool:
        sum_state_update = Point2d(0, 0)
        for item in self.state_update_history_:
            sum_state_update = sum_state_update + item
        if (np.sqrt(sum_state_update.x ** 2 + sum_state_update.y ** 2) < threshold):
            return True
        else:
            return False
