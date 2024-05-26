import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple
from bs_localization_problem import Point2d, BaseStation


class Visualizer:
    def __init__(self) -> None:
        self.truth_veh_state_ = Point2d(0, 0)
        self.vehicle_state_ = Point2d(0, 0)
        self.base_stations_ = []
        self.peb_ = 1

    def SetTruthVehicleState(self, state: Point2d) -> None:
        self.truth_veh_state_ = state

    def SetVehicleState(self, state: Point2d) -> None:
        self.vehicle_state_ = state

    def AddBaseStation(self, station: Tuple[BaseStation, float]) -> None:
        """
        station: (BaseStation, truth distance, measured distance, measurement covariance)
        """
        self.base_stations_.append(station)

    def SetEstimationPEB(self, peb: float) -> None:
        self.peb_ = peb

    def Show(self) -> None:
        fig, ax = plt.subplots()
        ax.set_xlim(-1, 12)
        ax.set_ylim(-1, 12)
        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.set_title('2D Coordinate')

        # draw base stations
        for item in self.base_stations_:
            bs = item[0]
            truth_dis = item[1]
            measure_dis = item[2]
            ax.scatter(
                bs.position_.x, bs.position_.y, marker='^', s=70, c='r')
            text = 'truth dis:{}\nmea dis:{}'.format(
                round(truth_dis, 4), round(measure_dis, 4))
            ax.text(bs.position_.x, bs.position_.y, text)

        # draw truth vehicle position
        ax.scatter(
            self.truth_veh_state_.x, self.truth_veh_state_.y, marker='*', s=30, c='r')
        circle = patches.Circle((self.truth_veh_state_.x, self.truth_veh_state_.y),
                                self.peb_, edgecolor='red', facecolor='none')
        ax.add_patch(circle)

        # draw estimated vehicle position
        ax.scatter(
            self.vehicle_state_.x, self.vehicle_state_.y, marker='o', s=30, c='g')
        text = '({},{})'.format(round(self.vehicle_state_.x, 4),
                                round(self.vehicle_state_.y, 4))
        ax.text(self.vehicle_state_.x, self.vehicle_state_.y, text)

        plt.grid()
        plt.show()
