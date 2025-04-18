#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import pickle
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QTimer
import pyqtgraph


class GraphSubscriber(Node):

    def __init__(self, window):
        super().__init__('graph')

        self.window = window

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/result',
            self.data_callback,
            10
        )
        self.subscription

    def data_callback(self, msg):
        self.window.receive_data(msg)


class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()

        self.setWindowTitle('Result')
        self.setGeometry(50, 50, 600, 650)

        self.ep = []
        self.data_list = []
        self.rewards = []
        self.count = 1
        self.size_ep = 0
        load_data = False

        if load_data:
            self.ep, self.data_list = self.load_data()
            self.size_ep = len(self.ep)

        self.plot()

        self.ros_subscriber = GraphSubscriber(self)
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros_subscriber,), daemon=True
        )
        self.ros_thread.start()

    def receive_data(self, msg):
        self.data_list.append(msg.data[0])
        self.ep.append(self.size_ep + self.count)
        self.count += 1
        self.rewards.append(msg.data[1])

    def plot(self):
        self.qValuePlt = pyqtgraph.PlotWidget(self, title='Average max Q-value')
        self.qValuePlt.setGeometry(0, 320, 600, 300)

        self.rewardsPlt = pyqtgraph.PlotWidget(self, title='Total reward')
        self.rewardsPlt.setGeometry(0, 10, 600, 300)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        self.show()

    def update(self):
        self.rewardsPlt.showGrid(x=True, y=True)
        self.qValuePlt.showGrid(x=True, y=True)

        self.rewardsPlt.plot(self.ep, self.data_list, pen=(255, 0, 0), clear=True)
        self.qValuePlt.plot(self.ep, self.rewards, pen=(0, 255, 0), clear=True)
        self.save_data([self.ep, self.data_list])

    def load_data(self):
        try:
            with open('graph.txt', 'rb') as f:
                x, y = pickle.load(f)
        except Exception as e:
            print('Data load error:', e)
            x, y = [], []
        return x, y

    def save_data(self, data):
        with open('graph.txt', 'wb') as f:
            pickle.dump(data, f)

    def closeEvent(self, event):
        if self.ros_subscriber is not None:
            self.ros_subscriber.destroy_node()
        rclpy.shutdown()
        event.accept()


def main():
    rclpy.init()
    app = QApplication(sys.argv)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
