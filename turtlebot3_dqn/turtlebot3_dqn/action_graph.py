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

import signal
import sys
import threading
import time

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtWidgets import QWidget
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class Ros2Subscriber(Node):

    def __init__(self, qt_thread):
        super().__init__('progress_subscriber')
        self.qt_thread = qt_thread

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/get_action',
            self.get_array_callback,
            10
        )

    def get_array_callback(self, msg):
        data = list(msg.data)

        self.qt_thread.signal_action0.emit(0)
        self.qt_thread.signal_action1.emit(0)
        self.qt_thread.signal_action2.emit(0)
        self.qt_thread.signal_action3.emit(0)
        self.qt_thread.signal_action4.emit(0)

        if data[0] == 0:
            self.qt_thread.signal_action0.emit(100)
        elif data[0] == 1:
            self.qt_thread.signal_action1.emit(100)
        elif data[0] == 2:
            self.qt_thread.signal_action2.emit(100)
        elif data[0] == 3:
            self.qt_thread.signal_action3.emit(100)
        elif data[0] == 4:
            self.qt_thread.signal_action4.emit(100)

        if len(data) >= 2:
            self.qt_thread.signal_total_reward.emit(str(round(data[-2], 2)))
            self.qt_thread.signal_reward.emit(str(round(data[-1], 2)))


class Thread(QThread):

    signal_action0 = pyqtSignal(int)
    signal_action1 = pyqtSignal(int)
    signal_action2 = pyqtSignal(int)
    signal_action3 = pyqtSignal(int)
    signal_action4 = pyqtSignal(int)
    signal_total_reward = pyqtSignal(str)
    signal_reward = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.node = None

    def run(self):
        self.node = Ros2Subscriber(self)
        rclpy.spin(self.node)
        self.node.destroy_node()


class Form(QWidget):

    def __init__(self, qt_thread):
        super().__init__(flags=Qt.Widget)
        self.qt_thread = qt_thread
        self.setWindowTitle('Action State')

        layout = QGridLayout()

        self.pgsb1 = QProgressBar()
        self.pgsb1.setOrientation(Qt.Vertical)
        self.pgsb1.setValue(0)
        self.pgsb1.setRange(0, 100)

        self.pgsb2 = QProgressBar()
        self.pgsb2.setOrientation(Qt.Vertical)
        self.pgsb2.setValue(0)
        self.pgsb2.setRange(0, 100)

        self.pgsb3 = QProgressBar()
        self.pgsb3.setOrientation(Qt.Vertical)
        self.pgsb3.setValue(0)
        self.pgsb3.setRange(0, 100)

        self.pgsb4 = QProgressBar()
        self.pgsb4.setOrientation(Qt.Vertical)
        self.pgsb4.setValue(0)
        self.pgsb4.setRange(0, 100)

        self.pgsb5 = QProgressBar()
        self.pgsb5.setOrientation(Qt.Vertical)
        self.pgsb5.setValue(0)
        self.pgsb5.setRange(0, 100)

        self.label_total_reward = QLabel('Total reward')
        self.edit_total_reward = QLineEdit('')
        self.edit_total_reward.setDisabled(True)
        self.edit_total_reward.setFixedWidth(100)

        self.label_reward = QLabel('Reward')
        self.edit_reward = QLineEdit('')
        self.edit_reward.setDisabled(True)
        self.edit_reward.setFixedWidth(100)

        self.label_left = QLabel('Left')
        self.label_front = QLabel('Front')
        self.label_right = QLabel('Right')

        layout.addWidget(self.label_total_reward, 0, 0)
        layout.addWidget(self.edit_total_reward, 1, 0)
        layout.addWidget(self.label_reward, 2, 0)
        layout.addWidget(self.edit_reward, 3, 0)

        layout.addWidget(self.pgsb1, 0, 4, 4, 1)
        layout.addWidget(self.pgsb2, 0, 5, 4, 1)
        layout.addWidget(self.pgsb3, 0, 6, 4, 1)
        layout.addWidget(self.pgsb4, 0, 7, 4, 1)
        layout.addWidget(self.pgsb5, 0, 8, 4, 1)

        layout.addWidget(self.label_left, 4, 4)
        layout.addWidget(self.label_front, 4, 6)
        layout.addWidget(self.label_right, 4, 8)

        self.setLayout(layout)

        qt_thread.signal_action0.connect(self.pgsb1.setValue)
        qt_thread.signal_action1.connect(self.pgsb2.setValue)
        qt_thread.signal_action2.connect(self.pgsb3.setValue)
        qt_thread.signal_action3.connect(self.pgsb4.setValue)
        qt_thread.signal_action4.connect(self.pgsb5.setValue)
        qt_thread.signal_total_reward.connect(self.edit_total_reward.setText)
        qt_thread.signal_reward.connect(self.edit_reward.setText)

    def closeEvent(self, event):
        if hasattr(self.qt_thread, 'node') and self.qt_thread.node is not None:
            self.qt_thread.node.destroy_node()
        rclpy.shutdown()
        event.accept()


def run_qt_app(qt_thread):
    app = QApplication(sys.argv)
    form = Form(qt_thread)
    form.show()
    app.exec_()


def main():
    rclpy.init()
    qt_thread = Thread()
    qt_thread.start()
    qt_gui_thread = threading.Thread(target=run_qt_app, args=(qt_thread,), daemon=True)
    qt_gui_thread.start()

    def shutdown_handler(sig, frame):
        print('shutdown')
        qt_thread.node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        shutdown_handler(None, None)


if __name__ == '__main__':
    main()
