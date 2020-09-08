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

# Authors: Gilbert #

import rospy
import time
from gazebo_msgs.msg import ModelState, ModelStates

class Combination():
    def __init__(self):
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.moving()

    def moving(self):
        state = 0
        while not rospy.is_shutdown():
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            for i in range(len(model.name)):
                if model.name[i] == 'obstacle_1':
                    obstacle_1 = ModelState()
                    obstacle_1.model_name = model.name[i]
                    obstacle_1.pose = model.pose[i]
                    if abs(obstacle_1.pose.position.x - 2) < 0.05 and abs(obstacle_1.pose.position.y - 2) < 0.05:
                        state = 0

                    if state == 0:
                        obstacle_1.pose.position.x -= 0.005
                        obstacle_1.pose.position.y -= 0.012
                        if abs(obstacle_1.pose.position.x - 1.5) < 0.05 and abs(obstacle_1.pose.position.y - 0.8) < 0.05:
                            state = 1

                    elif state == 1:
                        obstacle_1.pose.position.x -= 0.01
                        obstacle_1.pose.position.y += 0.007
                        if abs(obstacle_1.pose.position.x - 0.5) < 0.05 and abs(obstacle_1.pose.position.y - 1.5) < 0.05:
                            state = 2

                    elif state == 2:
                        obstacle_1.pose.position.x -= 0.008
                        obstacle_1.pose.position.y -= 0.002
                        if abs(obstacle_1.pose.position.x + 0.3) < 0.05 and abs(obstacle_1.pose.position.y - 1.3) < 0.05:
                            state = 3

                    elif state == 3:
                        obstacle_1.pose.position.x -= 0.007
                        obstacle_1.pose.position.y -= 0.005
                        if abs(obstacle_1.pose.position.x + 1) < 0.05 and abs(obstacle_1.pose.position.y - 0.8) < 0.05:
                            state = 4

                    elif state == 4:
                        obstacle_1.pose.position.x -= 0.01
                        obstacle_1.pose.position.y += 0.007
                        if abs(obstacle_1.pose.position.x + 2) < 0.05 and abs(obstacle_1.pose.position.y - 1.5) < 0.05:
                            state = 5

                    elif state == 5:
                        obstacle_1.pose.position.x -= 0.002
                        obstacle_1.pose.position.y -= 0.007
                        if abs(obstacle_1.pose.position.x + 2.2) < 0.05 and abs(obstacle_1.pose.position.y - 0.8) < 0.05:
                            state = 6

                    elif state == 6:
                        obstacle_1.pose.position.x += 0.004
                        obstacle_1.pose.position.y -= 0.011
                        if abs(obstacle_1.pose.position.x + 1.8) < 0.05 and abs(obstacle_1.pose.position.y + 0.3) < 0.05:
                            state = 7

                    elif state == 7:
                        obstacle_1.pose.position.x += 0.003
                        obstacle_1.pose.position.y -= 0.007
                        if abs(obstacle_1.pose.position.x + 1.5) < 0.05 and abs(obstacle_1.pose.position.y + 1) < 0.05:
                            state = 8

                    elif state == 8:
                        obstacle_1.pose.position.x += 0.009
                        obstacle_1.pose.position.y += 0.007
                        if abs(obstacle_1.pose.position.x + 0.6) < 0.05 and abs(obstacle_1.pose.position.y + 0.3) < 0.05:
                            state = 9

                    elif state == 9:
                        obstacle_1.pose.position.x += 0.011
                        obstacle_1.pose.position.y -= 0.011
                        if abs(obstacle_1.pose.position.x - 0.5) < 0.05 and abs(obstacle_1.pose.position.y + 1.4) < 0.05:
                            state = 10

                    elif state == 10:
                        obstacle_1.pose.position.x += 0.006
                        obstacle_1.pose.position.y -= 0.006
                        if abs(obstacle_1.pose.position.x - 1.1) < 0.05 and abs(obstacle_1.pose.position.y + 2) < 0.05:
                            state = 11

                    elif state == 11:
                        obstacle_1.pose.position.x += 0.009
                        obstacle_1.pose.position.y += 0.01
                        if abs(obstacle_1.pose.position.x - 2) < 0.05 and abs(obstacle_1.pose.position.y + 1) < 0.05:
                            state = 12

                    elif state == 12:
                        obstacle_1.pose.position.x -= 0.008
                        obstacle_1.pose.position.y += 0.01
                        if abs(obstacle_1.pose.position.x - 1.2) < 0.05 and abs(obstacle_1.pose.position.y + 0) < 0.05:
                            state = 13

                    elif state == 13:
                        obstacle_1.pose.position.x -= 0.007
                        obstacle_1.pose.position.y += 0.005
                        if abs(obstacle_1.pose.position.x - 0.5) < 0.05 and abs(obstacle_1.pose.position.y - 0.5) < 0.05:
                            state = 14

                    elif state == 14:
                        obstacle_1.pose.position.x += 0.017
                        obstacle_1.pose.position.y += 0.008
                        if abs(obstacle_1.pose.position.x - 2.2) < 0.05 and abs(obstacle_1.pose.position.y - 1.3) < 0.05:
                            state = 15

                    elif state == 15:
                        obstacle_1.pose.position.x -= 0.002
                        obstacle_1.pose.position.y += 0.007
                        if abs(obstacle_1.pose.position.x - 2) < 0.05 and abs(obstacle_1.pose.position.y - 2) < 0.05:
                            state = 0

                    self.pub_model.publish(obstacle_1)
                    time.sleep(0.1)

def main():
    rospy.init_node('combination_obstacle_1')
    try:
        combination = Combination()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()