#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
import csv

class TopicSubscriber:
    def __init__(self):
        rospy.init_node('topic_subscriber', anonymous=True)
        self.topic_name = '/result'
        self.csv_filename = '/home/khinggan/my_research/ros_frl/ros1_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/data.csv'
        self.data = []

        rospy.Subscriber(self.topic_name, Float64, self.callback)

    def callback(self, data):
        rospy.loginfo("Received data: %s", data.data)
        self.data.append(data.data)
        self.save_to_csv()

    def save_to_csv(self):
        with open(self.csv_filename, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['data'])
            writer.writerows([[item] for item in self.data])
        rospy.loginfo("Data saved to %s", self.csv_filename)

if __name__ == '__main__':
    try:
        subscriber = TopicSubscriber()
        rospy.spin()  # Keeps the node running until shutdown
    except rospy.ROSInterruptException:
        pass
