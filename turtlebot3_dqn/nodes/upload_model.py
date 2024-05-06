#!/usr/bin/env python3

from __future__ import print_function

import rospy
from turtlebot3_dqn.srv import PtModel,PtModelRequest
import torch
import pickle
from collections import OrderedDict


def upload_model():
    rospy.wait_for_service('get_model_upload')
    
    od = OrderedDict()
    od['a'] = torch.rand(2,3)
    od['b'] = torch.rand(2,3)
    od['c'] = torch.rand(2,3)
    od['d'] = torch.rand(2,3)

    print("\nDeserialized OrderedDict:")
    for key, value in od.items():
        print(key, value)

    # Serializing the OrderedDict to bytes
    od_bytes = pickle.dumps(od)

    req = PtModelRequest()
    req.req = od_bytes
    rospy.loginfo("The request data is: %s", req)

    try:
        client = rospy.ServiceProxy('get_model_upload', PtModel)
        resp = client(req)
        print(resp)
        rospy.loginfo("The response data is: %s", resp)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == '__main__':
    try:
        upload_model()
    except rospy.ROSInterruptException:
        pass