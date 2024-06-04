import random
import math
import numpy as np
from collections import deque

def get_rando_delta_wt():
    delta_weight = max(0.75, random.random()) - 0.75
    return delta_weight

class WaterSimulator():
    def __init__(self):
        air_time = 20
        self.i = 0
        self.weight_queue = deque([(0.0, 0.0)] * air_time)


    def get_outpouring(self, theta):
        """
        Outpouring can be estimated with theta, where angle is proportional to rate of flow
        outpouring: flow runs from 0 to 0.1g/step from 30deg to 90deg in a cubic relationship with 5% random
        """
        rand100percent = (random.random() * 2 - 1)
        max_flow = 1
        deg30 = np.pi/6
        deg90 = np.pi/2
        rand5percent = 0.05 * rand100percent
        if theta < deg30:
            outpouring = 0
        elif theta > deg90:
            outpouring = 1 + 1 * rand5percent
        else:
            anglesize = (theta - deg30) / (deg90 - deg30)
            outpouring = anglesize ** 1.2 * max_flow
        return outpouring
        
    def get_horizontal_speed(self, theta):
        """
        Horizontal speed varies from 0 - 0.01 mm/step in a quadratic relationship 
        with 1-20% random in a linear relationship wrt theta
        from 0 to 90 deg, and after 90, it is symmetric about 90
        """
        deg90 = np.pi/2
        max_speed = 0.01
        if theta < 0:
            hor_speed = 0
            return hor_speed
        elif theta > deg90:
            theta = np.pi - theta
        anglesize = theta / deg90
        randomsize = anglesize * (20-1) + 1 # a number from 1-20
        hor_speed = anglesize ** 2 * max_speed
        hor_speed = hor_speed * randomsize/100 * random.random()
        return hor_speed
    
    def get_landing_x(self, x, z, theta, hor_speed, radius):
        # t = np.sqrt(2 * (z-radius*np.sin(theta)) / 9810)
        # landing_x = x - hor_speed * t - radius*np.cos(theta)
        landing_x = x - radius*np.cos(theta) - 35
        # self.i += 1
        # if self.i == 1000:
        #     print("landing_x", x, z, landing_x)
        #     self.i = 0
        return landing_x

    def get_weight(self, x, z, theta, radius):
        outpouring = self.get_outpouring(theta)
        hor_speed = self.get_horizontal_speed(theta)
        landing_x = self.get_landing_x(x, z, theta, hor_speed, radius)
        if abs(landing_x) < radius * 0.9:
            return outpouring, 0
        elif radius * 0.9 < abs(landing_x) < radius * 1.1:
            spill_percent = (abs(landing_x) / radius - 0.9) / 0.2
            spill = outpouring * spill_percent
            return outpouring-spill, spill
        else:
            return 0, outpouring
        
    def step(self, x, z, theta, radius):
        self.weight_queue.append(self.get_weight(x,z,theta,radius))
        # self.i += 1
        # if self.i == 1000:
        #     # print(x,z,theta,self.weight_queue[0])
        #     self.i = 0
        return self.weight_queue.popleft()
    
if __name__ == "__main__":
    sim = WaterSimulator()
    x = 0
    z = 70
    theta = -0.05
    radius = 25
    # for _ in range(1000):
    #     weight, spill = sim.step(x,z,theta,radius)
    #     print(weight, spill)
    print("starting")
    import os
    print("Current working directory:", os.getcwd())
    with open('water_test.txt', 'w') as file:
        for _ in range(1000):
            weight, spill = sim.step(x,z,theta,radius)
            x+=0.01
            z+= 0.01
            theta+=0.002
            file.write(f"{x}, {z}, {theta}, {weight}, {spill} \n")
    print("finished")
