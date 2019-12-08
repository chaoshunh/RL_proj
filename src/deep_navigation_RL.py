#!/usr/bin/env python
"""
project script for course RL
This script collects dataset for RL algorithm
dataset form is as described from line 79~85

TODO: 
problem is that every move lasts too short. Need to move function moveRL() to main where can be called with lower frequency
actually it's action update frequency is too high. move should be updated frequently but action shouldn't be updated too quickly
"""
import sys, os, math, csv, time
import rospy
from std_msgs.msg import String, Empty, Float32
from drone_control.msg import filter_state, pos_status # message types
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata
import six.moves.urllib as urllib
from dqn_batch import DQNAgent
import numpy as np
import pandas as pd


# global parameters
pi = np.pi
curtpath = sys.path[0]
parenpath = os.path.join(sys.path[0], '..')

class deep_navigation:
    """
    Script for drone navigation using RL
    """
    def __init__(self,
                 dest_x=0.,
                 dest_y=0.,
                 adjust_interval=10.,
                 forward_speed=0.5,
                 destination_error=0.5,
                 col_data_mode=None, # whether is collecting data samples for RL training
                 k = 3, # state parameters of k nearest obstacles
                 dataset=None, # write to sample dataset
                 ckptload=None # load trained checkpoints
                 ):
        
        global curtpath
        global parenpath
        self.col_data_mode = col_data_mode

        self.pos_sub = rospy.Subscriber('/ardrone/predictedPose', filter_state, self.pos_update)
        self.velo_sub = rospy.Subscriber('/ardrone/navdata', Navdata, self.velo_update) 
        if not col_data_mode:
            self.state_size = 10
            self.action_size = 3
            self.agent = DQNAgent(self.state_size, self.action_size, filename=dataset)
            self.agent.load(str(parenpath + '/ckpt/' + ckptload))
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.alpha = 0.5
        self.beta = 0.7

        # original collision probability and steering direction
        self.coll = 1
        self.steer = 0
        self.line_z = 0

        # original time interval
        self.time_interval = 0. # accumulated time interval
        self.adjust_interval = adjust_interval # the time interval of adjustment towards the destination
        self.time_multi = 1000 # time multiplier from simulation to the reality
        self.destination_error = destination_error # the approximation error allowed for checking whether arrived at the destination

        # position & direction
        self.loc_x = 0.
        self.loc_y = 0.
        self.velo = 0.
        self.yaw = 0.
        self.condi = 2 # landed
        self.curr_angle = 0. # absolute yaw
        self.forward_speed = forward_speed
        self.dis_accumu = 0. # Accumulated flying distance 

        # destination
        self.dest_x = dest_x
        self.dest_y = dest_y

        # get environment obstacles coordinates
        self.envob = np.zeros([1,2])
        envFile = open(str(parenpath + '/assets/env.csv'), "r")
        reader = csv.reader(envFile)
        for item in reader:
            self.envob = np.vstack([self.envob, np.array(item).astype(float)])
        self.envob = self.envob[1:, :]

        # whether initialization
        # This step is necessary since the drone needs time to settle up in the initial state, o.w. it will not be stable while taking off
        self.ini = True
        self.w_ini = True
       
        if self.col_data_mode:

            # record performance data
            # state:
            # --------------------------------------------------------------------------------------
            # |distance and anglesbetween the drone and the obstacle (\in R^{1 * (2*k)}) | velocity | steering | position x | position y | distance till now | # trips |
            # --------------------------------------------------------------------------------------  
            # action space: 
            # velocity +, 0, -; steering left, forward, right. In total 9 actions      

            self.ac = 0
            self.reward = 0
            self.no_trip = 0
            self.write_path = str(parenpath + '/assets/' + dataset)
            
            
        
    def velo_update(self, data):
        self.velo = data.vx
        self.yaw_ardrone = data.rotZ # in degree

    def pos_update(self, data):
        """
        Update drone parameters
        """
        global pi
        factor = 6 # moveRL() would be called every 6 times of pos_update

        # check current situation (taking off? landing? etc.)
        self.condi = Sig = data.droneState
        if Sig == 3 or Sig == 4: # hovering or flying
            self.ini =  False
        elif Sig == 2 or Sig == 8:
            self.ini =  True

        # the coordinates in "drone" space and "env" space are different, the transformation is shown in ../assets/env_explain.pdf
        dis_add = np.sqrt(pow(data.y - self.loc_x, 2) + pow(-data.x - self.loc_y, 2))
        self.loc_x = data.y
        self.loc_y = - data.x
        self.yaw = data.yaw
        # print(self.yaw)
        # self.velo = data.dx
        self.droneState = data.droneState
        if self.velo >= 0.1: # make sure that the drone is moving rather than getting stuck by obstacles
            self.dis_accumu += dis_add

        if not self.ini:
            if self.col_data_mode:
                self.col_data_main()
            else:
                self.update_state()
            


    """
    function pos_act_update() and move() are combined for the situation col_data_mode = False
    """
    
    def update_state(self):
        global pi
        k = 3
        # update current state
        dists_k, angles_k = self.cal_obs_state(k = k)
        self.state = np.hstack([dists_k, angles_k, np.array([self.velo/1000, self.yaw/180*pi, self.loc_x, self.loc_y]).reshape([1,-1])])
        self.moveRL()
   
    def run_main(self):      
        # choose action from input, can have lower action update frequency
        dists, _ = self.params_obstacle()
        if np.min(dists) > 1.2 and not self.col_data_mode:
            self.ac = 1 # v = 0.5; steer = 0 
        else:
            self.ac = self.agent.act(self.state, False)
        self.action_result(action = self.ac)

    def head_to_dest_adjust(self):
        def sgn(x): return 1. if x > 0 else -1. if x < 0 else 0.
        if self.condi == 3 or self.condi == 4:    # while flying or hovering
            self.forward_speed = 0. # set forward speed to 0
            # angle of destination is available at ../assets/env_explain.pdf
            dest_angle = np.arctan((self.dest_y - self.loc_y)/(self.dest_x - self.loc_x))

            # make sure the dest_angle is in range (-pi, pi)
            if self.dest_x - self.loc_x > 0:
                dest_angle = - dest_angle
            if self.dest_x - self.loc_x < 0:
                dest_angle = pi - dest_angle if dest_angle > 0 else - dest_angle - pi

            # print("Adjust!", self.loc_x, self.loc_y, "curren angle:", self.yaw, "dest angle: ", dest_angle / pi * 180)

            if abs(self.yaw - dest_angle / pi * 180) <= 20:
                print("-------------------------direction reset done!-------------------------------")
                self.steer = 0.
                self.forward_speed = 0.5
                return False
            else:
                # print(dest_angle / pi * 180)
                # clockwilse z-;
                self.steer = - sgn(dest_angle / pi * 180 - self.yaw)
                return True
            

    """
    function action_result() and moveRL() are combined for the situation col_data_mode = True
    """

    def col_data_main(self):
        # print("true v:", self.velo, "data.yaw:", self.yaw, "ardrone_yaw:", self.yaw_ardrone)
        """
        1. update R_{t+1}
        2. write to csv s, a, R_{t+1}
        3. update current state s <- s'
        4. choose action and move to s'
        """         
        k = 3
        # update self.reward
        self.get_reward()

        if not self.w_ini:
            self.record_row = np.hstack([np.around(self.temp_record[0,:], 4), self.ac, self.reward])          
            
        if self.w_ini:
            self.w_ini = False

        # update **current** position parameters: s <- s'       
        dists_k, angles_k = self.cal_obs_state(k = k)
        self.temp_record = np.hstack([dists_k, angles_k, \
            np.array([self.velo, self.yaw, self.loc_x, self.loc_y, self.dis_accumu, self.no_trip]).reshape([1,-1])])
        # # choose action randomly
        # self.ac = np.random.randint(0, 9)
        # or: choose action from input, can have lower action update frequency
        self.ac = self.ac_input 
        self.action_result(action = self.ac)
        # print("v:", self.forward_speed, "yaw:", self.steer)
        # move
        self.moveRL()

    def get_action(self, action = None):
        self.ac_input = action

    def action_result(self, action):
        """
        # velocity +, 0, -; steering left, forward, right. In total 9 actions
        This function is used if col_data_mode = True
        """
        # # 9 actions
        # if action // 3 == 0:
        #     ad = 0.2
        #     self.forward_speed += ad if self.forward_speed < 1 - ad else 0 # increase speed (max speed: 1)
        # elif action // 3 == 2:
        #     minu = 0.4
        #     self.forward_speed -= minu if self.forward_speed > minu else 0 # decrease speed (min speed: 0.1)
        
        # if action % 3 == 0:
        #     self.steer = 1.  # left
        # elif action % 3 == 1:
        #     self.steer = 0  # forward
        # elif action % 3 == 2:
        #     self.steer = -1. # right (clockwise)

        # 3 actions
        if action % 3 == 0:
            self.steer = 1.  # left
            self.forward_speed = 0.5 
        elif action % 3 == 1:
            self.steer = 0  # forward
            self.forward_speed = 0.7
        elif action % 3 == 2:
            self.steer = -1. # right (clockwise)
            self.forward_speed = 0.5



        
    def moveRL(self):

        def sgn(x): return 1. if x > 0 else -1. if x < 0 else 0.
        global pi
        adjust_interval = self.adjust_interval
        t0 = rospy.Time.now().to_sec() * self.time_multi

        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0

        self.curr_angle = self.yaw / 180 * pi

        # take actions when it's not initial state and the drone is flying rather than landing on the ground
        if not self.ini:
            vel_msg.linear.x = self.forward_speed
            vel_msg.angular.z = self.steer
            self.velocity_publisher.publish(vel_msg)

            t1 = rospy.Time.now().to_sec() * self.time_multi
            self.time_interval += t1 - t0

    def get_reward(self):
        """
        reward gotten when 
        1. get to the destination, reward = 100 / error 
        2. collision with obstacles (distance within 0.4), reward = - 10
        """

        self.reward = -1

        # arrived at destination, reward = 100 / error
        error_x = abs(self.loc_x - self.dest_x)
        error_y = abs(self.loc_y - self.dest_y)
        if error_x <= self.destination_error and error_y <= self.destination_error:
            self.reward = 100 / (error_x + error_y)            
            self.no_trip += 1 # complete one trip, trip +1

        # restriction for action changing
        if self.ac != 1: 
            self.reward = -5
        
        # collision with obstacles
        dists, _ = self.params_obstacle()
        if min(dists) <= 0.8:
            self.reward = - 10   

        


    def params_obstacle(self):    
        global pi

        # distance to those obstacles
        dists_x = self.envob[:, 0] - self.loc_x
        dists_y = self.envob[:, 1] - self.loc_y
        dists = np.sqrt(dists_x * dists_x + dists_y * dists_y)

        # angles to those obstacles
        ob_xs = self.envob[:, 0]
        ob_ys = self.envob[:, 1]
        angles = np.arctan((ob_ys - self.loc_y)/(ob_xs - self.loc_x))
        # adjust angles
        # np.arctan(1) = pi/4, np.arctan(-1) = -pi/4, thus np.arctan range from (-pi/2, pi/2)
        # according to the angle illustration in ../assets/env_explain.pdf
        ind_adj = ob_xs - self.loc_x < 0
        angles[ind_adj] = pi/2 + angles[ind_adj]
        ind_adj = ob_xs - self.loc_x > 0
        angles[ind_adj] = - pi/2 + angles[ind_adj]
        return dists, angles

    def cal_obs_state(self, k = 3):
        """
        state data of the parameters relevant to obstacles
        1. get the angles of the k nearest obstacles
        2. get the distance to the k nearest obstacle
        """

        dists, angles = self.params_obstacle() # dist: 8 * 1 column, angles 8 * 1 column
        knear_ind = np.argpartition(dists, k)
        knear_ind = knear_ind[:k]

        dists_k = np.reshape(dists[knear_ind], [1, k])
        angles_k = np.reshape(angles[knear_ind], [1, k])

        return dists_k, angles_k # both are 1 * k array
        
    """
    other utils
    """
    def write_csv(self):
        try:
            if self.droneState == 3 or self.droneState == 7:
                with open(self.write_path, 'a+') as file_test:                   
                    writer = csv.writer(file_test)
                    writer.writerow(self.record_row)
        except:
            pass
            


def main():
    global pi

    rospy.init_node('deep_navigation', anonymous=True)

    fre = rospy.get_param('~rate') # frequency of update action, and writing to csv if train
    train = rospy.get_param('~isTrain') # whether is training
    dataset = rospy.get_param('~dataset') # sample data for training
    ckptload = rospy.get_param('~ckptload') # ckpt file name to be loaded

    
    rate = rospy.Rate(fre)  # hz, frequency of update action, and writing to csv if train
    adj_count = 25 #  count for a proper time to fly towards the destination
    adj_period = 100000
    dn = deep_navigation(dest_x=3.,
                         dest_y=-3.,
                         forward_speed=0.5,
                         destination_error=0.2,
                         col_data_mode= train,
                         dataset = dataset,
                         ckptload = ckptload
                         )

    while not rospy.is_shutdown():
        if train:
            dn.write_csv()
            ac = np.random.randint(0, 3)
            dn.get_action(action = ac)
        else:
            try:
                if adj_count == adj_period:
                    while dn.head_to_dest_adjust():
                        pass
                    if not dn.head_to_dest_adjust():
                        adj_count = 0
                else:
                    dn.run_main()
                    adj_count += 1 if adj_count + 1 <= adj_period else 0
            except:
                pass
        rate.sleep()
    


if __name__ == '__main__':
    main()
