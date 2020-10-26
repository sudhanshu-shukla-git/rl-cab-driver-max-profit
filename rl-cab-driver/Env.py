# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

#locations
loc_a = 2
loc_b = 12
loc_c = 4
loc_d = 7
loc_e = 8

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(i,j) for i in range(1,m+1) for j in range(1,m+1) if i!=j ]
        self.action_space.append((0,0)) # adding for (0,0) action  for idle state 
        self.state_space =[(i,j,k) for i in range(1,m+1) for j in range(0,t) for k in range(0,d)]
        self.state_init = (1,0,0)
        self.total_time=0
        self.max_time=24*30 

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
   #     """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        n = m+t+d # total size

        loc= int(state[0]) # location
        tim=int(state[1]) # time
        day=int(state[2]) # day

        state_encod = np.zeros((n,1))
        loc_arr= np.zeros((m,1))
        time_arr = np.zeros((t,1))
        day_arr= np.zeros((d,1))
        
        loc_arr[loc-1] = 1
        time_arr[tim] = 1
        day_arr[day] = 1
        
        state_encod = np.append(loc_arr,time_arr)
        state_encod= np.append(state_encod,day_arr) # final encoded state       
        
        return state_encod

     #   return state_encod


    # Use this function if you are using architecture-2 
#     def state_encod_arch2(self, state, action):
#      """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
#        n = m+t+d+m+m
#
#        loc=state[0]
#        tim=state[1]
#        day=state[2]
#        m1=action[0]
#        m2=action[1]
#        state_encod = np.zeros((n,1))
#        print((state_encod.shape))
#        state_encod[loc]=1.0
#        state_encod[m+tim]=1.0
#        state_encod[m+t+day]=1.0
#        state_encod[m+t+d+m1]=1.0
#        state_encod[m+t+d+m+m2]=1.0
#        return state_encod
#        
#    #     return state_encod
#

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0] # current location
        #print("location",location)
        if int(location) == 1:
            requests = np.random.poisson(loc_a)
        elif int(location) == 2:
            requests = np.random.poisson(loc_b)
        elif int(location) == 3:
            requests = np.random.poisson(loc_c)
        elif int(location) == 4:
            requests = np.random.poisson(loc_d)
        elif int(location) == 5:
            requests = np.random.poisson(loc_e)
        
        if requests >15:
            requests =15
        #print("requests",requests)
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        
        actions.append((0,0))

        return possible_actions_index,actions   
   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""      
        
        curr_loc = int(state[0]) # currebnt location 
        curr_time = int(state[1]) # current time of the day
        curr_day = int(state[2])   # current day of the week     
        #print("reward_func action",action)
        pickup_loc = int(action[0])# pickup location
        drop_loc = int(action[1])  # drop location
        
        # time to reach from current to pickup location
        time_elapsed_till_pickup = Time_matrix[curr_loc - 1,pickup_loc-1,curr_time,curr_day]    

        # updated time and day
        time_next = int( (curr_loc + time_elapsed_till_pickup) % t)
        day_next = int((curr_day + (curr_loc + time_elapsed_till_pickup)//t) % d)
        
         # check for the idle action
        if pickup_loc == 0 and drop_loc == 0:
            reward = (-C) # For action (0,0)
        else:
            #reward calculation
            reward = (R * Time_matrix[pickup_loc-1,drop_loc-1,time_next,day_next]
            - C * ( Time_matrix[pickup_loc-1][drop_loc-1][time_next][day_next] 
            + Time_matrix[curr_loc-1][pickup_loc-1][curr_time][curr_day]))           
                                   
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        curr_loc = int(state[0]) # current location
        curr_time = int(state[1])# current time of the day
        curr_day = int(state[2]) # current day of the week    
        
        pickup_loc = int(action[0]) # pickup location
        drop_loc= int(action[1] )  # drop location
       
        #idle action
        if pickup_loc == 0 and drop_loc == 0:            
            time_elapsed = 1 #Wait for one hour
            time_next = (curr_time + time_elapsed) % t
            day_next = (curr_day + (curr_time + time_elapsed)//t) % d
            next_state = (curr_loc, time_next, day_next) # Do not change the location
            self.total_time = self.total_time + time_elapsed
            
            if (self.total_time >= self.max_time):
                terminal_state = 1
            else:
                terminal_state = 0
            
            terminal_state = bool(terminal_state)# returns terminal state as True or False
            return next_state, terminal_state
        
       #from current loc to pickup location    
        time_elapsed_till_pickup = Time_matrix[int(curr_loc-1),int(pickup_loc-1),int(curr_time),int(curr_day)]   
                
        time_next = (curr_time + time_elapsed_till_pickup) % t
        day_next = (curr_day + (curr_time + time_elapsed_till_pickup)//t) % d
        
        
        #from pickup to drop
        time_elapsed = Time_matrix[int(pickup_loc-1) ,int(drop_loc-1),int(time_next),int(day_next)]    
        
        self.total_time = self.total_time + time_elapsed + time_elapsed_till_pickup
        
        time_next2 = (time_next + time_elapsed)% t
        day_next = (day_next + (time_next2)//t) % d
        
        time_next = int(time_next)
        day_next = int(day_next)

        # check whether it is a terminal state
        if (self.total_time >= self.max_time):
            terminal_state = 1
            self.total_time = 0
        else:
            terminal_state =0
         
        terminal_state = bool(terminal_state)# returns terminal state as True or False
        
        next_state = (drop_loc, time_next, day_next)

        return next_state, terminal_state


    def reset(self):
        return self.action_space, self.state_space, self.state_init