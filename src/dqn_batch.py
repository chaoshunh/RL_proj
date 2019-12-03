# -*- coding: utf-8 -*-
"""
This script is used for training checkpoints if directly run.
"""
import random, csv, os, sys, argparse
from tqdm import tqdm
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from readCSV import readCSV


EPISODES = 1000
state_size = 10
action_size = 3
pi = np.pi
parenpath = os.path.join(sys.path[0], '..')

class DQNAgent:
    def __init__(self, state_size, action_size, filename):
        global memory
        self.state_size = state_size
        self.action_size = action_size
        # self.memory = deque(maxlen=2000)
        self.memory = readCSV(state_size, action_size, filename=filename)
        self.gamma = 0.5   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.loss = 10000

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state, train = True):
        if train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for i in range(len(minibatch)):
        # for state, action, reward, next_state, done in minibatch:
            row = minibatch[i]
            done = False
            
            state = np.array(row[:10]).reshape([1, -1])
            action = np.array(row[-(self.state_size+2)]).reshape([1, -1])
            reward = row[-(self.state_size+1)]

            if reward < -1:
                done = True

            target = reward # target represents the Q-value
            if not done:
                next_state = np.array(row[-self.state_size:]).reshape([1, -1])
                if self.loss <= 50:
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            
            target_f = self.model.predict(state)
            target_f[0][int(action)] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        self.loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return self.loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":    

    parser = argparse.ArgumentParser(description='Train DQN Based Drone Collision Avoidance')
    parser.add_argument('--dataset', default='traj.csv', help='choose dataset for training')
    parser.add_argument('--ckptsave', default='ckpt.h5', help='ckpt file to save in ../ckpt folder')

    args = parser.parse_args()
    agent = DQNAgent(state_size, action_size, filename = args.dataset)
    memory = agent.memory
    done = False
    batch_size = 32

    
    for e in tqdm(range(EPISODES)):
        # state = env.reset()
        state = memory[0][:state_size]
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                # Logging training loss every 10 timesteps
                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}".format(e, EPISODES, time, loss))  
                    # with open(str(parenpath + "/assets/loss_3acs.csv"), 'a+') as file_test:                   
                    #     writer = csv.writer(file_test)
                    #     # step, loss
                    #     writer.writerow(np.array([e * 500 + time, loss]))
        # if  % 10 == 0:
        if loss <= 1:
            agent.save(str(parenpath + "/ckpt/" + args.ckptsave))
