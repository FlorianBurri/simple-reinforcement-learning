import matplotlib.pyplot as plt
import numpy as np
from config import UP, DOWN, RIGHT, LEFT, START, GOAL



class gridworld:
    def __init__(self):
        self.state = START
        self.path = [self.state]
        self.shape = [10,7]
        self.steps = []
    
    def reset(self):
        self.state = START
        self.path = [self.state]
        
    
    def step(self, action):
        ''' Returns next state and reward after taking action
        Inputs:
            action: int 0 <= action <= 3
        Output:
            nextState: new location: [row, column]
            reward: int (always -1 unless in goal state, then 0)
            goalReached: bool
        '''
        # make sure input is valid            
        if action not in [0,1,2,3]:
            print("Invalid action:", action)
            return -1
        
        if self.state == GOAL:
            return self.state, 0, True
            
        nextState = self.state.copy()
        
        # take action
        if action == UP and self.state[1] < 6:
            nextState[1] += 1
        
        elif action == DOWN and self.state[1] > 0:
            nextState[1] -= 1
        
        elif action == RIGHT and self.state[0] < 9:
            nextState[0] += 1
        
        elif action == LEFT and self.state[0] > 0:
            nextState[0] -= 1
            
        # add wind
        wind = [0,0,0,1,1,1,2,2,1,0]
        current_wind = wind[self.state[0]]
        nextState[1] += current_wind
        if nextState[1] > 6:
            nextState[1] = 6
        
        if nextState == GOAL:
            goalReached = True
            self.steps.append(len(self.path)+1)
            reward = 0
        else:
            goalReached = False
            reward = -1
            
        self.state = nextState
        self.path.append(nextState)
        return self.state, reward, goalReached
     
    
    def render(self):
        ''' Renders current Gridworld state
        T is the goal / terminal state
        X is the current position
        '''
    
#        # make sure input is valid    
#        if (0 > self.state[0] > 6) or (0 > self.state[1] > 9):
#            print("Error! state out of range: ", self.state)
#            return -1
        
        if self.state == GOAL:
            print("SUCCESS!!!")
        
        for y in range(6,-1,-1):
            print()
            for x in range(10):
                if [x, y] == self.state:
                    print(" X", end='')
                elif [x, y] == GOAL:
                    print(" G", end='')
                else:
                    print(" *", end='')
        print("\n ___________________")
        print(" 0 0 0 1 1 1 2 2 1 0")


    def renderPath(self):
        ''' Renders a path in the gridworld
        T is the goal / terminal state
        X is the path
        '''        
        for y in range(6,-1,-1):
            print()
            for x in range(10):
                if [x, y] == GOAL:
                    print(" G", end='')
                elif [x, y] in self.path:
                    print(" X", end='')
                else:
                    print(" *", end='')
        print("\n ___________________")
        print(" 0 0 0 1 1 1 2 2 1 0")
        
    
    def plotSteps(self):
        ''' 
        '''
        plt.plot(self.steps)
        plt.plot(moving_average(self.steps, n=len(self.steps)//20))
        plt.yscale('log')
        plt.xlabel('Episodes')
        plt.ylabel('Time steps')
        plt.title('Learning Curve')
        plt.legend(('Time steps', 'Moving average'), loc='upper right')
        plt.grid(axis='y', which='both')
        plt.show()


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = (ret[n:] - ret[:-n])/n
    ret[:n] = None 
    return ret
