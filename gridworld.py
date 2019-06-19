import numpy as np

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
GOAL = [3,7]


class environment:
    def __init__(self):
        self.state = [3, 0]
        self.path = [self.state]

    
    def reset(self):
        self.state = [3, 0]
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
        if action == UP and self.state[0] > 0:
            nextState[0] -= 1
        
        elif action == DOWN and self.state[0] < 6:
            nextState[0] += 1
        
        elif action == RIGHT and self.state[1] < 9:
            nextState[1] += 1
        
        elif action == LEFT and self.state[1] > 0:
            nextState[1] -= 1
            
        # add wind
        wind = [0,0,0,1,1,1,2,2,1,0]
        current_wind = wind[self.state[1]]
        nextState[0] -= current_wind
        if nextState[0] < 0:
            nextState[0] = 0
        
        if nextState == GOAL:
            goalReached = True
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
    
        # make sure input is valid    
        if (0 > self.state[0] > 6) or (0 > self.state[1] > 9):
            print("Error! state out of range: ", self.state)
            return -1
        
        if self.state == GOAL:
            print("SUCCESS!!!")
        
        for row in range(7):
            print()
            for col in range(10):
                if [row, col] == self.state:
                    print(" X", end='')
                elif [row, col] == GOAL:
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
        for row in range(7):
            print()
            for col in range(10):
                if [row, col] == GOAL:
                    print(" G", end='')
                elif [row,col] in self.path:
                    print(" X", end='')
                else:
                    print(" *", end='')
        print("\n ___________________")
        print(" 0 0 0 1 1 1 2 2 1 0")



def eGreedy(actionValues, epsilon):
    '''Choose action based on action values
    e-greedy
    '''
    if np.random.rand() < epsilon:
        # chose random action
        action = np.random.randint(4)
    
    else:
        options = []
        maxVal = np.max(actionValues)
        for i in range(4):
            if maxVal == actionValues[i]:
                options.append(i)
        action = np.random.choice(options)
    
    return action


def dynamicProgramming(maxDelta=0):
    V = np.zeros((7,10))
    env = environment()
    while True:
        delta = 0
        # loop over every state
        for row in range(7):
            for col in range(10):
                value = V[row,col]
                next_values = []
                # try every action in this state
                for action in range(4):
                    # for dynamicPogramming we need to be able to change
                    # the state of the environment manualy:
                    env.state = [row,col]
                    nextState, reward, _ = env.step(action)
                    nextValue = V[nextState[0], nextState[1]] + reward
                    next_values.append(nextValue)
                    
                V[row,col] = max(next_values)
                delta = max(delta, abs(value-V[row,col]))
        
        if delta <= maxDelta:
            break
    return V