import numpy as np

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
GOAL = [3,7]

def render(state):
    ''' Renders current Gridworld
    INPUT: state: current location: [row, column]
    T is the goal / terminal state
    X is the current position
    '''

    # make sure input is valid    
    if (0 > state[0] > 6) or (0 > state[1] > 9):
        print("Error! state out of range: ", state)
        return -1
    
    if state == GOAL:
        print("SUCCESS!!!")
    
    for row in range(7):
        print()
        for col in range(10):
            if [row, col] == state:
                print(" X", end='')
            elif [row, col] == GOAL:
                print(" G", end='')
            else:
                print(" *", end='')
    print("\n ___________________")
    print(" 0 0 0 1 1 1 2 2 1 0")


def renderPath(path):
    ''' Renders a path in the gridworld
    INPUT: list of states
    T is the goal / terminal state
    X is the path
    '''
    
    for row in range(7):
        print()
        for col in range(10):
            if [row, col] == GOAL:
                print(" G", end='')
            elif [row,col] in path:
                print(" X", end='')
            else:
                print(" *", end='')
    print("\n ___________________")
    print(" 0 0 0 1 1 1 2 2 1 0")


def env(state, action, render_s=False):
    ''' Returns next state and reward after taking action
    Inputs:
        state: current location: [row, column]
        action: int 0 <= action <= 3
        render: bool
    Output:
        nextState: new location: [row, column]
        reward: int (always -1 unless in goal state, then 0)
        goalReached: bool
    '''
    # make sure input is valid    
    if (0 > state[0] > 6) or (0 > state[1] > 9):
        print("Error! state out of range: ", state)
        return -1
    
    if action not in [0,1,2,3]:
        print("Invalid action:", action)
        return -1
    
    if state == GOAL:
        return state, 0, True
        
    nextState = state.copy()
    
    # take action
    if action == UP and state[0] > 0:
        nextState[0] -= 1
    elif action == DOWN and state[0] < 6:
        nextState[0] += 1
    elif action == RIGHT and state[1] < 9:
        nextState[1] += 1
    elif action == LEFT and state[1] > 0:
        nextState[1] -= 1
        
    # add wind
    wind = [0,0,0,1,1,1,2,2,1,0]
    current_wind = wind[state[1]]
    nextState[0] -= current_wind
    if nextState[0] < 0:
        nextState[0] = 0
    
    if nextState == GOAL:
        goalReached = True
        reward = 0
    else:
        goalReached = False
        reward = -1
    
    if render_s == True:
        render(state)
        render(nextState)
        
    return nextState, reward, goalReached


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
    while True:
        delta = 0
        # loop over every state
        for row in range(7):
            for col in range(10):
                value = V[row,col]
                next_values = []
                # try every action in this state
                for action in range(4):
                    nextState, reward, _ = env([row,col], action)
                    nextValue = V[nextState[0], nextState[1]] + reward
                    next_values.append(nextValue)
                V[row,col] = max(next_values)
                delta = max(delta, abs(value-V[row,col]))
        
        if delta <= maxDelta:
            break
    return V