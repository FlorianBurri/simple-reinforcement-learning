import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.patches import Rectangle
from time import time


UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
START = [3,0]
GOAL = [3,7]


class environment:
    def __init__(self):
        self.state = START
        self.path = [self.state]
        self.shape = [7,10]
        self.steps = []
    
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


#def renderStateValues(V, isActionValues=False):
#    if isActionValues:
#        V = V.max(axis=2)
#        title = "max action Values"
#    else:
#        title = "State Values"
#    
#    xd, yd = np.gradient(V)
#    plt.figure()
#    plt.title(title)
#    p = plt.imshow(V, cmap=plt.get_cmap('RdYlGn'))
##    plt.quiver(np.arange(0,10,1),np.arange(0,7,1),yd,xd,scale=100)
#    plt.colorbar(p)
#    
#    # print arrows to visualise the wind
#    wind = [0,0,0,1,1,1,2,2,1,0]
#    for x in range(10):
#        if wind[x] == 0:
#            continue
#        
#        for y in range(2,7,2):
#            plt.arrow(x=x, y=y-0.3, dx=0, dy=0.6-wind[x],
#                      width=0.03*wind[x],
#                      head_width=0.15*wind[x],
#                      head_length=0.35,
#                      fc='k',
#                      ec='k')
#    # highlight goal
#    Rectangle(xy=(1,2), width=3, height=1, fill = False,
#                          edgecolor ='black',linewidth=1)
#    plt.show()

def renderStateValues(V, isActionValues=False):
    if isActionValues:
        V = V.max(axis=2)
        title = "max action Values"
    else:
        title = "State Values"
    
    xd, yd = np.gradient(V)
    fig, ax = plt.subplots(1)
    img = ax.imshow(V, cmap=plt.get_cmap('RdYlGn'))
    fig.colorbar(img)
    ax.set_title(title)
    
    # print arrows to visualise the wind
    wind = [0,0,0,1,1,1,2,2,1,0]
    for x in range(10):
        if wind[x] == 0:
            continue
        
        for y in range(2,7,2):
            ax.arrow(x=x, y=y-0.3, dx=0, dy=0.6-wind[x],
                      width=0.03*wind[x],
                      head_width=0.15*wind[x],
                      head_length=0.35,
                      fc='k',
                      ec='k')
    
    # mark Start and Goal
    ax.text(x=GOAL[1],
            y=GOAL[0], 
            s='G',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=18,
            color='blue')
    
    ax.text(x=START[1],
            y=START[0],
            s='S',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=18,
            color='blue')
    
    plt.show()


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = (ret[n:] - ret[:-n])/n
    ret[:n] = None 
    return ret


# Defining the Reinforcement Learning algorithms

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




def sarsa(alpha, epsilon, episodes, env, printRate=0):
    Q = np.zeros((7,10,4))
    for episode in range(episodes):
        env.reset()
        t = 0
        goalReached = False
        state = env.state
        action = eGreedy(Q[env.state[0],env.state[1]], epsilon)
        while(not goalReached):
            _, reward, goalReached = env.step(action)
            nextAction = eGreedy(Q[env.state[0],env.state[1]], epsilon)
            # update action Values
            Q[state[0], state[1], action] += alpha * (reward +
                                 Q[env.state[0], env.state[1], nextAction]
                                 - Q[state[0], state[1], action])
            state = env.state
            action = nextAction
            t += 1
        
        if ((printRate != 0) and
                ((episode % printRate) == 0 or
                episode == episodes-1)):
            env.renderPath()
    
    return Q


def qLearning(alpha, epsilon, episodes, env, printRate=0):
    Q = np.zeros((7,10,4))
    for episode in range(episodes):
        env.reset()
        t = 0
        goalReached = False
        while(not goalReached):
            prevState = env.state
            action = eGreedy(Q[prevState[0], prevState[1]], epsilon)
            _, reward, goalReached = env.step(action)
            Q[prevState[0], prevState[1], action] += alpha * (reward +
                                 np.max(Q[env.state[0], env.state[1]]) - 
                                 Q[prevState[0], prevState[1], action])
            t += 1
        
        if ((printRate != 0) and
                ((episode % printRate) == 0 or
                episode == episodes-1)):
            env.renderPath()
    
    return Q


def nStepSarsa(alpha, epsilon, n, episodes, env, printRate=0):
    Q = np.zeros((7,10,4))
    for episode in range(episodes):
        env.reset()
        action = eGreedy(Q[env.state[0], env.state[1]], epsilon)
        R = [0]
        S = [env.state]
        A = [action]
        t = 0
        T = 1000000
        updateComplete = False
        goalReached = False
        while(not updateComplete):
            if not goalReached:
                state, reward, goalReached = env.step(action)
                R.append(reward)
                S.append(state)
                if goalReached:
                    T = t + 1
                else:
                    action = eGreedy(Q[state[0], state[1]], epsilon)
                    A.append(action)
            
            tUpdate = t - n + 1
            if tUpdate >= 0:
                G = np.sum(R[tUpdate+1 : tUpdate+n+1])
                if not goalReached:
                    G += Q[env.state[0], env.state[1], action]
                
                Q[S[tUpdate][0],S[tUpdate][1],A[tUpdate]] += alpha * (
                        G - Q[S[tUpdate][0],S[tUpdate][1],A[tUpdate]])
            t += 1
            if tUpdate == T-1:
                break
    
    return Q


# ****************************
# Testing the algorighms:
 
##### Dynamic Programming #####
dt = time()
V = dynamicProgramming(maxDelta=0)
dt = (time()-dt)*1000

# show results
print("Dynamic Programing:")
print("time = {:.0f}ms".format(dt))
renderStateValues(V)
#%%
input('Press Enter to continue...')
print('_'*30 + '\n')

##### Sarsa #####
dt = time()
env = environment()
Q = sarsa(alpha=0.5, epsilon=0.1, episodes=150, env=env)
dt = (time()-dt)*1000

# show results
print("Sarsa:")
print("time = {:.0f}ms".format(dt))
renderStateValues(Q, isActionValues=True)
env.plotSteps()

#plt.plot(np.cumsum(env.steps), range(len(env.steps)))
#plt.show()

input('Press Enter to continue...')
print('_'*30 + '\n')


##### Q-Learning #####
dt = time()
env = environment()
Q = qLearning(alpha=0.5, epsilon=0.1, episodes=150, env=env)
dt = (time()-dt)*1000

# show results
print("Q-Learning:")
print("time = {:.0f}ms".format(dt))
renderStateValues(Q, isActionValues=True)
env.plotSteps()

input('Press Enter to continue...')
print('_'*30 + '\n')

##### N-step Sarsa #####
dt = time()
env = environment()
Q = nStepSarsa(alpha=0.1, epsilon=0.1, n=5, episodes=150, env=env)
dt = (time()-dt)*1000

# show results
print("N-step Sarsa:")
print("time = {:.0f}ms".format(dt))
renderStateValues(Q, isActionValues=True)
env.plotSteps()

#plt.plot(np.cumsum(env.steps), range(len(env.steps)))
#plt.show()

input('Press Enter to continue...')
print('_'*30 + '\n')


