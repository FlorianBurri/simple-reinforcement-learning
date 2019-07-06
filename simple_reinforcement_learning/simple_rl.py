import numpy as np
from time import time
import environment
from helpers import eGreedy, renderStateValues


# Defining the Reinforcement Learning algorithms

def dynamicProgramming(maxDelta=0):
    V = np.zeros((10,7))
    env = environment.gridworld()
    while True:
        delta = 0
        # loop over every state
        for y in range(7):
            for x in range(10):
                value = V[x, y]
                next_values = []
                # try every action in this state
                for action in range(4):
                    # for dynamicPogramming we need to be able to change
                    # the state of the environment manualy:
                    env.state = [x,y]
                    nextState, reward, _ = env.step(action)
                    nextValue = V[tuple(nextState)] + reward
                    next_values.append(nextValue)
                    
                V[x, y] = max(next_values)
                delta = max(delta, abs(value-V[x, y]))
        
        if delta <= maxDelta:
            break
    return V


def sarsa(alpha, epsilon, episodes, env, printRate=0):
    Q = np.zeros((10,7,4))
    for episode in range(episodes):
        env.reset()
        t = 0
        goalReached = False
        state = env.state
        action = eGreedy(Q[tuple(env.state)], epsilon)
        while(not goalReached):
            _, reward, goalReached = env.step(action)
            nextAction = eGreedy(Q[tuple(env.state)], epsilon)
            # update action Values
            Q[tuple(state)][action] += alpha * (reward +
                                 Q[tuple(env.state)][nextAction] -
                                 Q[tuple(state)][action])
            state = env.state
            action = nextAction
            t += 1
        
        if ((printRate != 0) and
                ((episode % printRate) == 0 or
                episode == episodes-1)):
            env.renderPath()
    
    return Q


def qLearning(alpha, epsilon, episodes, env, printRate=0):
    Q = np.zeros((10,7,4))
    for episode in range(episodes):
        env.reset()
        t = 0
        goalReached = False
        while(not goalReached):
            prevState = env.state
            action = eGreedy(Q[tuple(prevState)], epsilon)
            _, reward, goalReached = env.step(action)
            Q[tuple(prevState)][action] += alpha * (reward +
                                 np.max(Q[tuple(env.state)]) - 
                                 Q[tuple(prevState)][action])
            t += 1
        
        if ((printRate != 0) and
                ((episode % printRate) == 0 or
                episode == episodes-1)):
            env.renderPath()
    
    return Q


def nStepSarsa(alpha, epsilon, n, episodes, env, printRate=0):
    Q = np.zeros((10,7,4))
    for episode in range(episodes):
        env.reset()
        action = eGreedy(Q[tuple(env.state)], epsilon)
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
                    action = eGreedy(Q[tuple(state)], epsilon)
                    A.append(action)
            
            tUpdate = t - n + 1
            if tUpdate >= 0:
                G = np.sum(R[tUpdate+1 : tUpdate+n+1])
                if not goalReached:
                    G += Q[tuple(env.state)][action]
                
                Q[tuple(S[tUpdate])][A[tUpdate]] += alpha * (
                        G - Q[tuple(S[tUpdate])][A[tUpdate]])
            t += 1
            if tUpdate == T-1:
                break
    
    return Q


def main(): 
    ##### Dynamic Programming #####
    dt = time()
    V = dynamicProgramming(maxDelta=0)
    dt = (time()-dt)*1000
    
    # show results
    print("Dynamic Programing:")
    print("time = {:.0f}ms".format(dt))
    renderStateValues(V)

    input('Press Enter to continue...')
    print('_'*30 + '\n')
    
    ##### Sarsa #####
    dt = time()
    env = environment.gridworld()
    Q = sarsa(alpha=0.5, epsilon=0.1, episodes=150, env=env)
    dt = (time()-dt)*1000
    
    # show results
    print("Sarsa:")
    print("time = {:.0f}ms".format(dt))
    renderStateValues(Q, isActionValues=True)
    env.plotSteps()
        
    input('Press Enter to continue...')
    print('_'*30 + '\n')
    
    
    ##### Q-Learning #####
    dt = time()
    env = environment.gridworld()
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
    env = environment.gridworld()
    Q = nStepSarsa(alpha=0.1, epsilon=0.1, n=5, episodes=150, env=env)
    dt = (time()-dt)*1000
    
    # show results
    print("N-step Sarsa:")
    print("time = {:.0f}ms".format(dt))
    renderStateValues(Q, isActionValues=True)
    env.plotSteps()
        
    input('Press Enter to continue...')
    print('_'*30 + '\n')


if __name__ == "__main__":
    main()
