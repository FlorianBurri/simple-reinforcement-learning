import numpy as np
import matplotlib.pyplot as plt
from config import START, GOAL


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


def renderStateValues(V, isActionValues=False):
    if isActionValues:
        V = V.max(axis=2)
        title = "max action Values"
    else:
        title = "State Values"
    
    fig, ax = plt.subplots(1)
    img = ax.imshow(V.T, cmap=plt.get_cmap('RdYlGn'), origin='lower')
    fig.colorbar(img)
    ax.set_title(title)
    
    # print arrows to visualise the wind
    wind = [0,0,0,1,1,1,2,2,1,0]
    for x in range(10):
        if wind[x] == 0:
            continue
        
        for y in range(0, 5, 2):
            ax.arrow(x=x, y=y+0.3, dx=0, dy=-0.6+wind[x],
                      width=0.03*wind[x],
                      head_width=0.15*wind[x],
                      head_length=0.35,
                      fc='k',
                      ec='k')
    
    # mark Start and Goal
    ax.text(x=GOAL[0],
            y=GOAL[1], 
            s='G',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=18,
            color='blue')
    
    ax.text(x=START[0],
            y=START[1],
            s='S',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=18,
            color='blue')
    
    plt.show()
