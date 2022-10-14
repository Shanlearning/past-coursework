### MDP Value Iteration and Policy Iteration
#import os
#os.chdir("C:\\Users\\zhong\\Dropbox\\statistics\\CS 790\\assignment1\\assignment1_coding")
#pip install -r requirements.txt


import numpy as np
import gym
import time
from lake_envs import *
import random

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy,k, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.
	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
    """ 
    # YOUR IMPLEMENTATION HERE #
    trails = []
    epsilon = 1/(5*k)
    if k>5:
        epsilon = 0
    for i in range(1000*nS):
        s = random.choice(range(0,nS))
        t = 0
        history = []
        terminal = False
        while terminal == False:
            epsilon_policy = policy
            for temp in range(len(policy)):
                if random.uniform(0,1) <= epsilon:
                    epsilon_policy[temp] =  np.random.choice( range(len(P[temp])), 1)[0]
            a = epsilon_policy[s]
            next_s = np.random.choice( range(len(P[s][a])), 1 ,p= [item[0] for item in P[s][a]])[0]
            r = P[s][a][next_s][2]
            history.append([t,s,a,r])
            terminal = P[s][a][next_s][3]
            t = t+1
            s = P[s][a][next_s][1]
        V = dict()
        t_H = len(history)
        for timestep in history:
            if timestep[1] not in V:
                t = timestep[0] 
                V[timestep[1]] = sum(np.power(gamma,np.array(range(t_H-t)))*np.array([item[3] for item in history[t:t_H]]))
        trails.append(V) 
    value_function = np.zeros(nS)
    for s in range(0,nS):
        temp = []
        for V in trails:
            if s in V:
                temp.append(V[s])
        if len(temp) > 0:
            value_function[s] =np.mean(temp)
    return value_function


def policy_improvement(P, nS, nA, value_from_policy,k , policy, gamma=0.9):
    """Given the value function from policy improve the policy.
    
    Parameters
    ----------
    P, nS, nA, gamma:
    	defined at beginning of file
    value_from_policy: np.ndarray
    	The value calculated from the policy
    policy: np.array
    	The previous policy.
    
    Returns
    -------
    new_policy: np.ndarray[nS]
    	An array of integers. Each integer is the optimal action to take
    	in that state according to the environment dynamics and the
    	given value function.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    new_policy = np.zeros(nS, dtype=int)
    Q = dict()
    for s in range(0,nS):
        Q[s] = {}
        max_a_val = -9999
        for a in P[s]:        
            Q[s][a] = max([item[2] for item in P[s][a]] + gamma * sum([item[0] * value_from_policy[item[1]] for item in P[s][a]]))
            if Q[s][a] >= max_a_val:
                max_a_val = Q[s][a]
                max_a = a        
        new_policy[s] = max_a
    ############################
    return new_policy

def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.
    
    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    
    Parameters
    ----------
    P, nS, nA, gamma:
    	defined at beginning of file
    tol: float
    	tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    diff = 9999
    ############################
    # YOUR IMPLEMENTATION HERE #
    k = 1
    while (diff >tol):
        policy = policy_improvement(P, nS, nA, value_function, policy,k)
        value_from_policy = policy_evaluation(P, nS, nA, policy,k)
        diff = max(np.absolute(value_function - value_from_policy))
        value_function = value_from_policy
        k = k+1
        print('value function: ')
        print(value_function)
        print('policy: ')
        print(policy)
        print('diff: ')
        print(diff)
    ############################
    return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.
    
    Parameters:
    ----------
    P, nS, nA, gamma:
    	defined at beginning of file
    tol: float
    	Terminate value iteration when
    		max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """
    
    value_function = np.zeros(nS)
    keep_search = True
    ############################
    # YOUR IMPLEMENTATION HERE #
    while keep_search:
        new_value_function = value_function.copy()
        for s in range(0,nS):
            Q_s_a = []
            for a in range(0,nA):
                Q_s_a.append( max([item[2] for item in P[s][a]]) + gamma * sum([item[0] * value_function[item[1]] for item in P[s][a]])  )
            new_value_function[s] = max(Q_s_a)
        keep_search = max(np.absolute(value_function - new_value_function)) > tol
        value_function = new_value_function.copy()
    policy =np.zeros(nS, dtype=int)
    Q = {}
    for s in range(0,nS):
        Q[s] = {}
        max_q = -9999
        for a in P[s]:
            Q[s][a] = max([item[2] for item in P[s][a]]) + gamma * sum([item[0] * value_function[item[1]] for item in P[s][a]])
            if Q[s][a]> max_q:
                max_q = Q[s][a]
                max_a = a
        policy[s] = int(max_a)
    print('value function: ')
    print(value_function)
    print('policy: ')
    print(policy)
    ############################
    return value_function, policy

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
  	print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

# comment/uncomment these lines to switch between deterministic/stochastic environments
    env = gym.make("Deterministic-4x4-FrozenLake-v0")
    #env = gym.make("Stochastic-4x4-FrozenLake-v0")
    
    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
    
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)
        
    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)


