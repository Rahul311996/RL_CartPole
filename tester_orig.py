import numpy as np
import torch
import gym
from lake_env import *

class Tester(object):

    def __init__(self):
        """
        Initialize the Tester object by loading your model.
        """
        self.env=gym.make('Stochastic-4x4-FrozenLake-v0')



    def evaluate_policy(self, env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
        """Evaluate the value of a policy.

        See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        policy: np.array
          The policy to evaluate. Maps states to actions.
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.

        Returns
        -------
        np.ndarray
          The value for the given policy
        """
        # TODO:

        value_function_old=np.reshape(np.random.rand(env.nS).T,(-1,1))
        value_function_new=np.zeros((16,1))
        policy=0.25*np.ones((16,4))
        gamma=0.9

        start=time.time()

        for iter in range(int(1e3)):
            delta=0
            state_action_function=np.zeros((16,4))
            for state in env.P:
                for action in env.P[state]:
                    for (transition_probability,next_state,reward,is_term) in env.P[state][action]:
                        if(is_term):
                            action_value=0
                        else:
                            action_value=gamma*value_function_old[next_state,0]
                        state_action_function[state][action]+=transition_probability*(reward+action_value)
                value_function_new[state]=np.sum(policy[state,:]*state_action_function[state,:])
                delta=max(delta,np.abs(value_function_new[state]-value_function_old[state]))
                value_function_old[state]=value_function_new[state]
            if(delta<tol):
                break

        return value_function_old, iter

    def policy_iteration(self, env, gamma, max_iterations=int(1e3), tol=1e-3):
        """Runs policy iteration.

        See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        You should use the improve_policy and evaluate_policy methods to
        implement this method.

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.

        Returns
        -------
        (np.ndarray, np.ndarray, int, int)
           Returns optimal policy, value function, number of policy
           improvement iterations, and number of value iterations.
        """
        # TODO:  Your code goes here.
        return None, None, 0, 0

    def value_iteration(self, env, gamma, max_iterations=int(1e3), tol=1e-3):
        """Runs value iteration for a given gamma and environment.

        See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
        book.

        http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        max_iterations: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.

        Returns
        -------
        np.ndarray, iteration
          The value function and the number of iterations it took to converge.
        """
        # TODO: Your Code goes here.
        value_function_old=np.reshape(np.random.rand(env.nS).T,(-1,1))
        value_function_new=np.zeros((16,1))
        gamma=0.9

        start=time.time()

        for iter in range(int(max_iterations)):
            delta=0
            state_action_function=np.zeros((16,4))
            for state in env.P:
                for action in env.P[state]:
                    for (transition_probability,next_state,reward,is_term) in env.P[state][action]:
                        if(is_term):
                            action_value=0
                        else:
                            action_value=gamma*value_function_old[next_state,0]
                        state_action_function[state][action]+=transition_probability*(reward+action_value)
                value_function_new[state]=np.max(state_action_function[state,:])
                delta=max(delta,np.abs(value_function_new[state]-value_function_old[state]))
                value_function_old[state]=value_function_new[state]

            if(delta<tol):
                break

        return value_function_old, iter

    def policy_gradient_test(self, state):
        """
        Parameters
        ----------
        state: np.ndarray
            The state from the CartPole gym environment.
        Returns
        ------
        np.ndarray
            The action in this state according to the trained policy.
        """
        # TODO. Your Code goes here.
        return 0
