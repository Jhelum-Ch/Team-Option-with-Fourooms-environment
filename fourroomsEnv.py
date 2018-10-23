import numpy as np
import itertools
from collections import Counter
from enum import IntEnum
from gym import core, spaces
from gym.envs.registration import register
from agent import Agent
from option import Option
import sys

if sys.version_info[0] < 3:
    print("Warning! Python 2 can lead to unpredictable behaviours. Please use Python 3 instead.")


class FourroomsMA:

    # Defines the atomic actions for the agents
    class Actions(IntEnum):
        # move
        up = 0
        down = 1
        left = 2
        right = 3
        # stay = 4


    def __init__(self, n_agents = 3):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""

        self.n_agents = n_agents

        # create occupancy matrix.
        # 0 : free cell
        # 1 : wall
        self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])

        # Intialize atomic actions, and action spaces both for individual agents and for the joint actions a = (a^0, a^1,..., a^n)
        self.agent_actions = FourroomsMA.Actions
        self.agent_action_space = spaces.Discrete(len(self.agent_actions))  # Not sure if needed
        self.joint_actions = list(itertools.product(range(len(self.agent_actions)), repeat=self.n_agents))

        # Initialize agents with a name (agent i) and an ID (i)
        self.agents = [Agent(ID = i, name = 'agent %d' % i) for i in range(self.n_agents)]
        self.agentNames = ['agent %d' % i for i in range(self.n_agents)]

        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))

        # Confused about this block of code
        # self.stateActiontuples = [(-2., -2) for _ in self.n_agents]
        # self.observation = {k: v for k in self.agentNames, v in self.stateActiontuples}
        # self.observationValues = ()

        # Directions: up, down, left, right
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.rng = np.random.RandomState(1234)


        # Generate cell numbering, along with conversion number <---> coordinates

        self.tocellnum = {}     # mapping: cell coordinates -> cell number
        self.tocellcoord = {}   # mapping: cell number -> cell coordinates
        cellnum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i,j] == 0:
                    self.tocellnum[(i,j)] = cellnum
                    self.tocellcoord[cellnum] = (i, j)
                    cellnum += 1

        self.cell_list = [value for value in list(self.tocellnum.values())]     # set of agent states S^j


        # Generate the set of joint states (s^0,..., s^n), discarding states with agent collisions (e.g. (2,2,3) )
        self.states_list = [s for s in list(itertools.product(self.cell_list, repeat=self.n_agents))
                            if len(s) == len(np.unique(s))]

        self.goals = [50, 62, 71, 98, 103]  # fixed goals
        self.init_states = self.cell_list   # initial agent states
        for g in self.goals:
            self.init_states.remove(g)

        self.initial_prior = 1. / len(self.states_list) * np.ones(len(self.states_list))  # it is a vector

        # Current real joint state of the environment.
        self.currstate = None

        # # visualize grid, where walls = -1 and all cells are numbered
        # grid = self.occupancy*-1
        # for s in self.cell_list:
        #     grid[self.tocellcoord[s]] = s
        # print(grid)


    # Should be implemented in algorithm, as it requires Q values
    # def broadcast(self, agent, Q0, Q1):
    #     """An agent broadcasts if the agent is at any goal or the intra-option value for
    #     no broadcast (Q0) is less than that with broadcast (Q1)"""
    #
    #     return float((agent.state in self.goals) or (Q0 < Q1))


    # TODO: Ensure belief is doing what it is supposed to. Make it a class. Also, reflect if it should be in the environment or in the algo

    def belief(self, y):
        """
        The common belief on the states of all agents based on common observation y
        """

        prior = np.zeros(len(self.states_list))
        posterior = np.zeros(len(self.states_list))

        broadcasts = [agent.actions.broadcast for agent in self.agents]
        observations_list = self.get_observation(broadcasts)[0]

        self.observationValues = [x[0] for x in observations_list]
        goals_list = list(itertools.product(self.goals, repeat=self.n_agents))

        sumtotal = 0.
        for i in range(len(self.states_list)):
            sumtotal += float(y == self.observationValues[i] * prior[i]) * prior[i]

        for i, s in enumerate(self.states_list):

            prior = self.initial_prior

            if y in goals_list:
                posterior[self.states_list.index(list(s))] = 1.
            else:
                posterior[self.states_list.index(list(s))] = \
                    (float(y == self.observationValues) * prior[self.states_list.index(list(s))]) / sumtotal

        return posterior

    def sample_from_belief(self, y, broadcasts):  # returns array
        return self.rng.choice(self.cellnum, self.n_agents, p=self.belief(y, broadcasts))


    # returns empty cells around a given cell (taken as coordinates)
    def empty_adjacent(self, cell):
        empty = []
        for d in self.directions:
            if self.occupancy[cell+d] == 0:
                empty.append(cell+d)

        return empty



    # reset the world with multiple agents
    def reset(self):
        # Sample initial joint state (s_0,...,s_n) without collision
        initial_state = tuple(self.rng.choice(self.init_states, self.n_agents, replace=False))
        for i in range(self.n_agents):
            self.agents[i].state = initial_state[i]     # Store state in agents

        self.currstate = initial_state
        return initial_state


    # update state of the world
    def step(self, actions, broadcasts):  # actions is a list, broadcasts is a list
        """
        Each agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other free directions, each with equal probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        original cell. Similarly, if two agents would move towards the same cell, then both agents
        remain in their original cell.

        We consider a case in which rewards are zero on all state transitions.
        """

        # Process movement based on real states (not belief)

        nextcells = []
        rand_nums = self.rng.uniform(size=self.n_agents)

        for i in range(self.n_agents):
            currcell = self.tocellcoord(self.agents[i].state)
            act = actions[i]
            dir = self.directions[act]

            if rand_nums[i] > 1/3:  # pick action as intended
                if self.occupancy[currcell + dir] == 0:
                    nextcells[i] = self.tocellnum(currcell + self.directions[act])
                else:
                    nextcells[i] = self.tocellnum(currcell)

            else:   # pick random action, except one initially intended
                empty_cells = self.empty_adjacent(currcell)
                if currcell + dir in empty_cells:
                    empty_cells.remove(currcell + dir)

                if len(empty_cells) == 0:   # impossible in current layout, but possible in general
                    nextcells[i] = self.tocellnum(currcell)
                else:
                    nextcells[i] = self.tocellnum(self.rng.choice(empty_cells))

        # check for inter-agent collisions:
        collisions = [c for c, count in Counter(nextcells).iteritems() if count > 1]
        for i in len(nextcells):
            if nextcells[i] in collisions:
                nextcells[i] = self.agents[i].state     # agent collided with another, so no movement
            else:
                self.agents[i].state = nextcells[i]     # movement is valid

        self.currstate = tuple(nextcells)


        y_list, goalsExplored, states = self.get_observation(broadcasts)
        done = goalsExplored == self.goals

        return y_list, float(done), done, None


    # TODO: Change observation in order to return the correct thing


    # get the list of common observation, y_list, based on the broadcast action of each agent
    def get_observation(self, broadcasts):
        goalsExplored = []
        y = self.observation
        for i, agent in enumerate(self.agents):
            agent.actions.broadcast = broadcasts[i]
            if agent.actions.broadcast == 1.:
                y[agent.name] = (agent.state, agent.actions.action)
                goalsExplored.append(agent.state)

        y_list = list(y.values())
        return y_list, goalsExplored, states


register(
    id='FourroomsMA-v0',
    entry_point='fourroomsMA:FourroomsMA',
    timestep_limit=20000,
    reward_threshold=1,  # should we modify this?
)
