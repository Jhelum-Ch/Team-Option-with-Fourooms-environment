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


    def __init__(self, n_agents = 3, goal_reward = 1, broadcast_penalty = -0.01, collision_penalty = -0.01):
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
        self.goal_reward = goal_reward
        self.broadcast_penalty = broadcast_penalty
        self.collision_penalty = collision_penalty

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
        self.goals.sort()                   # important if not already sorted in line above
        self.discovered_goals = []
        self.init_states = self.cell_list.copy()   # initial agent states
        for g in self.goals:
            self.init_states.remove(g)

        self.initial_prior = 1. / len(self.states_list) * np.ones(len(self.states_list))  # it is a vector

        # Current real joint state of the environment.
        self.currstate = None

        self.reset()

        # # visualize grid, where walls = -1 and all cells are numbered
        # grid = self.occupancy*-1
        # for s in self.cell_list:
        #     grid[self.tocellcoord[s]] = s
        # print(grid)


    # returns empty cells around a given cell (taken as coordinates) (unused in code)
    def empty_adjacent(self, cell):
        empty = []
        for d in self.directions:
            if self.occupancy[cell+d] == 0:
                empty.append(cell+d)

        return empty

    # returns all four cells adjacent to a given cell (taken as coordinates)
    def adjacent_to(self, cell):
        adj = []
        for d in self.directions:
            adj.append(tuple(cell+d))

        return adj


    # reset the world with multiple agents
    def reset(self):
        # Sample initial joint state (s_0,...,s_n) without collision
        initial_state = tuple(self.rng.choice(self.init_states, self.n_agents, replace=False))
        for i in range(self.n_agents):
            self.agents[i].state = initial_state[i]     # Store state in agents

        self.currstate = initial_state

        self.discovered_goals = []
        return initial_state


    # update state of the world
    def step(self, actions):  # actions is a list, broadcasts is a list
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

        assert len(actions) == len(self.agents), "Number of actions (" + str(
            len(actions)) + ") does not match number of agents (" + str(self.n_agents) + ")"

        # Process movement based on real states (not belief)

        # If all goals were discovered, end episode
        done = self.discovered_goals == self.goals

        rewards = [0] * self.n_agents

        if not done:

            nextcells = [None] * self.n_agents
            rand_nums = self.rng.uniform(size=self.n_agents)

            print(rand_nums)


            for i in range(self.n_agents):

                currcell = self.tocellcoord[self.agents[i].state]
                act = actions[i]
                direction = self.directions[act]

                if rand_nums[i] > 1/3:  # pick action as intended
                    if self.occupancy[tuple(currcell + direction)] == 0:
                        nextcells[i] = self.tocellnum[tuple(currcell+direction)]
                    else:
                        nextcells[i] = self.tocellnum[tuple(currcell)]     # wall collision
                        # rewards[i] += self.collision_penalty

                else:   # pick random action, except one initially intended
                    adj_cells = self.adjacent_to(currcell)      # returns list of tuples
                    adj_cells.remove(tuple(currcell+direction))

                    index = self.rng.choice(range(len(adj_cells)))
                    new_cell = adj_cells[i]

                    if self.occupancy[new_cell] == 0:
                        nextcells[i] = self.tocellnum[new_cell]
                    else:
                        nextcells[i] = self.tocellnum[tuple(currcell)]     # wall collision
                        # rewards[i] += self.collision_penalty

            # check for inter-agent collisions:
            collisions = [c for c, count in Counter(nextcells).items() if count > 1]
            while(len(collisions) != 0):        # While loop needed to handle edge cases
                for i in range(len(nextcells)):
                    if nextcells[i] in collisions:
                        nextcells[i] = self.agents[i].state     # agent collided with another, so no movement

                collisions = [c for c, count in Counter(nextcells).items() if count > 1]

            for i in range(self.n_agents):
                if nextcells[i] == self.agents[i].state:    # A collision happened for this agent
                    rewards[i] += self.collision_penalty
                else:
                    s = nextcells[i]                        # movement is valid
                    self.agents[i].state = s
                    if s in self.goals and s not in self.discovered_goals:
                        rewards[i] += self.goal_reward

            self.currstate = tuple(nextcells)

        return rewards, done, None      # Observations are not returned; they need to be queried with broadcasts


    # get the list of common observation, y_list, based on the broadcast action of each agent
    def get_observation(self, broadcasts):

        y_list = []      # Joint observation y = (y_0, ..., y_n)

        for i in range(self.n_agents):
            if broadcasts[i] == 1:
                y_list.append(self.agents[i].state)      # y_i = s^i_t if agent i broadcasts
            else:
                y_list.append(None)                      # y_i = None, otherwise

        for a in self.agents:
            if a.state in self.goals and a.state not in self.discovered_goals:
                self.discovered_goals.append(a.state)      # track discovered goals

        self.discovered_goals.sort()

        return y_list


register(
    id='FourroomsMA-v0',
    entry_point='fourroomsMA:FourroomsMA',
    timestep_limit=20000,
    reward_threshold=1,  # should we modify this?
)
