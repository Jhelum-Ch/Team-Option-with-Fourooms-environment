import numpy as np
from matplotlib import pyplot as plt
import itertools
from collections import Counter
from enum import IntEnum
import gym
from gym import core, spaces
from gym.envs.registration import register
from agent import Agent
from optionCritic.option import Option
from modelConfig import params
from optionCritic.Qlearning import IntraOptionQLearning
import copy
import sys
from rendering import *

if sys.version_info[0] < 3:
    print("Warning! Python 2 can lead to unpredictable behaviours. Please use Python 3 instead.")


# Size in pixels of a cell in the full-scale human view
CELL_PIXELS = 32


class FourroomsMA(gym.Env):


    # Defines the atomic actions for the agents
    class Actions(IntEnum):
        # move
        up = 0
        down = 1
        left = 2
        right = 3
        # stay = 4

    def __init__(self, n_agents = 3, goal_reward = 1., broadcast_penalty = -0.01, collision_penalty = -0.01, discount = 0.9):
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
        self.discoount = discount




        # Action enumeration for this environment
        self.actions = FourroomsMA.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        self.step_count = 0




        # create occupancy matrix.
        # 0 : free cell
        # 1 : wall
        self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])


        self.height = len(self.occupancy)
        self.width = len(self.occupancy[0])


        # Intialize atomic actions, and action spaces both for individual agents and for the joint actions a = (a^0, a^1,..., a^n)
        self.agent_actions = FourroomsMA.Actions
        self.action_space = spaces.Discrete(len(self.agent_actions))
        self.joint_actions = list(itertools.product(range(len(self.agent_actions)), repeat=self.n_agents))

        # Initialize agents with a name (agent i) and an ID (i)
        self.agents = [Agent(ID = i, name = 'agent %d' % i) for i in range(self.n_agents)]
        self.agentNames = ['agent %d' % i for i in range(self.n_agents)]

        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))


        # Directions: up, down, left, right
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.rng = np.random.RandomState(1234)


        # Generate cell numbering, along with conversion number <---> coordinates

        self.tocellnum = {}     # mapping: cell coordinates -> cell number
        self.tocellcoord = {}   # mapping: cell number -> cell coordinates
        cellnum = 0
        for i in range(self.height):
            for j in range(self.width):
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




        # Render used to generate image frames
        self.grid_render = None

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

        self.step_count = 0


        # Sample initial joint state (s_0,...,s_n) without collision
        initial_state = tuple(self.rng.choice(self.init_states, self.n_agents, replace=False))
        for i in range(self.n_agents):
            self.agents[i].state = initial_state[i]     # Store state in agents

        self.currstate = initial_state

        self.discovered_goals = []
        return initial_state


    # update state of the world
    def step(self, actions):  # actions is a list,
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


        rewards = [0.] * self.n_agents

        reward = 0.


        nextcells = [None] * self.n_agents
        rand_nums = self.rng.uniform(size=self.n_agents)

        for i in range(self.n_agents):

            currcell = self.tocellcoord[self.agents[i].state]
            if isinstance(actions,int):
                act = actions
            else:
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
                    self.discovered_goals.append(s)
            #rewards[i] += broadcasts[i]*self.broadcast_penalty


        self.currstate = tuple(nextcells)



        reward = np.sum(rewards)

        self.step_count += 1


        # If all goals were discovered, end episode
        done = len(self.discovered_goals) == len(self.goals)

           
        return reward, self.currstate, done, None 

    def neighbouringState(self,agent,action): # agent \in {0,1,2..}
        agent_curr_state = self.currstate[agent] 
        currcell = self.tocellcoord[agent_curr_state]
        direction = self.directions[action]
        if self.occupancy[tuple(currcell+direction)] == 1:
            return None
        else:
            neighbouring_cell = self.tocellnum[tuple(currcell+direction)]

            return neighbouring_cell



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


    def encode(self):
            """
            Generates an array encoding of the environment
            Encoding key:
            empty               : 0
            wall                : 1
            agent               : 2
            discovered goal     : 10
            undiscovered goal   : 100
            Agents can overlap with goals. This is indicated by keys (12 or 102).
            """

            # Start from occupancy
            encoding = self.occupancy.copy();

            # Add goals
            for g in self.goals:
                if g in self.discovered_goals:
                    encoding[self.tocellcoord[g]] += 10
                else:
                    encoding[self.tocellcoord[g]] += 100

            # Add agents
            for pos in self.currstate:
                encoding[self.tocellcoord[pos]] += 2

            return encoding

    def render(self, mode='human', close=False):
        """
        Render the whole-grid human view
        """

        if close:
            if self.grid_render:
                self.grid_render.close()
            return

        if self.grid_render is None:
            self.grid_render = Renderer(
                self.width * CELL_PIXELS,
                self.height * CELL_PIXELS,
                True if mode == 'human' else False
            )

        r = self.grid_render

        r.beginFrame()

        # Render the whole grid
        self._render_grid(r, CELL_PIXELS)

        #Draw agents
        for pos in self.currstate:

            loc = self.tocellcoord[pos]

            # Draw the agent
            r.push()
            r.translate(
                CELL_PIXELS * (loc[1] + 0.5),
                CELL_PIXELS * (loc[0] + 0.5)
            )
            r.setLineColor(255, 0, 0)
            r.setColor(255, 0, 0)
            r.drawCircle(0,0,12)
            r.pop()


        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()

        return r


    def _render_grid(self, r, tile_size):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        assert r.width == self.width * tile_size
        assert r.height == self.height * tile_size

        # Total grid size at native scale
        widthPx = self.width * CELL_PIXELS
        heightPx = self.height * CELL_PIXELS

        r.push()

        # Internally, we draw at the "large" full-grid resolution, but we
        # use the renderer to scale back to the desired size
        r.scale(tile_size / CELL_PIXELS, tile_size / CELL_PIXELS)

        # Draw the background of the in-world cells black
        r.fillRect(
            0,
            0,
            widthPx,
            heightPx,
            0, 0, 0
        )

        # Draw grid lines
        r.setLineColor(100, 100, 100)
        for rowIdx in range(0, self.height):
            y = CELL_PIXELS * rowIdx
            r.drawLine(0, y, widthPx, y)
        for colIdx in range(0, self.width):
            x = CELL_PIXELS * colIdx
            r.drawLine(x, 0, x, heightPx)

        # Render the grid

        grid = self.encode()

        for j in range(0, self.width):
            for i in range(0, self.height):
                cell = grid[i,j]
                if cell == 0:
                    continue

                r.push()
                r.translate(j * CELL_PIXELS, i * CELL_PIXELS)
                if cell == 1:
                    self._render_wall(r)
                elif cell == 10 or cell == 12:
                    self._render_goal(r, discovered=True)
                elif cell == 100 or cell == 102:
                    self._render_goal(r, discovered=False)
                r.pop()

        r.pop()

    @staticmethod
    def _render_wall(r):
        c = np.array([100, 100, 100])       # Grey

        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2])

        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

    @staticmethod
    def _render_goal(r, discovered=False):
        if not discovered:
            c = np.array([0, 255, 0])       # Bright green
        else:
            c = np.array([0, 100, 0])       # Dull green

        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2])

        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])



register(
    id='FourroomsMA-v1',
    entry_point='fourroomsEnv:FourroomsMA',
    timestep_limit=20000,
    reward_threshold=1,  # should we modify this?
)
