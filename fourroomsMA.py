import numpy as np
import itertools
from gym import core, spaces
from gym.envs.registration import register

# # state of agents 
# class AgentState(object):
#     def __init__(self):
#         # super(AgentState, self).__init__()
#         #  # event to braodcast 
#         #  self.b = None
#         self.state = None

#option of the agent: has three components, a physical action, a termination and a broadcast action
class Option(object):
    def __init__(self):
        # physical action
        self.action = None
        #termination 
        self.termination = None
        # broadcast action
        self.broadcast = 0



class Agent:
    def __init__(self):
        super(Agent, self).__init__()
        # name 
        self.name = ''
        # state
        self.state = None
        # action
        self.actions = Option()



class FourroomsMA:
    def __init__(self):
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
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        self.numAgents = 3
        # From any state the agent can perform one of four actions, up, down, left or right
        agent_action_space = spaces.Discrete(4) #check if this is a list
        self.action_space = list(itertools.product(self.agent_action_space, repeat = self.numAgents))
        
        self.agents = [Agent() for _ in range(numAgents)]
        self.options = [Option() for _ in range(numAgents)]
        #self.agents = []

        self.agentNames = []
        for i, agent in enumerate(self.agents):
            agent.name = 'agent %d' % i
            self.agentNames[i] = agent.name

        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))

        self.stateActiontuples = [(-2.,-2) for _ in numAgents]
        self.observation = {k:v for k in self.agentNames, v in self.stateActiontuples}
        self.observationValues = ()

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
        self.rng = np.random.RandomState(1234)

        self.states = {}
        self.tocellnum = {}
        self.tostate = {}
        self.cellnum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tocellnum[(i,j)] = self.cellnum
                    self.cellnum += 1
        self.tocell = {v:k for k,v in self.tocellnum.items()} #coordinate of a cell corresponding to the number of the cell
        
        self.cell_list = [value for value in list(self.tocellnum.values())]

        self.states_list = list(itertools.product(self.cell_list, repeat = self.numAgents)) #numbers state of word in tuples
        

        self.goals = [50,62,71,98,103]. #fixed goals
        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goals)
        self.initial_prior = 1./len(self.states_list)*np.ones(len(self.states_list)) # it is a vector 


    def broadcast(self, agent, Q0, Q1): 
        """An agent broadcasts if the agent is at any goal or the intra-option value for 
        no broadcast (Q0) is less than that with broadcast (Q1)"""
        
        return float((agent.state in self.goals) or (Q0 < Q1))


    #the common belief on the states of all agents based on common observation y
    def belief(self,y): 
        prior = np.zeros(len(self.states_list))
        posterior = np.zeros(len(self.states_list))

        broadcasts = [agent.actions.broadcast for agent in self.agents]
        observations_list = self.get_observation(broadcasts)[0]
        
        self.observationValues = [x[0] for x in observations_list]
        goals_list = list(itertools.product(self.goals, repeat = self.numAgents))

        sumtotal = 0.
        for i in range(len(self.states_list)):
            sumtotal += float(y==self.observationValues[i]*prior[i])*prior[i]
        
        for i, s in enumerate(self.states_list):

            prior = self.initial_prior

            if y in goals_list:
                posterior[self.states_list.index(list(s))] = 1.
            else: 
                posterior[self.states_list.index(list(s))] = \
                (float(y==self.observationValues)*prior[self.states_list.index(list(s))])/sumtotal
        
        return posterior

    def sample_from_belief(self,y,broadcasts): #returns array
        return self.rng.choice(self.cellnum, self.numAgents, p=self.belief(y,broadcasts))



    #for every agent find the list of cells that are empty
    def empty_around(self, own_cell, agent, y, broadcasts): 
        availAgent = []
        sample_state_from_belief = self.sample_from_belief(y,broadcasts)
        sample_belief_other_agents = np.delete(sample_state_from_belief,int(agent.name.split()[1]))
        
        for agent.actions.action in range(self.action_space.n):
            nextcellAgent = tuple(own_cell + self.directions[agent.actions.action])
            if not self.occupancy[nextcellAgent] and nextcellAgent not in sample_belief_other_agents:
                availAgent.append(nextcellAgent)
        return availAgent  


    #reset the world with multiple agents
    def reset(self):
        initialStates= self.rng.choice(self.init_states,numAgents,replace=False) #initial states of agents without collision
        state = tuple(np.zeros(self.numAgents))
        self.currentcell = {agent:() for agent in self.agents}
        for i, agent in enumerate(self.agents):
            agent.state = initialStates[i]
            state[i] = agent.state
            self.currentcell[agent.name] = self.tocell[agent.state] #current cell coordinates of each agent
        return state #returns the tuple containing the cell number of each agent

    
    #update state of the world
    def step(self, actions, broadcasts): #actions is a list, broadcasts is a list
        """
        Each agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
       
        nextcell = {}
        
        y_list, goalsExplored, states = self.get_observation(broadcasts) 
        done = goalsExplored == self.goals

        if not done:
            for i, agent in enumerate(self.agents):
                nextcell[agent.name] = tuple(self.currentcell[agent.name] + self.directions[actions[i]])
                if not self.occupancy[nextcell[agent.name]]:
                    self.currentcell[agent.name] = nextcell[agent.name]
                    if self.rng.uniform() < 1/3.:
                        empty_cells = self.empty_around(self.currentcell[agent.name],agent,y_list)
                        self.currentcell[agent.name] = empty_cells[self.rng.randint(len(empty_cells))]
                agent.state = self.tostate[self.currentcell[agent.name]]
                self.states[agent.name] = agent.state
 
        return y_list, float(done), done, None



    #get the list of common observation, y_list, based on the broadcast action of each agent
    def get_observation(self, broadcasts): 
        goalsExplored = []
        y = self.observation
         for i, agent in enumerate(self.agents):
            agent.actions.broadcast = broadcasts[i]
            if agent.actions.broadcast == 1.:
                y[agent.name] = (agent.state,agent.actions.action)
                goalsExplored.append(agent.state)
        y_list = list(y.values())
        return y_list, goalsExplored, states

      

register(
    id='FourroomsMA-v0',
    entry_point='fourroomsMA:FourroomsMA',
    timestep_limit=20000,
    reward_threshold=1, # should we modify this?
)
