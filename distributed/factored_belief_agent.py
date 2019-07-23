import collections
import numpy as np
import scipy.stats as stats
import random
from modelConfig import params
import copy
from itertools import product
from fourroomsEnv import FourroomsMA
import collections


class MultinomialDirichletBelief:
    def __init__(self, env, alpha, sample_count=10000):
        super(MultinomialDirichletBelief, self).__init__()
        self.env = copy.deepcopy(env)
        # self.env = env.deepcopy()
        self.alpha = alpha
        # self.joint_observation = joint_observation
        self.sample_count = sample_count  # sample_count is for rejection sampling
        # self.curr_joint_state = self.env.currstate
        #self.states_list = self.env.states_list
        # self.counts = collections.Counter(self.states_list)
        self.counts = collections.Counter(self.env.cell_list)

    def update(self, observation, old_feasible_states):  # old_feasible_states is a list of integers/lists
        # assert isinstance(counts, dict), "counts is a disctionary"

        self.observation = observation
        #print('obs', observation)
        if not observation is None and observation[1] is None:
            new_feasible_states = self.new_feasible_state(old_feasible_states, observation)
        elif not observation is None and observation[1] is not None:
            new_feasible_states = observation[0]
        else:
            new_feasible_states = self.new_feasible_state(old_feasible_states)
        self.old_feasible_states = new_feasible_states  # update feasible states

        # Set the counts vector zero
        # counts_vec = [self.counts.get(i, 0) for i in range(len(self.states_list))]
        self.counts_vec = [self.counts.get(i, 0) for i in range(len(self.env.cell_list))]

        if not self.observation is None:
            observed_state = self.observation[0]

            for item in self.env.cell_list:
                if item == observed_state:
                    self.counts_vec[self.env.cell_list.index(item)] += 10000000
                    break

        else:
            #print('old', old_feasible_states)
            b = self.new_feasible_state(self.old_feasible_states)
            #print('b', b)
            # list_final = []
            # #for i in range(len(self.joint_observation)):
            #     if self.observation is None:
            #         list_int = b[i]
            #         list_final.append(list_int)
            #     else:
            #         list_int = []
            #         list_int.append(int(self.joint_observation[i][0]))
            #         list_final.append(list_int)
            # updated_states_list = list(product(*list_final))

            # print('removing duplicates:')
            # for joint_state in updated_states_list:
            # 	if len(joint_state) > len(set(joint_state)):
            # 		print(joint_state, ' removed')
            # 		updated_states_list.remove(joint_state)
            #
            # import pdb; pdb.set_trace()
            # print(updated_states_list)

            # new_updated_states_list = []
            # for state in b:
            #     if len(list(joint_state)) == len(set(list(joint_state))):
            #         new_updated_states_list.append(joint_state)

            # print('Entering updation:')
            for item in b:
                # print(item)
                item = int(item)
                self.counts[item] += 10000000
                self.counts_vec[self.env.cell_list.index(item)] += 10000000

        return MultinomialDirichletBelief(self.env, np.add(self.alpha, self.counts_vec))

    def pmf(self):
        #print('alpha', self.alpha, 'counts_vec', self.counts_vec)
        a = stats.dirichlet.rvs(self.alpha + self.counts_vec, size=1, random_state=1)
        return a[0]

    def sampleJointState(self):  # sample one joint_state from posterior
        # self.joint_observation = joint_observation
        sampled_state_idx = int(np.random.choice(range(len(self.env.cell_list)), 1, p=self.pmf()))
        return self.env.cell_list[sampled_state_idx]

    # def rejectionSamplingNeighbour(self):
    #
    #     # determine neighborhood of each agent
    #     neighborhood = np.empty((params['env']['n_agents'], params['agent']['n_actions']))  # create an empty n-d-array
    #     for agent in range(params['env']['n_agents']):
    #         for action in range(params['agent']['n_actions']):
    #             # self.env.currstate = self.curr_joint_state
    #             neighboring_state = self.env.neighbouringState(agent, action)
    #             neighborhood[agent, action] = neighboring_state
    #     # print('neighborhood', neighborhood)
    #     # each agent rejects a sample from common-belief posterior based on its own neighborhood
    #     consistent = False
    #     sample_count = 0
    #     rs = np.zeros(params['env']['n_agents'])
    #     while consistent is False and sample_count <= self.sample_count:
    #         sampled_joint_state = self.sampleJointState()
    #         print(sampled_joint_state)
    #         for agent in range(params['env']['n_agents']):
    #             # rejection sampling
    #             rs[agent] = 1.0 * (sampled_joint_state[agent] in neighborhood[
    #                 agent])  # agent accepts if the corresponding joint-state component is in its true state's neighbourhood
    #         if np.prod(rs) == 1.0:
    #             consistent = True
    #
    #         sample_count += 1
    #
    #     return sampled_joint_state

    def estimated_feasible_state(self, agent_state, action=None):
        feasible_state_list = np.zeros(params['agent']['n_actions'])
#       # print('agent_state', agent_state, 'occupancy', self.env.occupancy[agent_state])
        #currcell = agent_state
        currcell = self.env.tocellcoord[agent_state]
        if action is None:
            for action in range(params['agent']['n_actions']):
                direction = self.env.directions[action]
               # print('currcell', currcell, 'direction', direction)
                if self.env.occupancy[tuple(currcell + direction)] != 1:
                    feasible_state_list[action] = self.env.tocellnum[tuple(currcell + direction)]
            return [item for item in feasible_state_list if item != 0.]
        else:
            direction = self.env.directions[action]
            if self.env.occupancy[tuple(currcell + direction)] == 1:
                next_state = self.env.tocellnum[tuple(currcell)]
            else:
                next_state = self.env.tocellnum[tuple(currcell + direction)]
            return next_state

    def new_feasible_state(self, old_feasible_states, obs = None):  # old_feasible_states can be either integer or
        # list
        new_feasible_states = []

        #for i in range(len((obs))):

        if obs is not None:
            if obs[1] is None or self.env.occupancy[tuple(self.env.tocellcoord[obs[0]] + self.env.directions[obs[1]])] == 1:
                lst = [self.env.tocellnum[item] for item in self.env.empty_adjacent(self.env.tocellcoord[obs[0]])]
                new_feasible_states.append(lst)
            else:
                new_feasible_states.append(self.env.tocellnum[tuple(self.env.tocellcoord[obs[0]] + self.env.directions[obs[1]])])

            #print('new', new_feasible_states)
            new_feasible_states = new_feasible_states[0]


        else:
            if isinstance(old_feasible_states, (int, np.integer)):
                new_list = self.estimated_feasible_state(old_feasible_states)  # new_list is list
                new_feasible_states.append(new_list)
            else:  # if isinstance(old_feasible_states[i],list)
                #print('old_feasible', old_feasible_states)
                new_list = [self.estimated_feasible_state(s) for s in
                            old_feasible_states]  # new_list is list of list

                flatten_new_list = [int(s) for item in new_list for s in item]

                new_feasible_states.append(list(set(flatten_new_list)))

                #print('new_list', new_list, 'flatten', flatten_new_list, )
                new_feasible_states = new_feasible_states[0]
                #print('new_feasible', new_feasible_states)

        return new_feasible_states

