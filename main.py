"""
Skeleton Main code to meant to run the workflow. Incomplete in that it lacks elements from the Option-Critic architecture

"""

import numpy as np
from belief import Belief
from fourroomsEnv import FourroomsMA
from option import Option

env = FourroomsMA()
belief = Belief(env.n_agents)   # sample_count if different from default (1000)

R = np.random.RandomState(1337)


# TODO: Implement/import proper intra-option policies
class RandomPolicy:
    # Placeholder intra-option policy

    def __init__(self, n_actions, seed = 1337):
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)


    def sample_action(self, state):
        return self.rng.randint(0, self.n_actions)


# TODO: Implement/import the proper algorithm to handle extra-policy Q-learning, broadcast and option selection
class RandomAlgo:
    # Placeholder random algorithm
    def __init__(self, seed=1234):
        self.rng = np.random.RandomState(seed)

    def chooseBroadcast(self, state, optionID):
        return self.rng.randint(0, 1)


# This would normally be replaced by the full Option Critic architecture
# TODO: Initialize with real algorihtm.
algo =  RandomAlgo()

n_options = env.n_agents + 1

options = [Option() for _ in range(n_options)]
free_options = [opt_id for opt_id in range(n_options)]

s = 11
opt_id = 0
for o in options:
    # Options initialized to completely random. Normally these would be softmax or something else
    # TODO: Initialize with real policies and termination functions.

    o.policy = RandomPolicy(n_actions = 4, seed = s*1000)
    o.termination = RandomPolicy(n_actions = 2, seed = s*201)
    o.ID = opt_id
    s += 1
    opt_id += 1

MAX_STEPS = 1000
done = False

state = env.currstate

t = 0

# Most up-to-date information on which options the agents are running
option_belief = [None]*env.n_agents


while t < MAX_STEPS or not done :


    if t == 0:
        # assign first options
        l = [n for n in range(0, n_options)]
        start_options = R.choice(l, env.n_agents, replace=False)

        for opt in start_options:
            free_options.remove(opt)

        for i in range(env.n_agents):
            # TODO: if desired, can only pass option IDs instead of full option.
            env.agents[i].option = options[ start_options[i] ]

        option_belief = start_options.copy()

    # 1. Sample s_t ~ b_t
    if t != 0:
        state = belief.sample_single_state()


    # 2. Sample a_t = [ a_t^j ~ pi^j(s_t) ]
    actions = [a.option.policy.sample_action(state) for a in env.agents]

    # 2.1 Simulate actions a_t based on most up do date information on running options
    simu_actions = [options[ID].policy.sample_action(state) for ID in option_belief]


    # 3. Compute s_t+1 = s_t + a_t assuming that the environment layout is known and that the transition is deterministic
    # Handles wall collisions but not agent collisions
    next_state = [None for _ in range(env.n_agents)]

    for i in range(env.n_agents):
        next_cell = tuple(env.tocellcoord[state[i]] + env.directions[simu_actions[i]])
        if env.occupancy[next_cell] == 0:
            next_state[i] = env.tocellnum[next_cell]
        else:
            next_state[i] = state[i]

    next_state = tuple(next_state)

    # 4. Enact actions on reward
    step_reward, done, _ = env.step(actions)

    # 5. For all agents, determine if they broadcast
    broadcasts = [None]*env.n_agents
    for i in range(env.n_agents):
        a = env.agents[i]
        broadcasts[i] = algo.chooseBroadcast(next_state, a.option.ID)

    # 6. Decrease reward based on number of broadcasting agents
    step_reward += np.sum(broadcasts)*env.broadcast_penalty

    # 7. Get observation based on broadcasts
    y = env.get_observation(broadcasts)

    # 7.1 Update option information
    for i in range(env.n_agents):
        if broadcasts[i] == 1:
            option_belief[i] = env.agents[i].option.ID


    # Initialize samples to starting state
    if t == 0:
        samples = np.zeros((env.n_agents, belief.N))
        for i in range(env.n_agents):
            for j in range(env.n_agents):
                samples[i,j] = env.currstate[i]

    # 8. Compute data Samples based on y, by filling the gaps from the belief.
    if np.sum(broadcasts) > 0 :
        # Samples are updated iff there was at least one broadcast
        samples = belief.sample()

        for i in range(y.size()):
            if y[i] is not None:
                # Correct sampled states with broadcasted info
                for j in range(0, belief.N):
                    samples[i, j] = y[i]

                # Correct next state info (used below for termination decision)
                next_state[i] = y[i]

    # 9. Update belief
    # belief.updateBeliefParameters(samples)

    # TODO: 10. TD evaluation and Policy improvement

    # 11. Check termination and assign new options
    for a in env.agents:
        if a.option.termination.sample_action(next_state) == 1:
            free_options.append(a.option.ID)
            a.option = None

    free_options.sort()

    for a in env.agents:
        if a.option is None:
            new_opt = R.choice(free_options)
            free_options.remove(new_opt)
            a.option = options[new_opt]


    # Increment time
    t += 1