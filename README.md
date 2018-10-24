# teamOptionFourooms
Code for team option paper

## Environment code description (fourroomsEnv.py)

### Hyperparameters:
All accessed as attributes of the environment
1. _int_ `n_agents` = number of agents (default = 3)
2. _int_ `goal_reward` = reward upon discovering (exploring) a goal state _for the first time_ (default = 1)
3. _int_ `collision_penalty` = penalty upon a move resulting in a collision, whether with a wall or another agent (default = -0.01)
4. _int_ `broadcast_penalty` = penalty upon broadcasting (default = -0.01)

### Useful attributes:
#### States:
1. _List_ `goals` = Goals in the environment (fixed to  `[50, 62, 71, 98, 103]`)
2. _List_ `discovered_goals` = Goals already discovered. These do not grant rewards anymore.
3. _tuple_ `currstate` = ground truth joint state s_t
4. _dict_ `tocellnum` = mapping from cell coordinates (as a tuple) to cell number (as an int)
5. _dict_ `tocellcoord` = mapping from cell number (as an int) to cell coordinates (as a tuple)

#### Agents:
1. _List_ `agents` = list of agents in the environment

For the **Agent class**:
1. _int_ `ID` = unique index corresponding to the agent's position in `env.agents`
2. _str_ `name` = name of the agent: "agent [ID]"
3. _int_ `state` = ground truth current state of the agent, stored as the cell number
4. _int_ `option` = Index of the current active option in the option list

### Important functions:
1. `reset()` : takes no argument. Resets agent positions and discovered goals. Returns initial joint state (_tuple_)
2. `step(actions)` : takes a _List_ of actions (one for each agent). Processes the moves and handles collisions. Returns:
  - _List_ `Rewards` = rewards for that transition for each agent (from discovering a goal and/or colliding).
  **Does not account for broadcasting penalty.**
  - _bool_ `done` = `True` if all goals are discovered, `False` otherwise
  - `None`
3. `get_observation(broadcasts)` : takes a _List_ of broadcasts (one for each agent). Returns the observations y_t 
(as a _List_) based on eq. (19). Observations only contain states, not actions. **Does not account for broadcasting penalty.**

## Expected workflow

Starting in the state s_t with belief up to date. The transition towards s_(t+1) requires the following steps:

1. For all agents j, sample action a^j according to the internal policy associated with the option `agents[j].option`.
Store all actions in a _List_ `actions`.
2. Call `step_rewards, done, _ = step(actions)`.
3. For all agents, determine if they broadcast based on eq. (18). Store decision in `broadcasts`, where _0 = no broadcast_ and
_1 = broadcast_.
4. Assign `env.broadcast_penalty` to agents that decided to broadcast.
5. Call `y_t = env.get_observation(broadcasts)` and update belief according to eq. (21).

6. Do necessary belief, Q-values and policy updates.

## Design choices

1. An agent can at most be penalized for a single collision per step.
2. All collisions are determined based on the ground truth. All action selections are based on the believed state.
2. The belief was removed from the environment class, since it is a property of the learning agent, not the environment.
3. The `step(actions)` and the `get_observation(broadcasts)` functions were separated in order to allow for updating the
belief immediately after receiving observation y_t and before sampling the new joint-state s_(t+1), which in turn is used to sample the actions.

