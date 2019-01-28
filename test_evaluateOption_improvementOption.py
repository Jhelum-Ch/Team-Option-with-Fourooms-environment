import itertools
from fourroomsEnv import FourroomsMA
from modelConfig import params
from optionCritic.option import Option, createOptions
from optionCritic.policies import SoftmaxOptionPolicy, SoftmaxActionPolicy
from optionCritic.termination import SigmoidTermination
from optionCritic.Qlearning import IntraOptionQLearning, IntraOptionActionQLearning
import optionCritic.gradients as grads

env = FourroomsMA()
avail_options, mu_policies = createOptions(env)
#print(avail_options[0].optionID, avail_options[0].policy, avail_options[0].termination, avail_options[0].available)

joint_state_list = set([tuple(np.sort(s)) for s in env.states_list])
joint_option_list = list(itertools.permutations(range(params['agent']['n_options']), params['env']['n_agents']))
# joint_action_list = list(itertools.product(range(len(env.agent_actions)), repeat=params['env']['n_agents']))

# mu_policy is the policy over options
mu_weights = dict.fromkeys(joint_state_list, dict.fromkeys(joint_option_list, 0))
# mu_policy = SoftmaxOptionPolicy(mu_weights)


pi_policies = [SoftmaxActionPolicy(len(env.cell_list), len(env.agent_actions)) for _ in avail_options]

# terminations take agent's state (not joint-state)
option_terminations = [SigmoidTermination(len(env.cell_list)) for _ in range(params['agent']['n_options'])]
critic = IntraOptionQLearning(params['env']['discount'], params['doc']['lr_Q'], option_terminations, SoftmaxOptionPolicy(mu_weights).weights)


action_critic = IntraOptionActionQLearning(params['env']['discount'], params['doc']['lr_Q'], option_terminations, SoftmaxActionPolicy(len(env.cell_list), len(env.agent_actions)).weights, critic)

joint_state = DOC(env, avail_options, mu_policies).joint_state
print('joint state', joint_state)
joint_option = DOC(env, avail_options, mu_policies).joint_option
print('joint option', joint_option)
# for agent in env.agents:
#     print('agent ID', agent.ID)
#     agent.state = joint_state[agent.ID]
#     agent.option = joint_option[agent.ID]
    

joint_action = DOC(env, avail_options, mu_policies).joint_action
# for agent in env.agents:
#     agent.action = joint_action[agent.ID]


# test with Agent~1
agent1 = env.agents[0]
print('agent1',agent1, 'agent1 option', agent1.option)

agent1_state = agent1.state
agent1_option = agent1.option
agent1_action = agent1.action

pi_policy_of_agent_option = pi_policies[agent1_option]
agent_option_termination = option_terminations[agent1_option]

intra_option_policy_improvement = grads.IntraOptionGradient(pi_policy_of_agent_option, params['doc']['lr_theta'])
termination_improvement = grads.TerminationGradient(agent_option_termination, critic, params['doc']['lr_phi'])



critic.start(joint_state, joint_option)
action_critic.start(joint_state, joint_option, joint_action)

evalOption = DOC(env, avail_options, mu_policies).evaluateOption(critic, action_critic, option_terminations, baseline=False)
imprvOption = DOC(env, avail_options, mu_policies).improveOption_of_agent(agent1.ID, intra_option_policy_improvement, termination_improvement, evalOption)