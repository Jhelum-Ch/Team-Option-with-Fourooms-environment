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

joint_state_list = list(itertools.permutations(env.cell_list, env.n_agents))
joint_option_list = list(itertools.permutations(range(params['agent']['n_options']), params['env']['n_agents']))
joint_action_list = list(itertools.product(range(len(env.agent_actions)), repeat=params['env']['n_agents']))

# mu_policy is the policy over options
mu_weights = dict.fromkeys(joint_state_list, dict.fromkeys(joint_option_list, 0))	#TODO: fix this
mu_policy = SoftmaxOptionPolicy(mu_weights)

pi_policies = [SoftmaxActionPolicy(len(env.cell_list), len(env.agent_actions)) for _ in avail_options]

# terminations take agent's state (not joint-state)
option_terminations = [SigmoidTermination(len(env.cell_list)) for _ in range(params['agent']['n_options'])]
critic = IntraOptionQLearning(params['env']['discount'], params['doc']['lr_Q'], option_terminations)


action_critic = IntraOptionActionQLearning(params['env']['discount'], params['doc']['lr_Q'], option_terminations, critic)


agentID = 1
joint_option = (1,3,2)
agent_optionID = joint_option[agentID]
pi_policy_of_agent_option = pi_policies[agent_optionID]
agent_option_termination = option_terminations[agent_optionID]

intra_option_policy_improvement = grads.IntraOptionGradient(pi_policy_of_agent_option, params['doc']['lr_theta'])
termination_improvement = grads.TerminationGradient(agent_option_termination, critic, params['doc']['lr_phi'])


joint_state = (11,32,43)
joint_option = (1,3,2)
joint_action = (0,1,3)

agent_state = joint_state[agentID]
agent_option = joint_option[agentID]
agent_action = joint_action[agentID]

critic.start(joint_state, joint_option)
action_critic.start(joint_state, joint_option, joint_action)

evalOption = DOC(env, avail_options, mu_policies).evaluateOption(critic, action_critic, joint_state, joint_option, joint_action, baseline=False)
imprvOption = DOC(env, avail_options, mu_policies).improveOption_of_agent(agentID, intra_option_policy_improvement, termination_improvement, joint_state, joint_option, joint_action, evalOption)

