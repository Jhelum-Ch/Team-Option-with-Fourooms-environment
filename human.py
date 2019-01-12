#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Orchard-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.move_left
        elif keyName == 'RIGHT':
            action = env.actions.move_right
        elif keyName == 'UP':
            action = env.actions.move_forward
        elif keyName == 'DOWN':
            action = env.actions.move_backward

        elif keyName == 'RETURN':
            action = env.actions.turn_left
        elif keyName == 'CTRL':
            action = env.actions.turn_right
        elif keyName == 'PAGE_DOWN':
            action = env.actions.stay

        elif keyName == 'SPACE':
            action = env.actions.fire

        elif keyName == 'ALT' and env.gift_enabled:
            action = env.actions.gift_reward

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step([action]*env.n_agents)

        print('step=%s, reward=%.2f' % (env.step_count, reward[0]))

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
