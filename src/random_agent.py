import argparse
import logging
import os
import sys

import gym
import gym_pull
gym_pull.pull('github.com/ppaquette/gym-doom')        # Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('ppaquette/DoomBasic-v0')

"""
Actions:
[0] - ATTACK - Shoot weapon - Values 0 or 1
[1] - USE - Use item - Values 0 or 1
[2] - JUMP - Jump - Values 0 or 1
[3] - CROUCH - Crouch - Values 0 or 1
[4] - TURN180 - Perform 180 turn - Values 0 or 1
[5] - ALT_ATTACK - Perform alternate attack
[6] - RELOAD - Reload weapon - Values 0 or 1
[7] - ZOOM - Toggle zoom in/out - Values 0 or 1
[8] - SPEED - Run faster - Values 0 or 1
[9] - STRAFE - Strafe (moving sideways in a circle) - Values 0 or 1
[10] - MOVE_RIGHT - Move to the right - Values 0 or 1
[11] - MOVE_LEFT - Move to the left - Values 0 or 1
[12] - MOVE_BACKWARD - Move backward - Values 0 or 1
[13] - MOVE_FORWARD - Move forward - Values 0 or 1
[14] - TURN_RIGHT - Turn right - Values 0 or 1
[15] - TURN_LEFT - Turn left - Values 0 or 1
[16] - LOOK_UP - Look up - Values 0 or 1
[17] - LOOK_DOWN - Look down - Values 0 or 1
[18] - MOVE_UP - Move up - Values 0 or 1
[19] - MOVE_DOWN - Move down - Values 0 or 1
[20] - LAND - Land (e.g. drop from ladder) - Values 0 or 1
[21] - SELECT_WEAPON1 - Select weapon 1 - Values 0 or 1
[22] - SELECT_WEAPON2 - Select weapon 2 - Values 0 or 1
[23] - SELECT_WEAPON3 - Select weapon 3 - Values 0 or 1
[24] - SELECT_WEAPON4 - Select weapon 4 - Values 0 or 1
[25] - SELECT_WEAPON5 - Select weapon 5 - Values 0 or 1
[26] - SELECT_WEAPON6 - Select weapon 6 - Values 0 or 1
[27] - SELECT_WEAPON7 - Select weapon 7 - Values 0 or 1
[28] - SELECT_WEAPON8 - Select weapon 8 - Values 0 or 1
[29] - SELECT_WEAPON9 - Select weapon 9 - Values 0 or 1
[30] - SELECT_WEAPON0 - Select weapon 0 - Values 0 or 1
[31] - SELECT_NEXT_WEAPON - Select next weapon - Values 0 or 1
[32] - SELECT_PREV_WEAPON - Select previous weapon - Values 0 or 1
[33] - DROP_SELECTED_WEAPON - Drop selected weapon - Values 0 or 1
[34] - ACTIVATE_SELECTED_WEAPON - Activate selected weapon - Values 0 or 1
[35] - SELECT_NEXT_ITEM - Select next item - Values 0 or 1
[36] - SELECT_PREV_ITEM - Select previous item - Values 0 or 1
[37] - DROP_SELECTED_ITEM - Drop selected item - Values 0 or 1
[38] - LOOK_UP_DOWN_DELTA - Look Up/Down - Range of -10 to 10 (integer). - Value is the angle - +5 will look up 5 degrees, -5 will look down 5 degrees
[39] - TURN_LEFT_RIGHT_DELTA - Turn Left/Right - Range of -10 to 10 (integer). - Value is the angle - +5 will turn right 5 degrees, -5 will turn left 5 degrees
[40] - MOVE_FORWARD_BACKWARD_DELTA - Speed of forward/backward movement - Range -100 to 100 (integer). - +100 is max speed forward, -100 is max speed backward, 0 is no movement
[41] - MOVE_LEFT_RIGHT_DELTA - Speed of left/right movement - Range -100 to 100 (integer). - +100 is max speed right, -100 is max speed left, 0 is no movement
[42] - MOVE_UP_DOWN_DELTA - Speed of up/down movement - Range -100 to 100 (integer). - +100 is max speed up, -100 is max speed down, 0 is no movement
"""


# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='ppaquette/DoomDeathmatch-v0', help='Select the environment to run')
    args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env.monitor.start(outdir, force=True, seed=0)

    # This declaration must go *after* the monitor call, since the
    # monitor's seeding creates a new action_space instance with the
    # appropriate pseudorandom number generator.
    agent = RandomAgent(env.action_space)

    episode_count = 100
    max_steps = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        for j in range(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Dump result info to disk
    env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    gym.upload(outdir)
