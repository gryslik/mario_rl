import tensorflow as tf
import time
import pandas as pd
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import warnings
import os
from sys import platform
import sys
import re

warnings.simplefilter("ignore", lineno=148)

if (platform == "darwin"):
    model_path = os.path.expanduser("~/gitRepos/mario_rl/trained_models/")
    os.chdir(os.path.expanduser("~/gitRepos/mario_rl/"))
else:
    model_path = '/home/ubuntu/data/code/mario2/trained_models/'
    os.chdir(os.path.expanduser("~/data/code/mario2/"))

# log filename
log_fn = "training_log.txt"

from helper_file import *
from DDQN_model import *

#######################################################################################
# Initialize environment and parameters
#######################################################################################
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
raw_image_dim = pre_process_image(env.reset()).shape
num_episodes = 10000
num_frames_to_stack = 4

# if there is a model name then get file string (for loading specific model) and the old epsilon value from the log. You can then restart training.
if (len(sys.argv) == 3):
    model_filename = sys.argv[1]
    old_epsilon_value = float(sys.argv[2])
    my_agent = DQN(env=env, single_frame_dim=raw_image_dim, num_frames_to_stack=num_frames_to_stack, old_model_filepath='trained_models/' + model_filename, old_epsilon_value=old_epsilon_value)
else:
    my_agent = DQN(env=env, single_frame_dim=raw_image_dim, num_frames_to_stack=num_frames_to_stack)

totalreward = []
steps = []
flag_result = []
final_x_position = []

### This is the main execution loop
print("Epsilon decay value is: " + str(my_agent.epsilon_decay))
for episode in range(num_episodes):
    print("----------------------------------------------------------------------------")
    print("Episode: " + str(episode) + " started with memory buffer of size: " + str(
        my_agent.memory.size) + " and writing to index: " + str(my_agent.memory.index) + " and with epsilon: " + str(
        my_agent.epsilon))
    time_start = time.time()
    cur_state = np.repeat(pre_process_image(env.reset())[:, :, np.newaxis], num_frames_to_stack, axis=2)  # reshape it (90x128x4)

    episode_reward = 0
    step = 0
    done = False
    current_x_position = []

    while not done:
        if (step % 100 == 0):
            print("At step: " + str(step))
            print_timings = True
        else:
            print_timings = False

        action = my_agent.act(cur_state)  # make a prediction
        my_agent.update_target_model(episode, step, False, False)

        new_state, reward, done, info = take_skip_frame_step(env, action, num_frames_to_stack)  # take a step when you repeat the same action for 4 frames
        step += 1

        # Add to memory
        my_agent.remember(cur_state, action, reward, new_state, done)

        # fit the model
        my_agent.replay(print_time=print_timings)

        # set the current_state to the new state
        cur_state = new_state

        episode_reward += reward
        current_x_position.append(info['x_pos'])

        if info['flag_get']:
            print("Breaking due to getting flag!")
            print("Current position is:" + str(info['x_pos']))
            break
        if step > 3000:
            print("Breaking due to out of steps.")
            break

    totalreward.append(episode_reward)
    steps.append(step)
    flag_result.append(info['flag_get'])
    final_x_position.append(current_x_position[-1])

    if info['flag_get']:
        info_str = "Episode: " + str(episode) + " -- SUCCESS -- with a total reward of: " + str(
            episode_reward) + "and at position: " + str(final_x_position[-1])
        print(info_str)
        my_agent.save_model(model_path + "episode-{}_model_success.h5".format(episode))

    else:
        info_str = "Episode: " + str(episode) + " -- FAILURE -- with a total reward of: " + str(episode_reward) + " and at position: " + str(final_x_position[-1])
        print(info_str)
        if episode % 10 == 0:
            my_agent.save_model(model_path + "episode-{}_model_failure.h5".format(episode))

    time_end = time.time()
    tf.keras.backend.clear_session()

    print("Episode: " + str(int(episode)) + " completed in steps/time/avg_running_reward: " + str(
        steps[-1]) + " / " + str(int(time_end - time_start)) + " / " + str(np.array(totalreward)[-100:].mean()))
    print("----------------------------------------------------------------------------")

    ## logging info to log file
    info_str = "Episode: " + str(int(episode)) + "; steps: " + str(steps[-1]) + "; time: " + str(
        int(time_end - time_start)) + "; epsilon: " + str(my_agent.epsilon) + "; total reward: " + str(
        episode_reward) + "; final position: " + str(final_x_position[-1]) + "; avg_running_reward: " + str(
        np.array(totalreward)[-100:].mean())

    ## write to log file
    log = open(log_fn, "a+")  # append mode and create file if it doesnt exist
    log.write(info_str +
              "\n" +
              "----------------------------------------------------------------------------" +
              "\n")
    log.close()

env.close()

results_df = pd.DataFrame(totalreward, columns=['episode_reward'])
results_df['steps_taken'] = steps
results_df['flag_get'] = flag_result
results_df['x_pos'] = final_x_position
results_df['average_running_reward'] = results_df['episode_reward'].rolling(window=100).mean()
results_df.to_csv(model_path + "training_results.csv")


