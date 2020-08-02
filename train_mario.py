import tensorflow as tf
import time
import pandas as pd
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import warnings
import os
from sys import platform
warnings.simplefilter("ignore", lineno=148)


if (platform == "darwin"):
    model_path = os.path.expanduser("~/gitRepos/mario_rl/trained_models/")
    os.chdir(os.path.expanduser("~/gitRepos/mario_rl/"))
else:
    model_path = '/home/ubuntu/data/code/mario2/trained_models/'
    os.chdir(os.path.expanduser("~/data/code/mario2/"))

from helper_file import *
from DDQN_model import *

#######################################################################################
# Initialize environment and parameters
#######################################################################################
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
raw_image_dim = pre_process_image(env.reset()).shape
num_episodes = 10000
num_frames_to_collapse = 4
num_frames_to_stack = 4
my_agent = DQN(env=env, single_frame_dim=raw_image_dim,num_frames_to_stack=num_frames_to_stack)
totalreward = []
steps = []
flag_result = []
final_x_position = []


### This is the main execution loop
print("Epsilon decay value is: " + str(my_agent.epsilon_decay))
for episode in range(num_episodes):
    print("----------------------------------------------------------------------------")
    print("Episode: " + str(episode) + " started with memory buffer of size: " + str(my_agent.memory.size) + " and writing to index: "+ str(my_agent.memory.index) + " and with epsilon: " + str(my_agent.epsilon))
    time_start = time.time()
    cur_state = np.repeat(pre_process_image(env.reset())[:, :, np.newaxis], num_frames_to_stack, axis=2) #reshape it (120x128x4).
    episode_reward = 0
    step = 0
    done = False
    current_x_position = []

    while not done:
        if(step % 100 == 0):
            print("At step: " + str(step))

        action = my_agent.act(cur_state) # make a prediction
        my_agent.update_target_model(episode, step, False, False)

        new_state, reward, done, info = take_skip_frame_step(env, action, num_frames_to_collapse) #take a step when you repeat the same action for 4 frames
        #new_state = generate_stacked_state(cur_state, new_state) #make the new state (120x128x4) to account for frame stacking
        step += 1

        #reward = check_stuck_and_penalize(current_x_position, info['x_pos'], reward)

        if(new_state.shape != (120,128,4)): #sanity check to make sure model always gives what's expected
            import pdb
            pdb.set_trace()

        #Add to memory
        my_agent.remember(cur_state, action, reward, new_state, done)

        #fit the model
        my_agent.replay()

        #set the current_state to the new state
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


    #if len(final_x_position) > 2 and final_x_position[-1] >= final_x_position[-2] + 200:
    #    print("Updating model weights due to large improvement. Total reward of: " + str(episode_reward) + " and at position: " + str(final_x_position[-1]) + " compared to position of: " + str(final_x_position[-2]))
    #    my_agent.update_target_model(episode, step, False, True)
    #    my_agent.save_model(model_path + "episode-{}_model_improvement.h5".format(episode))
    if info['flag_get']:
        print("Episode: " + str(episode) + " -- SUCCESS -- with a total reward of: " + str(episode_reward) + " and at position: " + str(final_x_position[-1]))
        my_agent.save_model(model_path + "episode-{}_model_success.h5".format(episode))
    else:
        print("Episode: " + str(episode) + " -- FAILURE -- with a total reward of: " + str(episode_reward) + " and at position: " + str(final_x_position[-1]))
        if episode % 10 == 0:
            my_agent.save_model(model_path + "episode-{}_model_failure.h5".format(episode))

    time_end = time.time()
    tf.keras.backend.clear_session()
    print("Episode: " + str(int(episode)) + " completed in steps/time/avg_running_reward: " + str(steps[-1]) + " / " + str(int(time_end - time_start)) + " / " + str(np.array(totalreward)[-100:].mean()))
    print("----------------------------------------------------------------------------")
env.close()

results_df = pd.DataFrame(totalreward, columns=['episode_reward'])
results_df['steps_taken'] = steps
results_df['flag_get'] = flag_result
results_df['x_pos'] = final_x_position
results_df['average_running_reward'] = results_df['episode_reward'].rolling(window=100).mean()
results_df.to_csv(model_path + "training_results.csv")


