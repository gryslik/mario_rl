import tensorflow as tf
import time
import sys
from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import warnings
import os
from sys import platform
warnings.simplefilter("ignore", lineno=148)
from helper_file import *
import re

save_individual_frames = False

warnings.simplefilter("ignore", lineno=148)

###Warning it expects a number!
model_number = int(sys.argv[1])
#model_number = 4200

if (platform == "darwin"):
    model_path = os.path.expanduser("~")
    os.chdir(os.path.expanduser("~/gitRepos/mario_rl/"))
    video_path = os.path.expanduser("~/gitRepos/mario_rl/recorded_videos/")
else:
    model_path = '/home/ubuntu/data/code/mario/models6-DDQN/'
    #os.chdir(os.path.expanduser("~/gitRepos/mario_rl/"))


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')   #2-2, 1-1
env = JoypadSpace(env, RIGHT_ONLY)
env = gym.wrappers.Monitor(env, video_path+str(model_number), force=True)

regex = re.compile('episode-'+str(model_number)+"_", re.UNICODE)

for file in os.listdir(model_path):
    if regex.match(file):
        model_file_path = model_path+file

print("Processing model: " + model_file_path)
model = tf.keras.models.load_model(model_file_path)


num_frames_to_stack = 4
state = np.repeat(pre_process_image(env.reset())[:, :, np.newaxis], num_frames_to_stack, axis=2) #reshape it (120x128x4).
#show_bitmap_image(long_state[:,:,3]) sample code
done = False;
step_counter = 0
while not done and step_counter < 2000: # Now we need to take the same action every 4 steps
    prediction_values = model.predict(np.expand_dims(state, axis=0).astype('float16'))
    action = np.argmax(prediction_values)
    state, reward, done, info = take_skip_frame_step(env, action, 4, True)
    if save_individual_frames:
        for frame_idx in range(num_frames_to_stack):
            save_bitmap_data(state[:,:,frame_idx], video_path + str(model_number)+"/individual frames/" + str(step_counter),frame_idx)
    step_counter+=1
    print("Steps: " +  str(step_counter) + " --- position: " + str(info['x_pos']))

env.close()