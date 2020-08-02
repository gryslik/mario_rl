import random
import numpy as np

def remove_color(bitmap):
    return np.mean(bitmap, axis=2).astype(np.uint8)

def down_sample(bitmap):
    return bitmap[::2, ::2]

def max_pool(bitmap):
    return np.max(bitmap, axis=0)

def pre_process_image(bitmap):
    return(down_sample(remove_color(bitmap)))


def pre_process_images(bitmaps):
    image_stack = []
    for i in range(len(bitmaps)):
        image_stack.append(down_sample(remove_color(bitmaps[i])))
    return max_pool(np.reshape(image_stack, (len(bitmaps), image_stack[0].shape[0], image_stack[0].shape[1])))


def take_skip_frame_step(env, action, num_frames_to_collapse, render = False):
    image_list = []
    image_reward = 0
    for i in range(num_frames_to_collapse):
        new_state, reward, done, info = env.step(action)
        if render:
            env.render()
        image_list.append(new_state.copy())
        image_reward += reward
        if done:
            break
    combined_state = pre_process_images(image_list)
    return combined_state, image_reward, done, info

def generate_stacked_state(old_state, new_state): #old state is (x,y,num_frames), new state is (x,y)
    num_frames = old_state.shape[2]
    result_state = old_state.copy()
    for i in range(num_frames): #This keeps a rolling stack of frames
        if i < num_frames - 1: #0 -> 1, 1-> 2, 2-> 3:   #i < 3   (i in range (4)) -> 0,1,2,3
            result_state[:, :, i] = old_state[:, :, i + 1]
        else:
            result_state[:, :, i] = new_state
    return result_state

def check_stuck_and_penalize(current_x_position, current_position, reward):
    if len(current_x_position) > 250:
        avg_position = round(np.array(current_x_position[-250:]).mean(), 0)
        #current_position = info['x_pos']
        if current_position - avg_position < 1:  ## No real movement
            reward += -5  # make it negative so that it doesn't get stuck and tries new things
            reward = max(reward, -15)  # can't go below -15
    return reward