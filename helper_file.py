import random
import numpy as np
import os
import cv2


def remove_color(bitmap):
    return np.mean(bitmap, axis=2).astype(np.uint8)

# downsampling code adapted from https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/
def down_sample(bitmap):
    resized_bitmap = cv2.resize(bitmap, (84, 110), interpolation=cv2.INTER_AREA)
    focused_bitmap = resized_bitmap[18:102, :]
    focused_bitmap = np.reshape(focused_bitmap, [84, 84])
    return focused_bitmap

def max_pool(bitmap):
    return np.max(bitmap, axis=0)

def pre_process_image(bitmap):
    return(down_sample(remove_color(bitmap)))

def pre_process_images(bitmaps):
    image_stack = []
    for i in range(len(bitmaps)):
        image_stack.append(pre_process_image(bitmaps[i]))
    return np.dstack(image_stack)

def take_skip_frame_step(env, action, num_frames_to_collapse, render=False):
    image_list = []
    image_reward = 0
    for i in range(num_frames_to_collapse):
        new_state, reward, done, info = env.step(action)
        if render:
            env.render()
        image_list.append(new_state.copy())
        image_reward += reward
        if done:
            num_frames_to_repeat = num_frames_to_collapse-i-1
            ## if we fail during the frames to collapse, we will simply append the last frame enough times to preserve shape
            for i in range(num_frames_to_repeat):
                image_list.append(new_state.copy())
                image_reward += reward
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

def check_stuck_and_penalize(current_x_position, current_position, reward): #I don't think this will be needed anymore. It's part of the reward.
    if len(current_x_position) > 250:
        avg_position = round(np.array(current_x_position[-250:]).mean(), 0)
        #current_position = info['x_pos']
        if current_position - avg_position < 1:  ## No real movement
            reward += -5  # make it negative so that it doesn't get stuck and tries new things
            reward = max(reward, -15)  # can't go below -15
    return reward

def convert_to_int(value):
    try:
        return(int(value))
    except Exception as e:
        return None

def save_bitmap_data(data,fpath, frame_idx):
    from matplotlib import pyplot as plt
    os.makedirs(fpath, exist_ok=True)
    plt.imsave(fname=fpath + "/" + str(frame_idx) + ".png", arr=data, cmap="gray")

def show_bitmap_data(data):
    from matplotlib import pyplot as plt
    plt.imshow(X=data, cmap="gray")
    plt.show()
