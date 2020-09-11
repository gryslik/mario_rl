import collections
import tensorflow as tf
import numpy as np
import random
import time


class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

class DQN:
    def __init__(self, env, single_frame_dim,  num_frames_to_stack, old_model_filepath=None):
        self.env = env

        #self.memory = collections.deque(maxlen=10000) #this is slow. Using my custom class.
        self.burnin = 3000
        self.memory = ReplayMemory(max_size=30000)

        self.gamma = 0.9
        self.epsilon = 1.0
        # self.epsilon = 0.2 ## when restarting training...(should automate this)
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.999995
        self.learning_rate = 0.00025
        self.update_target_step_count = 5000
        self.num_steps_since_last_update = 0
        self.single_frame_dim = single_frame_dim
        self.num_frames_to_stack = num_frames_to_stack
        # self.learn_each = 3 ## learn every 4th frame
        # self.learn_step = 0

        if(old_model_filepath==None):
            self.model = self.create_model()  # Will do the actual predictions
            self.target_model = self.create_model()  # Will compute what action we DESIRE from our model
        else:
            self.model = tf.keras.models.load_model(old_model_filepath)
            self.target_model = tf.keras.models.load_model(old_model_filepath)
        # Otherwise we are changing the goal at each timestep.

    def update_target_model(self, current_episode, current_step, update_by_episode_count, force_update):

        self.num_steps_since_last_update +=1
        # for smaller problems you might want to update just by episode. For larger problems, usually cap it at 5k or max for episode or whatever heuristic you prefer.
        if force_update or (update_by_episode_count and current_step == 0) or (not update_by_episode_count and (self.num_steps_since_last_update == self.update_target_step_count)):
            print("Updating_target_model at episode/step: " + str(current_episode) + " / " +str(current_step))
            self.target_model.set_weights(self.model.get_weights())
            self.num_steps_since_last_update = 0

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, 8, 4, input_shape=(self.single_frame_dim[0], self.single_frame_dim[1], self.num_frames_to_stack), activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 4, 2, activation="relu"))
        model.add(tf.keras.layers.Conv2D(64, 3, 1, activation="relu"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation="relu"))
        model.add(tf.keras.layers.Dense(self.env.action_space.n))
        # model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate)) ## huber loss takes advantage of l1/l2
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size=32, print_time=False):
        if self.memory.size < self.burnin:
            return
        # Break if no training (borrowed from https://towardsdatascience.com/using-reinforcement-learning-to-play-super-mario-bros-on-nes-using-tensorflow-31281e35825)

        # if self.learn_step < self.learn_each:
        #     self.learn_step += 1
        #     return

        samples = self.memory.sample(batch_size)

        ########################
        #Setting Up Data
        ########################
        all_states = np.reshape([np.squeeze(x[0]) for x in samples], (batch_size, self.single_frame_dim[0],  self.single_frame_dim[1], self.num_frames_to_stack))
        all_actions = np.reshape([x[1] for x in samples], (batch_size,))
        all_rewards = np.reshape([x[2] for x in samples], (batch_size,))
        all_new_states = np.reshape([np.squeeze(x[3]) for x in samples], (batch_size, self.single_frame_dim[0], self.single_frame_dim[1], self.num_frames_to_stack))
        all_dones = np.reshape([x[4] for x in samples], (batch_size,))

        ####################################
        # Doing the prediction and updating rewards
        ####################################
        all_targets = np.array(self.model.predict_on_batch(all_states.astype('float16')))  # this is what we will update
        Q_0 = np.array(self.model.predict_on_batch(all_new_states.astype('float16')))  # This is what we use to find what max action we should take
        Q_target = np.array(self.target_model.predict_on_batch(all_new_states.astype('float16')))  # This is the values we will combine with max action to update the target

        max_actions = np.argmax(Q_0, axis=1)  # This is the index we will use to take from Q_target
        max_Q_target_values = Q_target[np.arange(len(Q_target)), np.array(max_actions)]  # The target will be updated with this.
        all_targets[np.arange(len(all_targets)), np.array(all_actions)] = all_rewards + self.gamma * max_Q_target_values * (~all_dones)  # Actually do the update

        ######
        # Training
        self.model.train_on_batch(all_states.astype('float16'), all_targets)  # reweight network to get new targets
        ######

        # # Reset learn step
        # self.learn_step = 0

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            reshaped_state = np.expand_dims(state, axis=0).astype('float16')
            return np.argmax(self.model.predict(reshaped_state)[0]) #The predict returns a (1,7)

    def save_model(self, fn):
        self.model.save(fn)
