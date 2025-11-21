"""
Improved DQN training script based on your original code.
Main fixes:
- Reasonable defaults: learning_rate, gamma, epsilon schedule
- Memory initialization uses n_features
- Q-network outputs linear Q-values (no softmax)
- Safe sampling/learning logic and epsilon increment
- Avoid in-place modification of states when normalizing
"""

import numpy as np
import tensorflow.compat.v1 as tf
from UAV_env import UAVEnv
import time
from state_normalization import StateNormalization
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

MAX_EPISODES = 2000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=1e-3,        # smaller lr for Adam
            gamma=0.99,                # discount factor
            e_greedy_max=0.99,
            replace_target_iter=200,
            memory_size=MEMORY_CAPACITY,
            batch_size=BATCH_SIZE,
            e_greedy_increment=1e-4,   # epsilon increment per learn step
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon_max = e_greedy_max
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment

        # epsilon: start with some exploration
        self.epsilon = 0.1

        # total learning step
        self.learn_step_counter = 0

        # initialize memory: each row stores [s (n_features), a (1), r (1), s_ (n_features)]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2), dtype=np.float32)

        # build nets
        self._build_net()

        # target / eval params
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

        # memory counter
        self.memory_counter = 0

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')   # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_') # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')                 # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')                   # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e3 = tf.layers.dense(e1, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e3')
            # Q values should be linear outputs (no softmax)
            self.q_eval = tf.layers.dense(e3, self.n_actions, activation=None,
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 100, tf.nn.relu6, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t3 = tf.layers.dense(t1, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t3')
            self.q_next = tf.layers.dense(t3, self.n_actions, activation=None,
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t4')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None,)
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None,)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        # s, s_ expected to be 1-D arrays of length n_features
        if s.ndim != 1:
            s = s.flatten()
        if s_.ndim != 1:
            s_ = s_.flatten()
        transition = np.hstack((s, a, [r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # observation: 1-D array of length n_features
        observation = observation[np.newaxis, :]
        # greedy with probability epsilon
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value.flatten())
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # only learn if we have enough samples
        if self.memory_counter < self.batch_size:
            self.learn_step_counter += 1  # 即使不学习，也要增加计数器以保持同步
            return

        # replace target params periodically
        # 在learn_step_counter增加之前检查，这样每个倍数只会更新一次
        # 例如：learn_step_counter=199时调用，检查199%200!=0，不更新，然后变成200
        #       learn_step_counter=200时调用，检查200%200==0，更新，然后变成201
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # optional: print replacement info
            print('\n[target_params_replaced]\n')

        # sample batch memory (if buffer not full, sample from [0, memory_counter))
        sample_size = min(self.batch_size, self.memory_counter)
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), size=sample_size)
        batch_memory = self.memory[sample_index, :]

        feed_dict = {
            self.s: batch_memory[:, :self.n_features],
            self.a: batch_memory[:, self.n_features].astype(np.int32),
            self.r: batch_memory[:, self.n_features + 1],
            self.s_: batch_memory[:, -self.n_features:],
        }

        _, cost = self.sess.run([self._train_op, self.loss], feed_dict=feed_dict)
        self.cost_his.append(cost)

        # increase epsilon
        if self.epsilon < self.epsilon_max:
            self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)

        # 增加学习步数计数器（在更新检查之后）
        self.learn_step_counter += 1


if __name__ == '__main__':

    env = UAVEnv()
    Normal = StateNormalization()  # 输入状态归一化
    np.random.seed(1)
    tf.set_random_seed(1)

    s_dim = env.state_dim
    n_actions = env.n_actions

    DQN = DeepQNetwork(n_actions, s_dim, output_graph=False)

    t_start = time.time()
    ep_reward_list = np.array([])         # use numpy arrays for append
    ep_total_delay_list = np.array([])
    MAX_EP_STEPS = env.slot_num

    for i in range(MAX_EPISODES):
        # initial observation
        s = env.reset()
        ep_reward = 0.0
        ep_offload_sum = 0.0
        j = 0

        while j < MAX_EP_STEPS:
            # use copy to avoid in-place modification if normalization does that
            s_norm = Normal.state_normal(s.copy())
            a = DQN.choose_action(s_norm)

            # take action
            s_, r, is_terminal, step_redo, reset_offload_ratio = env.step(a)
            if step_redo:
                # skip this step, do not increment j, state stays the same
                continue

            if reset_offload_ratio:
                # adjust action as in original logic
                t1 = a % 11
                a = a - t1

            t1_used = a % 11
            off_ratio_used = 0.0 if reset_offload_ratio else (t1_used * 0.1)
            ep_offload_sum += off_ratio_used

            # store transition with normalized states (use copy for safety)
            s__norm = Normal.state_normal(s_.copy())
            DQN.store_transition(s_norm, a, r, s__norm)

            # start learning when batch is available
            DQN.learn()

            # swap observation
            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS - 1 or is_terminal:
                total_delay = -ep_reward
                avg_delay = total_delay / (j + 1 if j >= 0 else 1)
                avg_offloading_ratio = ep_offload_sum / (j + 1 if j >= 0 else 1)
                print('Episode:', i, ' Steps: %2d' % j,
                      ' Reward: %7.2f' % ep_reward, ' total_delay: %7.2f' % total_delay,
                      ' avg_delay: %7.4f' % avg_delay, ' avg_offloading_ratio: %.3f' % avg_offloading_ratio,
                      ' Explore: %.3f' % DQN.epsilon)
                ep_reward_list = np.append(ep_reward_list, ep_reward)
                ep_total_delay_list = np.append(ep_total_delay_list, -ep_reward)
                file_name = 'output_gpt.txt'
                with open(file_name, 'a') as file_obj:
                    file_obj.write("\n======== This episode is done ========")
                break

            j += 1

    print('Running time: ', time.time() - t_start)
    np.save("history_dqn/dqn_total_delay_gpt.npy", ep_total_delay_list)

    # smoothing
    window = 20
    if len(ep_total_delay_list) >= window:
        kernel = np.ones(window) / window
        smooth_delay = np.convolve(ep_total_delay_list, kernel, mode='valid')
        x_smooth = np.arange(window - 1, window - 1 + len(smooth_delay))
        plt.plot(ep_total_delay_list, color='lightsteelblue', linewidth=1, alpha=0.5, label='Raw')
        plt.plot(x_smooth, smooth_delay, color='steelblue', linewidth=2, label='Smoothed (MA-20)')
        plt.legend()
    else:
        plt.plot(ep_total_delay_list, color='steelblue', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Delay")
    plt.savefig("history_dqn/dqn_total_delay_gpt.png")
    plt.show()
