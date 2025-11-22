"""
This part of code is the Deep Q Network (DQN) brain.

view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: r1.2
"""

import numpy as np
import tensorflow.compat.v1 as tf
import os
from UAV_env import UAVEnv
import time
from state_normalization import StateNormalization

tf.disable_v2_behavior()
MAX_EPISODES = 1000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,# 动作维度
            n_features,# 状态特征维度
            learning_rate=0.01,# 学习率 α
            reward_decay=0.9  ,# 奖励折扣 γ
            e_greedy=0.9,
            replace_target_iter=200,# 每多少步更新一次 target 网络参数
            memory_size=MEMORY_CAPACITY,# 经验回放池容量
            batch_size=BATCH_SIZE, # 每次从经验池采样的 batch 大小
            e_greedy_increment=1e-4,# ε 从小变大的增量（如果用逐步加探索）
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # self.epsilon = 0.99
        # self.epsilon = 0.9

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # memory里存放当前和下一个state，动作和奖励
        self.memory = np.zeros((MEMORY_CAPACITY, self.n_features * 2 + 2), dtype=np.float32)
        # consist of [target_net, evaluate_net]
        # eval_net：当前用于选择动作、计算 TD 目标的网络
        # target_net：延迟更新的目标网络，提供更稳定的目标 Q 值
        self._build_net()

        # e_params：eval_net 的所有可训练参数
        # t_params：target_net 的所有可训练参数
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        # target_param = eval_param
        # 每隔 replace_target_iter 步，把 target_net 的所有参数直接拷贝为 eval_net 的参数
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        # 初始化所有变量（网络权重、偏置、计数器等）。
        # cost_his 用来记录每次 learn() 的 loss（TD error），方便之后画 loss 曲线看有没有收敛。
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            # e2 = tf.layers.dense(e1, 48, tf.nn.relu6, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='e2')
            e3 = tf.layers.dense(e1, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e3')
            self.q_eval = tf.layers.dense(e3, self.n_actions, None, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 100, tf.nn.relu6, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            # t2 = tf.layers.dense(t1, 48, tf.nn.relu6, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='t2')
            t3 = tf.layers.dense(t1, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t3')
            self.q_next = tf.layers.dense(t3, self.n_actions, None, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t4')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            # self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


if __name__ == '__main__':

    env = UAVEnv()
    Normal = StateNormalization()  # 输入状态归一化
    np.random.seed(1)
    tf.set_random_seed(1)
    s_dim = env.state_dim
    n_actions = env.n_actions
    a_bound = env.action_bound
    DQN = DeepQNetwork(n_actions, s_dim, output_graph=False)
    t1 = time.time()
    ep_reward_list = []
    MAX_EP_STEPS = env.slot_num
    for i in range(MAX_EPISODES):
        # initial observation
        s = env.reset()
        ep_reward = 0
        j = 0
        while j < MAX_EP_STEPS:

            # RL choose action based on observation
            a = DQN.choose_action(Normal.state_normal(s))
            # RL take action and get next observation and reward
            s_, r, is_terminal, step_redo, reset_offload_ratio = env.step(a)
            if step_redo:
                continue
            if reset_offload_ratio:
                # 卸载比率重新设置为0
                offload_remainder = a % 11
                a = a - offload_remainder
            DQN.store_transition(Normal.state_normal(s), a, r, Normal.state_normal(s_))

            if DQN.memory_counter > MEMORY_CAPACITY:
                # learn()方法内部会更新epsilon，这里不需要重复更新
                DQN.learn()

            # swap observation
            s = s_
            ep_reward += r

            if j == MAX_EP_STEPS - 1 or is_terminal:
                print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, 'Epsilon: %.3f' % DQN.epsilon)
                ep_reward_list = np.append(ep_reward_list, ep_reward)
                file_name = 'output.txt'
                with open(file_name, 'a') as file_obj:
                    file_obj.write("\n======== This episode is done ========")  # 本episode结束
                break

            j = j+1

# 计算每10个episode的平均总时延（用于后续对比）
average_delays_per_10 = []  # 存储每10个episode的平均时延
sub_group_size = 10  # 每10个episode一组

num_sub_groups = len(ep_reward_list) // sub_group_size + (1 if len(ep_reward_list) % sub_group_size else 0)

for sub in range(num_sub_groups):
    sub_start = sub * sub_group_size
    sub_end = min(sub_start + sub_group_size, len(ep_reward_list))
    sub_rewards = ep_reward_list[sub_start:sub_end]
    if len(sub_rewards) > 0:
        avg_delay = -np.mean(sub_rewards)  # 奖励是负的延迟
        average_delays_per_10.append(avg_delay)

# 保存每10个episode的平均总时延数据
# 使用绝对路径保存到项目根目录，确保无论从哪个目录运行都能找到文件
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # 上一级目录（项目根目录）
output_file = os.path.join(project_root, "dqn_average_delay_per_10.npy")
np.save(output_file, np.array(average_delays_per_10))
print(f"已保存每10个episode的平均总时延数据: {len(average_delays_per_10)} 个数据点")
print(f"文件保存位置: {output_file}")
