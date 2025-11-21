"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.

Using:
tensorflow 1.14.0
gym 0.15.3
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from UAV_env import UAVEnv
import time
import matplotlib.pyplot as plt
from state_normalization import StateNormalization

#####################  hyper parameters  ####################
MAX_EPISODES = 2000
# MAX_EPISODES = 50000

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
# LR_A = 0.1  # learning rate for actor
# LR_C = 0.2  # learning rate for critic
GAMMA = 0.001  # optimal reward discount
# GAMMA = 0.999  # reward discount
TAU = 0.01  # soft replacement
VAR_MIN = 0.01
# MEMORY_CAPACITY = 5000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
OUTPUT_GRAPH = False


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(ddpg, eval_episodes=10):
    # eval_env = gym.make(env_name)
    eval_env = UAVEnv()
    # eval_env.seed(seed + 100)
    avg_reward = 0.
    for i in range(eval_episodes):
        state = eval_env.reset()
        # while not done:
        for j in range(int(len(eval_env.UE_loc_list))):
            action = ddpg.choose_action(state)
            action = np.clip(action, *a_bound)
            state, reward = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)  # memory里存放当前和下一个state，动作和奖励
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')  # 输入
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return temp[0]

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # transition = np.hstack((s, [a], [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################
np.random.seed(1)
tf.set_random_seed(1)

env = UAVEnv()
MAX_EP_STEPS = env.slot_num
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound  # [-1,1]

ddpg = DDPG(a_dim, s_dim, a_bound)

# var = 1  # control exploration
var = 0.01  # control exploration
t1 = time.time()
ep_reward_list = []
ep_total_delay_list = []
s_normal = StateNormalization()

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    ep_offload_sum = 0.0

    j = 0
    while j < MAX_EP_STEPS:
        # Add exploration noise
        a = ddpg.choose_action(s_normal.state_normal(s))
        a = np.clip(np.random.normal(a, var), *a_bound)  # 高斯噪声add randomness to action selection for exploration
        a_sent = a.copy()
        s_, r, is_terminal, step_redo, offloading_ratio_change, reset_dist = env.step(a)
        if step_redo:
            continue
        if reset_dist:
            a[2] = -1
        if offloading_ratio_change:
            a[3] = -1
        # 记录本步使用的卸载率（若被环境强制重置，则记为0；否则取发送给环境的a_sent[3]并截断到[0,1]）
        off_ratio_used = 0.0 if offloading_ratio_change else float(np.clip(a_sent[3], 0.0, 1.0))
        ep_offload_sum += off_ratio_used
        ddpg.store_transition(s_normal.state_normal(s), a, r, s_normal.state_normal(s_))  # 训练奖励缩小10倍

        if ddpg.pointer > MEMORY_CAPACITY:
            # var = max([var * 0.9997, VAR_MIN])  # decay the action randomness
            ddpg.learn()
        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS - 1 or is_terminal:
            total_delay = -ep_reward
            avg_delay = total_delay / (j + 1 if j >= 0 else 1)
            avg_offloading_ratio = ep_offload_sum / (j + 1 if j >= 0 else 1)
            print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward, ' total_delay: %7.2f' % total_delay, ' avg_delay: %7.4f' % avg_delay, ' avg_offloading_ratio: %.3f' % avg_offloading_ratio, ' Explore: %.3f' % var)
            ep_reward_list = np.append(ep_reward_list, ep_reward)
            ep_total_delay_list = np.append(ep_total_delay_list, total_delay)
            # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
            file_name = 'ddpg_f_ue6e8_sun_task_size60/output.txt'
            with open(file_name, 'a') as file_obj:
                file_obj.write("\n======== This episode is done ========")  # 本episode结束
            break
        j = j + 1

    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #     eval_policy(ddpg, env)

print('Running time: ', time.time() - t1)

# 绘制Reward图
plt.figure(1)
window = 20
if len(ep_reward_list) >= window:
    kernel = np.ones(window) / window
    smooth_reward = np.convolve(ep_reward_list, kernel, mode='valid')
    x_smooth = np.arange(window - 1, window - 1 + len(smooth_reward))
    plt.plot(ep_reward_list, color='lightcoral', linewidth=1, alpha=0.5, label='Raw')
    plt.plot(x_smooth, smooth_reward, color='crimson', linewidth=2, label='Smoothed (MA-20)')
    plt.legend()
else:
    plt.plot(ep_reward_list, color='crimson', linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("history_ddpg/ddpg_reward.png")
plt.show()

# 绘制Total Delay图（与DQN保持一致）
plt.figure(2)
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
# 保存total_delay数据供对比使用
np.save("history_ddpg/ddpg_total_delay.npy", ep_total_delay_list)
plt.savefig("history_ddpg/ddpg_total_delay.png")
plt.show()
