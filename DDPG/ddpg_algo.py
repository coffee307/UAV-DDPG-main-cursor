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
import os
from state_normalization import StateNormalization

#####################  hyper parameters  ####################
MAX_EPISODES = 1000
# MAX_EPISODES = 50000

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
# LR_A = 0.1  # learning rate for actor
# LR_C = 0.2  # learning rate for critic
GAMMA = 0.9  # optimal reward discount (调整到更合理的值，考虑长期奖励)
# GAMMA = 0.999  # reward discount
TAU = 0.005  # soft replacement (降低更新频率，使目标网络更稳定)
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
        # 添加模型保存相关的变量
        self.saver = None

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
        
        # 创建模型保存器
        self.saver = tf.train.Saver(max_to_keep=1)

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)
    
    def save_model(self, path='best_model/ddpg_model'):
        """保存模型"""
        if self.saver is not None:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.saver.save(self.sess, path)
            print(f"模型已保存到: {path}")
    
    def load_model(self, path='best_model/ddpg_model'):
        """加载模型"""
        if self.saver is not None:
            try:
                self.saver.restore(self.sess, path)
                print(f"模型已从 {path} 加载")
                return True
            except Exception as e:
                print(f"加载模型失败: {e}")
                return False
        return False

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
var_decay = 0.9995  # 探索噪声衰减率（更慢的衰减）
var_min = 0.005  # 最小探索噪声（保持一定探索）
t1 = time.time()
ep_reward_list = []
ep_total_delay_list = []
s_normal = StateNormalization()

# 添加最佳性能跟踪
best_avg_delay = float('inf')
best_episode = 0
patience = 50  # 如果连续50个episode性能下降超过20%，则回退到最佳模型
no_improve_count = 0
performance_window = 20  # 计算最近20个episode的平均延迟来判断性能
stability_threshold = 1.15  # 如果当前性能比最佳性能差15%以上，则回退

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
            # 缓慢衰减探索噪声，但保持最小探索
            var = max([var * var_decay, var_min])
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
            
            # 性能监控和最佳模型保存
            if len(ep_total_delay_list) >= performance_window:
                # 计算最近performance_window个episode的平均延迟
                recent_avg_delay = np.mean(ep_total_delay_list[-performance_window:])
                
                # 如果找到更好的性能，保存模型
                if recent_avg_delay < best_avg_delay:
                    best_avg_delay = recent_avg_delay
                    best_episode = i
                    no_improve_count = 0
                    # 保存最佳模型
                    ddpg.save_model('best_model/ddpg_model')
                    print(f'*** 新的最佳性能! 平均延迟: {best_avg_delay:.2f}, Episode: {i} ***')
                else:
                    no_improve_count += 1
                    # 如果性能持续下降超过阈值，回退到最佳模型
                    if recent_avg_delay > best_avg_delay * stability_threshold:
                        if no_improve_count >= patience:
                            print(f'性能下降超过阈值，回退到Episode {best_episode}的最佳模型 (延迟: {best_avg_delay:.2f})')
                            ddpg.load_model('best_model/ddpg_model')
                            # 降低探索噪声，专注于利用
                            var = var_min
                            no_improve_count = 0
                        elif no_improve_count % 10 == 0:
                            # 每10个episode检查一次，如果性能仍然很差，提前回退
                            print(f'警告: 性能下降，当前平均延迟: {recent_avg_delay:.2f}, 最佳: {best_avg_delay:.2f}')
            
            # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
            file_name = 'output.txt'
            with open(file_name, 'a') as file_obj:
                file_obj.write("\n======== This episode is done ========")  # 本episode结束
            break
        j = j + 1

    # # Evaluate episode
    # if (i + 1) % 50 == 0:
    #     eval_policy(ddpg, env)
    
    # 如果已经找到最佳性能且持续没有改进，可以考虑提前停止
    # 但这里我们继续训练，只是会回退到最佳模型

print('Running time: ', time.time() - t1)
print(f'最佳性能: 平均延迟 {best_avg_delay:.2f}, 出现在 Episode {best_episode}')

# 训练结束后，加载最佳模型用于最终评估
if best_episode > 0:
    print('加载最佳模型进行最终评估...')
    ddpg.load_model('best_model/ddpg_model')

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
output_file = os.path.join(project_root, "ddpg_average_delay_per_10.npy")
np.save(output_file, np.array(average_delays_per_10))
print(f"已保存每10个episode的平均总时延数据: {len(average_delays_per_10)} 个数据点")
print(f"文件保存位置: {output_file}")
