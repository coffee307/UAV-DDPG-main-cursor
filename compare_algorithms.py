"""
算法对比程序
读取DDPG和DQN的每10个episode平均总时延数据，并绘制对比图
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_plot_comparison():
    """加载数据并绘制对比图"""
    
    # 文件路径
    ddpg_file = 'ddpg_average_delay_per_10.npy'
    dqn_file = 'dqn_average_delay_per_10.npy'
    
    # 检查文件是否存在
    ddpg_exists = os.path.exists(ddpg_file)
    dqn_exists = os.path.exists(dqn_file)
    
    if not ddpg_exists and not dqn_exists:
        print("错误: 未找到任何数据文件！")
        print(f"  请确保以下文件存在:")
        print(f"    - {ddpg_file}")
        print(f"    - {dqn_file}")
        return
    
    # 加载DDPG数据
    ddpg_data = None
    if ddpg_exists:
        try:
            ddpg_data = np.load(ddpg_file)
            print(f"✓ 成功加载DDPG数据: {len(ddpg_data)} 个数据点")
        except Exception as e:
            print(f"✗ 加载DDPG数据失败: {e}")
            ddpg_exists = False
    
    # 加载DQN数据
    dqn_data = None
    if dqn_exists:
        try:
            dqn_data = np.load(dqn_file)
            print(f"✓ 成功加载DQN数据: {len(dqn_data)} 个数据点")
        except Exception as e:
            print(f"✗ 加载DQN数据失败: {e}")
            dqn_exists = False
    
    if not ddpg_exists and not dqn_exists:
        print("错误: 无法加载任何数据！")
        return
    
    # 绘制对比图
    plt.figure(figsize=(12, 7))
    
    if ddpg_exists and ddpg_data is not None:
        x_ddpg = np.arange(len(ddpg_data)) * 10  # 每个点代表10个episode
        plt.plot(x_ddpg, ddpg_data, marker='o', linestyle='-', color='r', 
                 label='DDPG', linewidth=2, markersize=4, alpha=0.8, markevery=max(1, len(ddpg_data)//20))
    
    if dqn_exists and dqn_data is not None:
        x_dqn = np.arange(len(dqn_data)) * 10  # 每个点代表10个episode
        plt.plot(x_dqn, dqn_data, marker='s', linestyle='-', color='b', 
                 label='DQN', linewidth=2, markersize=4, alpha=0.8, markevery=max(1, len(dqn_data)//20))
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Total Delay (per 10 episodes)", fontsize=12)
    plt.title("Average Total Delay Comparison: DDPG vs DQN", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # 保存图片
    output_file = 'algorithm_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比图已保存到: {output_file}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("性能统计 (每10个episode的平均总时延)")
    print("=" * 60)
    
    if ddpg_exists and ddpg_data is not None:
        print(f"\nDDPG:")
        print(f"  数据点数: {len(ddpg_data)}")
        print(f"  最小延迟: {min(ddpg_data):.2f}")
        print(f"  最大延迟: {max(ddpg_data):.2f}")
        print(f"  平均延迟: {np.mean(ddpg_data):.2f}")
        print(f"  最终延迟: {ddpg_data[-1]:.2f}")
        print(f"  标准差: {np.std(ddpg_data):.2f}")
    
    if dqn_exists and dqn_data is not None:
        print(f"\nDQN:")
        print(f"  数据点数: {len(dqn_data)}")
        print(f"  最小延迟: {min(dqn_data):.2f}")
        print(f"  最大延迟: {max(dqn_data):.2f}")
        print(f"  平均延迟: {np.mean(dqn_data):.2f}")
        print(f"  最终延迟: {dqn_data[-1]:.2f}")
        print(f"  标准差: {np.std(dqn_data):.2f}")
    
    # 对比分析
    if ddpg_exists and dqn_exists and ddpg_data is not None and dqn_data is not None:
        print(f"\n对比分析:")
        if min(dqn_data) > 0:
            improvement = (min(dqn_data) - min(ddpg_data)) / min(dqn_data) * 100
            print(f"  DDPG相比DQN的最小延迟改进: {improvement:.2f}%")
        
        avg_improvement = (np.mean(dqn_data) - np.mean(ddpg_data)) / np.mean(dqn_data) * 100
        print(f"  DDPG相比DQN的平均延迟改进: {avg_improvement:.2f}%")
        
        final_improvement = (dqn_data[-1] - ddpg_data[-1]) / dqn_data[-1] * 100
        print(f"  DDPG相比DQN的最终延迟改进: {final_improvement:.2f}%")
    
    print("=" * 60)
    
    # 显示图片
    plt.show()

if __name__ == '__main__':
    print("=" * 60)
    print("算法对比程序")
    print("=" * 60)
    print("\n正在加载数据并生成对比图...\n")
    
    load_and_plot_comparison()
    
    print("\n程序执行完成！")

