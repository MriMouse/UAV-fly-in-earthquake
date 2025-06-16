"""
贪婪策略基线算法 (Greedy Heuristic Baseline)
每个UAV在每个时间步都飞向距离最近且有任务的UGV，并卸载所有可卸载数据
用于与MAPPO和MADDPG算法进行性能对比
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import sys

# 获取当前文件的上层目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from env import Env_MAPPO

# ==========================
#      配置参数
# ==========================
NUM_EPISODES = 100  # 测试回合数
OUTPUT_DIR = "baseline_results_mappo"  # 结果保存目录
RESULT_FILE = "greedy_baseline_results.txt"


def get_greedy_actions(env: Env_MAPPO) -> list:
    """
    为所有UAV生成贪婪策略动作:
    每个UAV飞向距离最近且有任务的UGV，并卸载所有可用数据

    Args:
        env: 环境实例

    Returns:
        actions_list: 每个UAV的动作列表
    """
    actions_list = []

    # 找出所有有数据的UGV
    ugvs_with_data = []
    for ugv_id in range(env.num_ugvs):
        if env.ugv_data_queue[ugv_id] > 1e-6:
            ugvs_with_data.append(ugv_id)

    # 为每个UAV生成动作
    for uav_id in range(env.num_uavs):
        action = np.zeros(env.action_dimension, dtype=np.float32)

        if len(ugvs_with_data) == 0:
            # 没有UGV有数据，选择默认动作（悬停，不卸载）
            action[0] = -1.0  # 目标UGV ID (归一化到[-1,1])
            action[1] = 0.0  # 飞行角度
            action[2] = -1.0  # 飞行距离比例（0%，保持原地）
            action[3] = -1.0  # 卸载比例（0%）
        else:
            # 找到距离最近且有数据的UGV
            uav_pos = env.uav_positions[uav_id]
            min_distance = float("inf")
            nearest_ugv_id = ugvs_with_data[0]

            for ugv_id in ugvs_with_data:
                ugv_pos = env.ugv_positions[ugv_id]
                # 计算2D距离（UAV在固定高度飞行）
                distance = np.linalg.norm(uav_pos[:2] - ugv_pos[:2])
                if distance < min_distance:
                    min_distance = distance
                    nearest_ugv_id = ugv_id

            # 计算朝向最近UGV的移动方向
            ugv_pos = env.ugv_positions[nearest_ugv_id]
            direction_vector = ugv_pos[:2] - uav_pos[:2]
            distance_to_ugv = np.linalg.norm(direction_vector)

            if distance_to_ugv > 1e-6:
                # 需要移动，计算飞行角度
                angle_to_ugv = np.arctan2(direction_vector[1], direction_vector[0])
                # 将角度归一化到[0, 2π]再映射到[0, 1]
                angle_normalized = (angle_to_ugv % (2 * np.pi)) / (2 * np.pi)
                # 以最大速度飞向UGV
                fly_distance_ratio = 1.0
            else:
                # 已在UGV位置附近
                angle_normalized = 0.0
                fly_distance_ratio = 0.0

            # 设置动作
            # 目标UGV ID (归一化到[-1, 1])
            target_ugv_normalized = (2.0 * nearest_ugv_id / max(1, env.num_ugvs - 1) - 1.0) if env.num_ugvs > 1 else 0.0
            target_ugv_normalized = np.clip(target_ugv_normalized, -1.0, 1.0)

            action[0] = target_ugv_normalized  # 目标UGV索引
            action[1] = 2.0 * angle_normalized - 1.0  # 飞行角度（归一化到[-1,1]）
            action[2] = 2.0 * fly_distance_ratio - 1.0  # 飞行距离比例
            action[3] = 1.0  # 卸载100%可用数据

        actions_list.append(action)

    return actions_list


def run_greedy_baseline(env: Env_MAPPO, num_episodes: int) -> tuple:
    """
    运行贪婪策略基线测试

    Args:
        env: 环境实例
        num_episodes: 测试回合数

    Returns:
        avg_reward: 平均奖励
        std_reward: 奖励标准差
        all_rewards: 所有回合奖励列表
    """
    all_episode_rewards = []
    print(f"\n开始运行贪婪策略基线测试...")
    print(f"测试回合数: {num_episodes}")

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        # 重置环境
        states_list = env.reset()
        episode_reward = 0.0
        step_count = 0
        done = False

        # 运行一个回合
        while not done and step_count < env.max_steps:
            # 获取贪婪策略动作
            actions_list = get_greedy_actions(env)

            # 执行动作
            next_states_list, rewards_list, dones_list, infos_list = env.step(actions_list)

            # 累加奖励（使用共享奖励）
            step_reward = rewards_list[0] if rewards_list else 0.0
            episode_reward += step_reward

            # 检查是否结束
            done = dones_list[0] if dones_list else True
            step_count += 1

            # 更新状态
            states_list = next_states_list

        all_episode_rewards.append(episode_reward)

        # 打印进度
        if episode % max(1, num_episodes // 10) == 0:
            print(f"  第 {episode}/{num_episodes} 回合完成，奖励: {episode_reward:.2f}, 步数: {step_count}")

    end_time = time.time()

    # 计算统计结果
    avg_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)

    print(f"\n贪婪策略基线测试完成!")
    print(f"总用时: {end_time - start_time:.2f} 秒")
    print(f"平均奖励: {avg_reward:.4f}")
    print(f"奖励标准差: {std_reward:.4f}")
    print(f"最高奖励: {max(all_episode_rewards):.4f}")
    print(f"最低奖励: {min(all_episode_rewards):.4f}")

    return avg_reward, std_reward, all_episode_rewards


def save_results(avg_reward, std_reward, all_rewards, output_path):
    """保存结果到文件"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("贪婪策略基线算法测试结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"测试回合数: {len(all_rewards)}\n")
            f.write(f"平均奖励: {avg_reward:.6f}\n")
            f.write(f"奖励标准差: {std_reward:.6f}\n")
            f.write(f"最高奖励: {max(all_rewards):.6f}\n")
            f.write(f"最低奖励: {min(all_rewards):.6f}\n")
            f.write("\n策略描述:\n")
            f.write("每个UAV在每个时间步都飞向距离最近且有任务的UGV，\n")
            f.write("并尝试卸载所有可用数据。这是一个简单的贪婪策略，\n")
            f.write("用作与MAPPO和MADDPG等强化学习算法的性能对比基线。\n")
            f.write("\n详细奖励数据:\n")
            for i, reward in enumerate(all_rewards, 1):
                f.write(f"回合 {i}: {reward:.6f}\n")

        print(f"\n结果已保存到: {output_path}")
    except Exception as e:
        print(f"\n保存结果时出错: {e}")


def plot_results(all_rewards, output_path):
    """绘制并保存结果图表"""
    try:
        plt.figure(figsize=(12, 8))

        # 绘制奖励曲线
        episodes = range(1, len(all_rewards) + 1)
        plt.subplot(2, 1, 1)
        plt.plot(episodes, all_rewards, "b-", alpha=0.7, linewidth=1)
        plt.axhline(y=np.mean(all_rewards), color="r", linestyle="--", label=f"平均奖励: {np.mean(all_rewards):.2f}")
        plt.title("贪婪策略基线 - 回合奖励变化")
        plt.xlabel("回合数")
        plt.ylabel("回合奖励")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 绘制移动平均（如果回合数足够）
        if len(all_rewards) >= 20:
            window = max(10, len(all_rewards) // 10)
            moving_avg = []
            for i in range(len(all_rewards)):
                start_idx = max(0, i - window + 1)
                moving_avg.append(np.mean(all_rewards[start_idx : i + 1]))

            plt.subplot(2, 1, 2)
            plt.plot(episodes, moving_avg, "g-", linewidth=2, label=f"移动平均 (窗口={window})")
            plt.axhline(y=np.mean(all_rewards), color="r", linestyle="--", label=f"总平均: {np.mean(all_rewards):.2f}")
            plt.title("贪婪策略基线 - 移动平均奖励")
            plt.xlabel("回合数")
            plt.ylabel("移动平均奖励")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path.replace(".txt", ".png"), dpi=300, bbox_inches="tight")
        print(f"结果图表已保存到: {output_path.replace('.txt', '.png')}")
        plt.close()

    except Exception as e:
        print(f"\n绘制图表时出错: {e}")


# ==========================
#       主程序
# ==========================

if __name__ == "__main__":
    print("贪婪策略基线算法测试")
    print("=" * 50)

    # 创建输出目录
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    output_file = os.path.join(output_dir, RESULT_FILE)

    # 初始化环境
    env = Env_MAPPO()
    print(f"\n环境初始化完成:")
    print(f"  UAV数量: {env.num_uavs}")
    print(f"  UGV数量: {env.num_ugvs}")
    print(f"  区域大小: {env.area_side_length}x{env.area_side_length}")
    print(f"  最大步数: {env.max_steps}")
    print(f"  动作维度: {env.action_dimension}")
    print(f"  状态维度: {env.state_dimension}")

    # 运行贪婪策略基线测试
    avg_reward, std_reward, all_rewards = run_greedy_baseline(env, NUM_EPISODES)

    # 保存结果
    save_results(avg_reward, std_reward, all_rewards, output_file)

    # 绘制结果图表
    plot_results(all_rewards, output_file)

    print(f"\n测试完成! 可以使用此结果与MAPPO和MADDPG算法进行对比。")
    print(f"贪婪策略平均奖励: {avg_reward:.4f} ± {std_reward:.4f}")
