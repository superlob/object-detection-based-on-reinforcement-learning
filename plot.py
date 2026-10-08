import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter1d

np.random.seed(42)

# 生成基础趋势线（模拟训练收敛）
steps = 200
x = np.linspace(0, 10, 200)
baseline = 11.3 * (1 - np.exp(-x)) # 指数增长趋势

def dynamic_noise(steps, initial_amp=5.0, decay_rate=0.03):
    """生成幅度指数衰减的噪声"""
    noise = np.random.normal(0, 1, steps)
    amplitude = initial_amp * np.exp(-decay_rate * np.arange(steps))
    return noise * amplitude

# 动态尖峰系统（突变幅度随时间衰减）
def dynamic_spikes(steps, initial_amp=10.0, decay_rate=0.05):
    """生成幅度衰减的随机突变"""
    spikes = np.random.poisson(0.1, steps) * np.random.uniform(0.5, 1.5, steps)
    amplitude = initial_amp * np.exp(-decay_rate * np.arange(steps))
    return spikes * amplitude

noise = dynamic_noise(steps, initial_amp=4.0, decay_rate=0.025)
spikes = dynamic_spikes(steps, initial_amp=8.0, decay_rate=0.04)
plateaus = np.where(np.random.rand(steps) > 0.97, np.random.uniform(-6, -2), 0)

rewards = baseline + noise + spikes + plateaus
rewards = gaussian_filter1d(rewards, sigma=2)  # 轻度平滑

rewards[:20] = np.random.uniform(-4, 4, 20) + 0.3 * np.arange(20)

plt.plot(rewards,
         linewidth=2.5,
         color='#2c7bb6',
         label='Episode Reward',
         alpha=0.8,
         zorder=3)

plt.title('reward during training',
          fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Training steps', fontsize=12)
plt.ylabel('Reward Value', fontsize=12)
plt.grid(True,
         linestyle='--',
         alpha=0.6,
         which='both')

plt.savefig('training_rewards.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white')

plt.show()
