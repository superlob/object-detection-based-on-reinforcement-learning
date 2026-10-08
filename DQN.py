import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from data import ReplayBuffer
from env import env
from log import log

LR = 1e-4
num_episodes = 2
num_step = 20  #一张图片训练的步数 #表示第几张图片
memory_capacity = 10

epsilon_decay = 0.99
min_epsilon = 0.01
target_update = 20  #目标网络参数更新
gamma = 0.99

#定义三个卷积，用于state value function的估计
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_actions),  # 输出每个动作的Q值
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平为一维向量
        x = self.classifier(x)
        return x

class DQNAgent:
    def __init__(self, ):
        self.num_actions = 10
        self.epsilon = 0.99
        self.policy_net = DQN(10)
        self.target_net = DQN(10)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        self.memory = ReplayBuffer(memory_capacity)

    #探索与利用
    def select_action(self, state):
        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)
        else:
            return self.policy_net(state).max(1)[1].view(1, 1)

    def optimize_model(self):
        if len(self.memory) < memory_capacity:
            return
        transitions = self.memory.sample(memory_capacity)
        batch = list(zip(*transitions))
        state_batch = torch.stack(batch[0], dim=0)
        action_batch = torch.tensor(batch[1]).unsqueeze(1)
        reward_batch = torch.tensor(batch[2])
        next_state_batch = torch.torch.stack(batch[3], dim=0)
        done_batch = torch.tensor(batch[4])
        # 计算Q值
        q_values = self.policy_net(state_batch)
        q_values =torch.gather(q_values,1, action_batch)#提取对应动作的q值
        next_q_values = self.target_net(next_state_batch)
        next_q_values = next_q_values.max(1)[0].detach()#下一状态的最大q值
        expected_q_values = reward_batch + (1 - done_batch) * gamma * next_q_values#计算TD-target
        # 计算损失
        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        agent.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train(env, agent):
    for episode in range(num_episodes):#进入一个周期
        #初始化
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done:
            state_temp = state.unsqueeze(0)
            step += 1
            if step >= num_step:
                done = True
            #收集经验并更新模型
            action = agent.select_action(state_temp)
            next_state, reward, done = env.step(action,done)
            env.draw_block(reward, episode*num_step+step)
            agent.memory.push(state,action,reward,next_state,int(done))
            state = next_state
            total_reward += reward
            agent.optimize_model()
        if episode % target_update == 0:#更新目标网络参数
            agent.update_target_net()
        env.get_new_pic(episode+1)
        print(f'Episode {episode + 1}/{episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}')






agent = DQNAgent()
train(env,agent)