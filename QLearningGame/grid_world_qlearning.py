import numpy as np
import pygame
import random
import time

# --- 游戏配置 ---
GRID_SIZE = 5  # 5x5 的网格
CELL_SIZE = 100  # 每个格子的像素大小
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE

# 颜色定义 (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # 陷阱
GREEN = (0, 255, 0)  # 宝藏
BLUE = (0, 0, 255)  # 智能体
GRAY = (128, 128, 128)  # 障碍物/墙

# Pygame 初始化
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Q-Learning Grid World")
clock = pygame.time.Clock()

# --- Q-Learning 参数 ---
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 5000  # 训练回合数
EPSILON = 1.0  # 探索率
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# 动作: 0=上, 1=下, 2=左, 3=右
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# --- 游戏环境 ---
class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        # 设置起点 (0, 0)
        self.start_pos = (0, 0)
        # 设置宝藏位置 (终点)
        self.treasure_pos = (GRID_SIZE - 1, GRID_SIZE - 1)
        # 设置陷阱 (可自定义)
        self.trap_pos = [(1, 1), (3, 2)]
        # 设置障碍物
        self.wall_pos = [(2, 2)]

        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        self.done = False
        return self.get_state()

    def get_state(self):
        # 状态可以用一维索引表示，或者直接用 (x, y) 元组
        # 这里我们返回 (x, y) 元组，Q表使用字典或二维数组
        return self.agent_pos

    def step(self, action):
        if self.done:
            return self.get_state(), 0, self.done

        # 获取动作向量
        move = ACTIONS[action]
        new_x = self.agent_pos[0] + move[0]
        new_y = self.agent_pos[1] + move[1]

        # 边界检查
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            # 检查是否是障碍物
            if (new_x, new_y) not in self.wall_pos:
                self.agent_pos = (new_x, new_y)

        # 计算奖励
        reward = -1  # 每走一步扣分

        # 检查是否到达宝藏
        if self.agent_pos == self.treasure_pos:
            reward += 10
            self.done = True

        # 检查是否掉入陷阱
        if self.agent_pos in self.trap_pos:
            reward -= 10
            self.done = True

        return self.get_state(), reward, self.done

    def draw(self):
        screen.fill(WHITE)
        # 绘制网格线
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(screen, BLACK, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
            pygame.draw.line(screen, BLACK, (0, y), (WINDOW_WIDTH, y))

        # 绘制元素
        def draw_cell(pos, color):
            x, y = pos
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

        # 绘制陷阱
        for pos in self.trap_pos:
            draw_cell(pos, RED)
        # 绘制宝藏
        draw_cell(self.treasure_pos, GREEN)
        # 绘制障碍物
        for pos in self.wall_pos:
            draw_cell(pos, GRAY)
        # 绘制智能体
        draw_cell(self.agent_pos, BLUE)

        pygame.display.flip()


# --- Q-Learning 算法 ---
# --- Q-Learning 算法 (修正版)---
def train():
    env = GridWorld()
    # 初始化 Q 表
    q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))

    # 创建 EPSILON 的局部变量，避免 UnboundLocalError
    epsilon = EPSILON

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        while not env.done:
            # Epsilon-greedy 策略
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)  # 随机探索
            else:
                # 利用：选择 Q 值最大的动作
                x, y = state
                action = np.argmax(q_table[x, y])

            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q-Learning 更新公式
            old_q = q_table[state[0], state[1], action]
            next_max_q = np.max(q_table[next_state[0], next_state[1]]) if not done else 0
            new_q = old_q + LEARNING_RATE * (reward + DISCOUNT * next_max_q - old_q)
            q_table[state[0], state[1], action] = new_q

            state = next_state

        # 衰减 epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY  # 修改局部变量 epsilon

        if episode % 500 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    print("训练完成！")
    return q_table


# --- 测试/演示函数 ---
def test(q_table):
    env = GridWorld()
    state = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 使用训练好的 Q 表进行决策 (贪婪策略)
        x, y = state
        action = np.argmax(q_table[x, y])
        state, reward, done = env.step(action)
        env.draw()
        clock.tick(5)  # 控制演示速度

        if done:
            print("到达目标或陷阱！按关闭窗口退出。")
            time.sleep(2)
            state = env.reset()  # 自动重置重新演示

    pygame.quit()


# --- 主程序 ---
if __name__ == "__main__":
    print("开始训练 AI...")
    trained_q_table = train()
    print("启动演示界面...")
    test(trained_q_table)
    print(trained_q_table)