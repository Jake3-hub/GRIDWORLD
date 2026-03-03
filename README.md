# GRIDWORLD
The Grid World game is a classic reinforcement learning environment testing the decision-making capabilities of an artificial intelligence agent. The Q-Learning algorithm is employed to train the agent.
## 🎮 Game Environment

### The Objective
The agent (represented as a blue square) starts at the top-left corner. The objective is to reach the treasure (green square) at the bottom-right corner while avoiding hazards.

### Rules & Rewards
-   **Treasure (Green):** +10 Reward (Game Ends in Success).
-   **Trap (Red):** -10 Reward (Game Ends in Failure).
-   **Wall (Gray):** -1 Reward (Invalid move, agent stays in place).
-   **Normal Step:** -1 Reward (Penalty for time, encourages efficiency).

### State and Action Space
-   **State Space:** The agent's position $(x, y)$ on the 5x5 grid (25 discrete states).
-   **Action Space:** 4 possible movements:
    1.  Up
    2.  Down
    3.  Left
    4.  Right

---

## ⚙️ Algorithm Implementation

### Q-Learning Theory
The core of this project is the Q-Learning update rule. The agent maintains a **Q-table** $Q(s, a)$ which estimates the quality of taking action $a$ in state $s$.

The update rule is defined as:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

### Exploration vs. Exploitation
To balance learning new paths and using known good paths, the algorithm uses **Epsilon-Greedy Strategy**:
-   With probability $\epsilon$, choose a random action (Exploration).
-   With probability $1-\epsilon$, choose the action with the highest Q-value (Exploitation).
-   $\epsilon$ decays over time ($EPSILON_DECAY$) to shift from exploration to exploitation.

---

## 🛠️ Setup & Installation

To run this project locally, please follow these steps:

### 1. Prerequisites
Ensure you have Python 3 installed.

### 2. Clone the Repository
bash
git clone https://github.com/your-username/GRIDWORLD.git
cd GRIDWORLD

### 3. Install Dependencies
This project requires numpy and pygame.
pip install numpy pygame

### 4. Run the Script
python grid_world_qlearning.py
