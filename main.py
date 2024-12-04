import numpy as np
from copy import deepcopy
import random

class World():
    def __init__(self):
        # Map given in assignment
        self.map = [['', '', '', '', '', ''],
                    ['', 'G', '', 'B', '', ''],
                    ['', '', 'B', 'P', 'B', ''],
                    ['', 'B', '', 'SB', 'B', ''],
                    ['B', 'P', 'SB', 'W', 'S', ''],
                    ['', 'B', '', 'S', '', ''],
                    ['', '', '', '', '', '']]
        # Starting location
        self.location = (6, 0)

        self.policy = {}
        self.rows = len(self.map)
        self.cols = len(self.map[0])
        
        # Init policy to zeros
        self.policy = {}
        for i in range(6):
            for j in range(7):
                self.policy[(i, j)] = None
                #state = (i, j)
                #actions = self.valid_moves(state)
                #action_values = {action: 0.0 for action in actions}
                #self.policy[f'{i}, {j}'] = action_values

    
    def valid_moves(self, state: tuple) -> set:
        moves = []
        x, y = state
        
        if x > 0:
            moves.append('^')  # Up
        if x < self.rows - 1:
            moves.append('v')  # Down
        if y > 0:
            moves.append('<')  # Left
        if y < self.cols - 1:
            moves.append('>')  # Right
        
        return moves
    
    def get_rewards(self):
        rewards = np.full((self.rows, self.cols), -1)  # Default small penalty
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.map[i][j]
                if cell == 'G':
                    rewards[i][j] = 10000
                elif cell == 'P':
                    rewards[i][j] = -100000
                elif cell == 'W':
                    rewards[i][j] = -100000
        return rewards
    
    def get_next_state(self, state, action):
        x, y = state
        if action == '^':
            return max(0, x - 1), y
        elif action == 'v':
            return min(self.rows - 1, x + 1), y
        elif action == '<':
            return x, max(0, y - 1)
        elif action == '>':
            return x, min(self.cols - 1, y + 1)
        return state

    def value_iteration(self, gamma=0.95, time_horizon=100):
        rewards = self.get_rewards()
        V = np.zeros((self.rows, self.cols))  # Initialize values to 0
        for t in range(time_horizon):
            new_V = deepcopy(V)
            for i in range(self.rows):
                for j in range(self.cols):
                    state = (i, j)
                    actions = self.valid_moves(state)
                    action_values = []
                    for action in actions:
                        expected_value = 0
                        # Intended move
                        next_state = self.get_next_state(state, action)
                        expected_value += 0.7 * V[next_state]
                        # Reverse move
                        reverse_action = {'^': 'v', 'v': '^', '<': '>', '>': '<'}[action]
                        reverse_state = self.get_next_state(state, reverse_action)
                        expected_value += 0.15 * V[reverse_state]
                        # Stall
                        expected_value += 0.15 * V[state]
                        action_values.append(expected_value)
                    new_V[i, j] = max(action_values) * gamma + rewards[i][j]
                    self.policy[(i, j)] = actions[np.argmax(action_values)] if action_values else None
            V = new_V
        return V
    
    def policy_evaluation(self, policy, gamma=0.95, time_horizon=50):
        rewards = self.get_rewards()
        V = np.zeros((self.rows, self.cols))  # Initialize value function to zero
        for _ in range(time_horizon):  # Run for a fixed number of iterations
            new_V = deepcopy(V)
            for i in range(self.rows):
                for j in range(self.cols):
                    state = (i, j)
                    action = policy[state]
                    if action is None:
                        continue
                    expected_value = 0
                    # Intended move
                    next_state = self.get_next_state(state, action)
                    expected_value += 0.7 * V[next_state]
                    # Reverse move
                    reverse_action = {'^': 'v', 'v': '^', '<': '>', '>': '<'}[action]
                    reverse_state = self.get_next_state(state, reverse_action)
                    expected_value += 0.15 * V[reverse_state]
                    # Stall
                    expected_value += 0.15 * V[state]
                    new_V[i, j] = expected_value * gamma + rewards[i, j]
            V = new_V
        return V

    def policy_iteration(self, gamma=0.95, time_horizon=50):
        rewards = self.get_rewards()
        policy = {state: np.random.choice(self.valid_moves(state)) if self.valid_moves(state) else None
                  for state in [(i, j) for i in range(self.rows) for j in range(self.cols)]}
        while True:
            # Policy Evaluation
            V = self.policy_evaluation(policy, gamma, time_horizon)
            # Policy Improvement
            policy_optimal = True
            for i in range(self.rows):
                for j in range(self.cols):
                    state = (i, j)
                    actions = self.valid_moves(state)
                    if not actions:
                        continue
                    action_values = []
                    for action in actions:
                        expected_value = 0
                        # Intended move
                        next_state = self.get_next_state(state, action)
                        expected_value += 0.7 *  V[next_state]
                        # Reverse move
                        reverse_action = {'^': 'v', 'v': '^', '<': '>', '>': '<'}[action]
                        reverse_state = self.get_next_state(state, reverse_action)
                        expected_value += 0.15 * V[reverse_state]
                        # Stall
                        expected_value += 0.15 * V[state]
                        action_values.append(expected_value)
                    best_action = actions[np.argmax(action_values)]
                    new_policy = deepcopy(policy)
                    new_policy[state] = best_action
                    new_V = self.policy_evaluation(new_policy, gamma, time_horizon)
                    if new_V[state] > V[state]:
                        policy[state] = best_action
                        policy_optimal = False
                        V = new_V
            if policy_optimal:
                break
        return V, policy
    
    def epsilon_greedy_q_learning(self, alpha=0.5, epsilon=0.5, gamma=0.98, episodes=10000):
        Q = {
            (i, j): {action: 0.0 for action in ['^', 'v', '<', '>']}
            for i in range(self.rows) for j in range(self.cols)
        }
        
        # Initialize rewards
        rewards = self.get_rewards()

        # List to collect cumulative rewards for each run
        #cumulative_rewards = []

        # Run the episodes
        for episode in range(episodes):
            # Reset agent to the starting position
            state = self.location
            episode_reward = 0

            while True:
                # Choose action: epsilon-greedy strategy
                if random.random() < epsilon:
                    action = random.choice(list(Q[state].keys()))  # Explore
                else:
                    action = max(Q[state], key=Q[state].get)  # Exploit
                    
                # Determine actual action outcome based on transition probabilities
                prob = random.random()
                if prob < 0.15:  # Reverse
                    reverse_action = {'^': 'v', 'v': '^', '<': '>', '>': '<'}.get(action, action)
                    if reverse_action not in Q[state]:
                        reverse_action = action
                    next_state = self.get_next_state(state, reverse_action)
                elif prob < 0.30:  # Stall
                    next_state = state
                else:
                    next_state = self.get_next_state(state, action)
                    
                #episode_reward = rewards[next_state]
                
                # Update Q-value
                best_next_q = max(Q[next_state].values()) if Q[next_state] else 0
                Q[state][action] = Q[state][action] + alpha * (
                    rewards[next_state] + gamma * best_next_q - Q[state][action]
                )
                
                # Transition to the next state
                state = next_state
                
                # Terminate if we reach a terminal state (goal or pit)
                if self.map[state[0]][state[1]] in {'G', 'P', 'W'}:
                    break
            
            # Store cumulative reward for the episode
            #cumulative_rewards.append(episode_reward)
        
        # Derive the policy from the Q-table
        policy = {
            state: max(Q[state], key=Q[state].get) if Q[state] else None
            for state in Q
        }
        
        expected_values = {
            state: max(Q[state].values()) if Q[state] else 0
            for state in Q
        }
        
        return Q, policy, expected_values
     

def main():
    
    world = World()
    V = world.value_iteration(gamma=0.95, time_horizon=50)
    print(V)
    for i in range(world.rows):
        print(' '.join(world.policy[(i, j)] or '.' for j in range(world.cols)))
        
        
        
    print("\npolicy iteration\n")
    value, policy = world.policy_iteration(gamma=0.95, time_horizon=100)
    print(value)
        
    for i in range(world.rows):
        print(' '.join(policy[(i, j)] if policy[(i, j)] else '.' for j in range(world.cols)))
        
    results = []
    for run in range(5):
        print(f"Run {run + 1}:")
        Q, policy, expected_value = world.epsilon_greedy_q_learning(alpha=0.5, epsilon=0.5, gamma=0.98, episodes=500)

        # Collect results
        results.append({
            'Q': Q,
            'policy': policy,
            'value': expected_value
        })

        # Output cumulative rewards and policy for the run
        print("Value Grid:")
        #print(expected_value)
        values = np.zeros((7, 6))
        for (i, j), value in expected_value.items():
            values[i, j] = value
        for row in values:
            print(" ".join(f"{value:.2f}" for value in row))
        
        print("\nPolicy Grid:")
        for i in range(world.rows):
            print(' '.join(policy[(i, j)] if policy[(i, j)] else '.' for j in range(world.cols)))
        print("\n")



if __name__ == '__main__':
    main()