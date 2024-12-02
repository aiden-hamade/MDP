import numpy as np
import json

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
        
        #if y == 0:
        #    moves.remove('down')
        #if x == 5:
        #    moves.remove('right')
        #if x == 0:
        #    moves.remove('left')
        #if y == 6:
        #    moves.remove('up')
        
        return moves
    
    def get_rewards(self):
        rewards = np.full((self.rows, self.cols), -0.1)  # Default small penalty
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.map[i][j]
                if cell == 'G':
                    rewards[i][j] = 1000
                elif cell == 'P':
                    rewards[i][j] = -100000
                elif cell == 'W':
                    rewards[i][j] = -100000
                elif cell == 'B':
                    rewards[i][j] = -1
                elif cell in {'S', 'SB'}:
                    rewards[i][j] = -1
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

    def value_iteration(self, gamma=0.95, time_horizon=50):
        rewards = self.get_rewards()
        V = np.zeros((self.rows, self.cols))  # Initialize values to 0
        for t in range(time_horizon):
            new_V = np.copy(V)
            for i in range(self.rows):
                for j in range(self.cols):
                    state = (i, j)
                    actions = self.valid_moves(state)
                    action_values = []
                    for action in actions:
                        value = 0
                        # Intended move
                        next_state = self.get_next_state(state, action)
                        value += 0.7 * (rewards[next_state] + gamma * V[next_state])
                        # Reverse move
                        reverse_action = {'^': 'v', 'v': '^', '<': '>', '>': '<'}[action]
                        reverse_state = self.get_next_state(state, reverse_action)
                        value += 0.15 * (rewards[reverse_state] + gamma * V[reverse_state])
                        # Stall
                        value += 0.15 * (rewards[state] + gamma * V[state])
                        action_values.append(value)
                    new_V[i, j] = max(action_values) if action_values else V[i, j]
                    self.policy[(i, j)] = actions[np.argmax(action_values)] if action_values else None
            if np.max(np.abs(new_V - V)) < 1e-4:
                break
            V = new_V
        return V
    
    def policy_evaluation(self, policy, gamma=0.95, time_horizon=50):
        """
        Evaluates a policy for a given time horizon.
        """
        rewards = self.get_rewards()
        V = np.zeros((self.rows, self.cols))  # Initialize value function to zero
        for _ in range(time_horizon):  # Run for a fixed number of iterations
            new_V = np.copy(V)
            for i in range(self.rows):
                for j in range(self.cols):
                    state = (i, j)
                    action = policy[state]
                    if action is None:
                        continue
                    value = 0
                    # Intended move
                    next_state = self.get_next_state(state, action)
                    value += 0.7 * (rewards[next_state] + gamma * V[next_state])
                    # Reverse move
                    reverse_action = {'^': 'v', 'v': '^', '<': '>', '>': '<'}[action]
                    reverse_state = self.get_next_state(state, reverse_action)
                    value += 0.15 * (rewards[reverse_state] + gamma * V[reverse_state])
                    # Stall
                    value += 0.15 * (rewards[state] + gamma * V[state])
                    new_V[i, j] = value
            V = new_V
        return V

    def policy_iteration(self, gamma=0.95, time_horizon=50):
        """
        Runs Policy Iteration with a fixed time horizon for evaluation.
        """
        rewards = self.get_rewards()
        policy = {state: np.random.choice(self.valid_moves(state)) if self.valid_moves(state) else None
                  for state in [(i, j) for i in range(self.rows) for j in range(self.cols)]}
        while True:
            # Policy Evaluation
            V = self.policy_evaluation(policy, gamma, time_horizon)
            # Policy Improvement
            policy_stable = True
            for i in range(self.rows):
                for j in range(self.cols):
                    state = (i, j)
                    actions = self.valid_moves(state)
                    if not actions:
                        continue
                    action_values = []
                    for action in actions:
                        value = 0
                        # Intended move
                        next_state = self.get_next_state(state, action)
                        value += 0.7 * (rewards[next_state] + gamma * V[next_state])
                        # Reverse move
                        reverse_action = {'^': 'v', 'v': '^', '<': '>', '>': '<'}[action]
                        reverse_state = self.get_next_state(state, reverse_action)
                        value += 0.15 * (rewards[reverse_state] + gamma * V[reverse_state])
                        # Stall
                        value += 0.15 * (rewards[state] + gamma * V[state])
                        action_values.append(value)
                    best_action = actions[np.argmax(action_values)]
                    if best_action != policy[state]:
                        policy_stable = False
                    policy[state] = best_action
            if policy_stable:
                break
        return V, policy
    
    def epsilon_greedy_q_learning(self, alpha=0.5, epsilon=0.5, gamma=0.98, episodes=500):
        """
        Runs epsilon-greedy Q-learning on the Wumpus World.
        """
        # Initialize Q-table
        Q = {
            (i, j): {action: 0.0 for action in self.valid_moves((i, j))}
            for i in range(self.rows) for j in range(self.cols)
        }
        
        # Initialize rewards
        rewards = self.get_rewards()

        # List to collect cumulative rewards for each run
        cumulative_rewards = []

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
                
                # Determine next state and reward
                next_state = self.get_next_state(state, action)
                reward = rewards[next_state]
                episode_reward += reward

                # Determine actual action outcome based on transition probabilities
                prob = random.random()
                if prob < 0.15:  # Reverse
                    action = {'^': 'v', 'v': '^', '<': '>', '>': '<'}[action]
                    next_state = self.get_next_state(state, action)
                elif prob < 0.30:  # Stall
                    next_state = state
                
                # Update Q-value
                best_next_q = max(Q[next_state].values()) if Q[next_state] else 0
                Q[state][action] = Q[state][action] + alpha * (
                    reward + gamma * best_next_q - Q[state][action]
                )
                
                # Transition to the next state
                state = next_state
                
                # Terminate if we reach a terminal state (goal or pit)
                if self.map[state[0]][state[1]] in {'G', 'P', 'W'}:
                    break
            
            # Store cumulative reward for the episode
            cumulative_rewards.append(episode_reward)
        
        # Derive the policy from the Q-table
        policy = {
            state: max(Q[state], key=Q[state].get) if Q[state] else None
            for state in Q
        }
        
        return Q, policy, cumulative_rewards
     

def main():
    world = World()
    V = world.value_iteration(gamma=0.95, time_horizon=50)
    print(V)
    #with open('test.json', 'w') as file:
        #json.dump(world.policy, file)
    for i in range(world.rows):
        print(' '.join(world.policy[(i, j)] or '.' for j in range(world.cols)))
        
        
        
    print("\npolicy iteration\n")
    value, policy = world.policy_iteration(gamma=0.95, time_horizon=50)
    print(value)
        
    for i in range(world.rows):
        print(' '.join(policy[(i, j)] if policy[(i, j)] else '.' for j in range(world.cols)))
        
    results = []
    for run in range(5):
        print(f"Run {run + 1}:")
        Q, policy, rewards = world.epsilon_greedy_q_learning(alpha=0.5, epsilon=0.5, gamma=0.98, episodes=500)

        # Collect results
        results.append({
            'Q': Q,
            'policy': policy,
            'rewards': rewards
        })

        # Output cumulative rewards and policy for the run
        print("Cumulative Rewards:")
        print(rewards[-10:])  # Last 10 rewards as a sample

        print("\nPolicy Grid:")
        for i in range(world.rows):
            print(' '.join(policy[(i, j)] if policy[(i, j)] else '.' for j in range(world.cols)))
        print("\n")



if __name__ == '__main__':
    main()