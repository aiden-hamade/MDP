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
        self.location = (0, 0)

        # Init policy to zeros
        self.policy = {}
        for i in range(6):
            for j in range(7):
                state = (i, j)
                actions = self.valid_moves(state)
                action_values = {action: 0.0 for action in actions}
                self.policy[f'{i}, {j}'] = action_values

    
    def valid_moves(self, state: tuple) -> set:
        moves = {'up', 'down', 'left', 'right'}
        x, y = state

        if y == 0:
            moves.remove('down')
        if x == 5:
            moves.remove('right')
        if x == 0:
            moves.remove('left')
        if y == 6:
            moves.remove('up')
        
        return moves


def main():
    world = World()
    with open('test.json', 'w') as file:
        json.dump(world.policy, file)


if __name__ == '__main__':
    main()