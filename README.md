# MDP
CS463G Markov Decision Processes

You will find one grid-worlds (complete with Wumpus) under Files.  You will run each of three algorithms on the grid-world. What makes this interesting is that your avatar is exploring the grid world in a manual transmission car (which the Wumpus might eat, along with your avatar -- yumm, crunchy shell!)  Your avatar isn't great at manual transmission.  With probability 0.15, the car will end up in reverse, and go backwards from the intended direction.  With probability 0.15, it will stall out and go nowhere.  With probability 0.7, it goes where you send it.  If it hits an edge of the grid world, it stays in its square.

Algorithms:

 - Value Iteration
 - Policy Iteration
 - Epsilon Greedy Q learning

Parameters:
 - For all algorithms, the discount rate gamma will be 0.95.
 - For VI and PI, run the algorithms with time horizons 50 and 100.
 - For epsilon greedy, set alpha to 0.5 and epsilon to 0.5 and gamma to .98.  Run this 5 times and collect data.

To submit:
 - Your code for each algorithm.
 - Images of the policies found for each algorithm, using ^, v, >, < as arrows to indicate the direction/action for each square.  Note that there will be a grid for each of VI and PI and each horizon and each grid world, and 5 for each grid world for the reinforcement learning algorithm (epsilon greedy).
 - Grids showing the expected values of each square.  Again, there should be 9 grids.  You might want to align the policy and value grids for each instance.  (So you end up with value grid followed by policy grid followed by....)
 - What you learned.
 - In the comments, with whom you worked.  Put online sources in the text -- including LLM citations with prompts.
