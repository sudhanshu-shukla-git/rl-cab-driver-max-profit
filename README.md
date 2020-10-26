# Reinforcement Learning-based system for assisting cab drivers to maximise the profit for Cab Transport Company

Create the environment: You are given the ‘Env.py’ file with the basic code structure. This is the "environment class" - each method (function) of the class has a specific purpose. Please read the comments around each method carefully to understand what it is designed to do. Using this framework is not compulsory, you can create your own framework and functions as well.

Build an agent that learns to pick the best request using DQN. You can choose the hyperparameters (epsilon (decay rate), learning-rate, discount factor etc.) of your choice.

Training depends purely on the epsilon-function you choose. If the ϵ decays fast, it won’t let your model explore much and the Q-values will converge early but to suboptimal values. If ϵ decays slowly, your model will converge slowly. We recommend that you try converging the Q-values in 4-6 hrs.  We’ve created a sample ϵ-decay function at the end of the Agent file (Jupyter notebook) that will converge your Q-values in ~5 hrs.

Convergence- You need to converge your results. The Q-values may be suboptimal since the agent won't be able to explore much in 5-6 hours of simulation. But it is important that your Q-values converge. 
