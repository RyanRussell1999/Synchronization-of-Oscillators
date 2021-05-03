# Synchronization-of-Oscillators
This is the code for the senior project completed by Jakob Harig and Ryan Russell at Bradley University on the Synchronization of Oscillators Using Reinforcement Learning.

## Code
This folder contains all of the code written while completing this project. Much of this code is preliminary or meant for the cartpole system. This code is also not as well organized but is here for the used of documentation of all the code written throughout the course of this project. This code may prove useful for future work. 

## NVIDIA Jetson Tests
This folder contains code to test the variety of parts needed to physically implement the follower oscillator systems. In this there is a code to drive DC motors, code to measure distance in meters using the ultrasonic sensor, code to communicate between NVIDIA Jetsons using XBees, and code to synchronize the speed of motors driven by different NVIDIA Jetsons using XBees. (Files within specific folders must be run on specific Jetsons) All of these verify the use of these components individually with the NVIDIA Jetson Nano and through the use of Python scripts. These scripts are later used as the foundation to interface with these components within our complete physical implementation for the synchronization of oscillators using reinforcement learning. 

## Physical Implementation of RL Algorithm
This folder contains the code used to physically implement the reinforcement learning algorithm to synchronize oscillators on a virtual leader. Within the Blue oscillator, there is one file for a single follower oscillator (Leader) and another for two follower oscillators (Pos). The Orange oscillator has one file for two follower oscillators. These files have a virtual leader following a square wave for velocity. The single follower yields promising results, however there are issues with the two followers. This code is only to be run on the NVIDIA Jetson Nano, following the same GPIO configuration as in the project.

![Alt text](image/Single_Oscillator_Results.jpg?raw=true "Single Oscillator Synchronization Results")

## Simulation of RL Algorithm
This folder contains simulations of the reinforcement learning algorithm to synchronize oscillators to a leader following a sinusoidal trajectory. There are simulations for a linear controller, the RL controller, and the RL controller with limited velocity. All of these have two followers and one leader. All simulations yield synchronization in a short period  of time. The results from these verify the algorithm and laid groundwork to implement the algorithm physically.

![Alt text](image/LinearControllerResults_Part.jpg?raw=true "Multiagent Linear Controller Results")

![Alt text](image/Simulation_Results_PartTime_noLim.jpg?raw=true "RL Controller Results")

![Alt text](image/Simulation_Results_PartTime.jpg?raw=true "RL Controller Results (Velocity Limited)")
