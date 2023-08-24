# Machine-Learning-Algorithms-For-Autonomous-Driving
Background:
	 The primary aim of this project is to attain a thorough comprehension of the operational principles employed in autonomous navigation, leveraging machine learning techniques and other software applications, including RViz and Gazebo, on a mobile robot integrated with the Robot Operating System (ROS). Autonomous navigation can be simply defined as the process by which any mobile unit is able to accurately maneuver from its initial position to its desired destination, within an environment. For humans, we have our senses, such as our eyes, to see the environment we are in, process a route and navigate through it using instructions our organic processing unit, the brain, has formed. For vehicles, it is more demanding. The first challenge for autonomous navigation is for the mobile unit to understand the geographical makeup of its environment, as well as its location within it. This vital information is needed to provide detail of where the mobile unit is currently located, as well as where it would like to navigate to. The mobile unit must determine an optimal solution to navigate through the environment while avoiding static and dynamic obstacles. The second challenge is that the mobile unit must also be able to abide by traffic laws, and therefore must not only be able to detect and recognize regulatory traffic signs and lights, but act upon them as well. For example, while in navigation, the vehicle must be able to recognize a stop sign and come to a complete stop before moving when it is safe to. This process can be summed up as a 4-step plan: 1) sense, to collect real time data, 2) perceive, interpret the data, 3) plan, create the route, and 4) act, to allow the vehicle to navigate safely through the route. 
The implementation of autonomous navigation through the design of the project will impact several groups of people. After completing the project and gaining familiarity from the experience made, the work done will be scaled up and implemented onto a Fiat 500 that is provided by the Mechatronics lab and approved by Dr. Alirezaee. This will require coordination with the sensors team, as our design will need different sensors that will be installed onto specific locations of the vehicle. While the primary focus of this project is not consumer-centric, the design can be modeled by ADAS (advanced drive assisted systems) teams, aiming to address road safety in assisted driving, and therefore enhancing public safety.

Objective: 
This project’s main goal is to investigate, understand, and apply the various working principles and operations of autonomous driving capabilities onto a mobile robot embedded with ROS. This can be done by implementing the 4 main processing steps of autonomous driving discussed previously, which are sensing, perception, planning, and acting. The mobile unit must be composed of state-of-the-art sensing capabilities, and therefore will be equipped with several key sensors. As these sensors collect data from the physical world, the specified data must be interpreted. This would aid the system in understanding its environment, localization, and the ability to recognize or track objects. The third step of the design will be planning a route for the mobile unit to follow. This will be composed of a known starting position within the environments, as well as a specified destination for it to navigate to. Finally, the last step of the design will be a summation of the previous steps to display the mobile unit’s ability to physically move from one location to another, while detecting and avoiding obstacles. 

Scope:
	To understand the scope of this project, we must understand its goal, which is to investigate, understand, and apply the various working principles and operations of autonomous driving capabilities. To achieve our design goals, the team worked together and in parallel to investigate different aspects of the project. The first step included the extraction of data from the sensors, which consist of a single camera and a LiDAR system. Using LiDAR, the team was able to create a virtual environment and identify both static and dynamic obstacles. Within this environment, the mobile unit was also localized, a destination was selected, the mobile unit calculated the shortest path to get there, and avoided obstacles detected by the LiDAR system. Additionally, machine learning algorithms, specifically computer vision techniques, were applied for object detection and recognition, and lane detection. Input data was acquired from the camera lens, while the OpenCV based codes were implemented. The design of the OpenCV algorithms were designed in house, while the SLAM, mapping and localization capabilities of the robot were developed using RViz and Gazebo.  