# ======== Modified by Xiao for multi-drones collision scenarios ======== #

from typing import Optional

import numpy as np

from openrl.envs.mpe.core import Agent, Landmark, World
from openrl.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self):
        self.render_mode = None

    def make_world(
        self,
        render_mode: Optional[str] = None,
        world_length: int = 25,
        num_agents: int = 3,
        num_landmarks: int = 5,
    ):
        self.render_mode = render_mode
        world = World()
        world.name = "simple_spread"
        world.world_length = world_length
        # set any world properties first
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_landmarks = num_landmarks  # 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True
            # =================================== #
            # Set a smaller size for agent circle
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            # =================================== #
            # Add 2 obstacles as collide landmarks
            landmark.name = "landmark %d" % i
            landmark.movable = False
            if i < world.num_landmarks - 2:
                landmark.collide = False
            else:
                landmark.collide = True
            
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, np_random: Optional[np.random.Generator] = None):
        # random properties for agents
        if np_random is None:
            np_random = np.random.default_rng()
        world.assign_agent_colors()

        world.assign_landmark_colors()
        # =================================== #
        # Set color for obstacles
        for l in world.landmarks:
            if l.collide == True:
                l.color = np.array([0.9, 0.1, 0.1])
                l.size = 0.15

        # =================================== #
        # Set the position of agent
        agent_p = np.array([[-0.5, 0.5], [0.3, -0.6], [0.7, -0.1]])
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = agent_p[i]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # =================================== #
        # Use the random generated position
        # for agent in world.agents:
        #     agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     agent.state.p_vel = np.zeros(world.dim_p)
        #     agent.state.c = np.zeros(world.dim_c)

        # =================================== #
        # Set the position of obstacles and landmarks 
        landmark_p = np.array([[-0.2, -0.1], [0, 0], [0.2, 0.2], [-0.2, 0.2], [0.2, -0.2]])
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = landmark_p[i]
            landmark.state.p_vel = np.zeros(world.dim_p)
        # =================================== #
        # Use the random generated position
        # for i, landmark in enumerate(world.landmarks):
        #     if landmark.collide == False:
        #         landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
        #     else:
        #         landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for landmark in world.landmarks:
            # =================================== #
            # Only compute the dist of true landmarks
            if landmark.collide == False:
                dists = [
                    np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos)))
                    for a in world.agents
                ]
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.1:
                    occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
            # =================================== #
            # Count the obstacle collision
            for l in world.landmarks:
                if l.collide == True:
                    if self.is_collision_obstacle(agent, l):
                        rew -= 1
                        collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    # =================================== #
    # Define the obstacle collision function
    def is_collision_obstacle(self, agent, landmark):
        delta_pos = agent.state.p_pos - landmark.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent.size + landmark.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for landmark in world.landmarks:
            # =================================== #
            # Reward of ture landmarks
            if landmark.collide == False:
                dists = [
                    np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos)))
                    for a in world.agents
                ]
                rew -= min(dists)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
            # =================================== #
            # Reward of obstacle collision
            for l in world.landmarks:
                if l.collide == True:
                    if self.is_collision_obstacle(agent, l):
                        rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        )
