from multiprocessing import Queue
from random import choice
from board import Entity, neighbors, toroidal_distance_2
import numpy as np
import numpy.typing as npt
import random
import collections
import math

AntMove = tuple[tuple[int, int], tuple[int, int]]


def valid_neighbors(
    row: int, col: int, walls: npt.NDArray[np.int_]
) -> list[tuple[int, int]]:
    return [n for n in neighbors((row, col), walls.shape) if not walls[n]]


class myBot: # classes of ants: foragers, attackers 

    def __init__(
        self,
        walls: npt.NDArray[np.int_],
        harvest_radius: int,
        vision_radius: int,
        battle_radius: int,
        max_turns: int,
        time_per_turn: float,
    ) -> None:
        self.walls = walls
        self.collect_radius = harvest_radius
        self.vision_radius = vision_radius
        self.battle_radius = battle_radius
        self.max_turns = max_turns
        self.time_per_turn = time_per_turn

        self.explored_map = set()
        self.gather_map = npt.NDArray[np.int_]
        self.attack_map = npt.NDArray[np.int_]
        self.defend_map = npt.NDArray[np.int_]

        self.ant_roles = {}

    @property
    def name(self):
        return "burt"
    
    def get_frontier_array(self): # get the "frontier" of vision
        rows, cols = self.walls.shape
        frontier = np.zeros((rows, cols))
        
        # We only care about the boundary of what we know
        for r, c in self.explored_map:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = (r + dr) % rows, (c + dc) % cols
                
                if (nr, nc) not in self.explored_map and not self.walls[nr, nc]:
                    frontier[nr, nc] = 1
                    
        return frontier

    def move_ants(self, vision, stored_food) -> set:
        out_moves = set()

        # print("-----------")

        # goals = np.array(
        #     [
        #          [0, 0, 0, 1],
        #          [0, 0, 0, 1],
        #          [0, 0, 0, 0],
        #          [0, 0, 0, 0]
        #          ])
        
        # walls = np.array(
        #     [
        #          [0, 0, 1, 0],
        #          [0, 0, 1, 0],
        #          [0, 1, 0, 0],
        #          [0, 1, 0, 0]
        #          ])
        
        # init_ = np.full((4, 4), np.inf)

        # print(self.generate_dijkstra_map(init_, goals, walls))
        # print(self.avoidance_map(init_, goals, walls))
        # assert(False)
        
        current_tiles = {coord for coord, kind in vision}
        self.explored_map.update(current_tiles)
        
        vis_data = {
            "ants": [c for c, k in vision if k == Entity.FRIENDLY_ANT],
            "food": [c for c, k in vision if k == Entity.FOOD],
            "my_hills": [c for c, k in vision if k == Entity.FRIENDLY_HILL],
            "enemy_hills": [c for c, k in vision if k == Entity.ENEMY_HILL],
            "enemy_ants": [c for c, k in vision if k == Entity.ENEMY_ANT]
        }

        # print(len(vis_data["ants"]))

        self.assign_permanent_roles(vis_data["ants"], vis_data["my_hills"])
        rows, cols = self.walls.shape

        attack_goals = np.zeros(self.walls.shape)

        food_locations = np.zeros(self.walls.shape)
        for f in vis_data["food"]:
            food_locations[f] = 2
            attack_goals[f] = 2
        
        for f in vis_data["enemy_hills"]:
            food_locations[f] = 0.4 # ends up being -2 when multiplied 

        ant_locations = np.zeros(self.walls.shape)
        for f in vis_data["ants"]:
            ant_locations[f] = 4
        
        for f in vis_data["enemy_ants"]:
            ant_locations[f] = 10
        
        frontier_locations = self.get_frontier_array()

        for hill in vis_data["enemy_hills"]:
            attack_goals[hill] = 20
        
        significant_advantage = (len(vis_data["enemy_hills"]) == 1) and (len(vis_data["ants"]) > len(vis_data["enemy_ants"]) * 2)

        goals = frontier_locations*(2)+food_locations*(5)

        if significant_advantage: # significant advantage, just start full sending including gatherers
            for hill in vis_data["enemy_hills"]:
                attack_goals[hill] = 1000
                goals[hill] = 1000        

        attack_goals = attack_goals*(-1)
        goals = goals*(-1)
        
        attackers = [a for a in vis_data["ants"] if self.ant_roles.get(a) == "attacker"]

        # if len(attackers) > 0:
        #     avg_r = int(np.mean([p[0] for p in attackers])) % rows
        #     avg_c = int(np.mean([p[1] for p in attackers])) % cols
            
        #     # Place a pull at the center of the squad
        #     # -15 is strong enough to group them, but won't stop them from attacking
        #     attack_goals[avg_r, avg_c] = -15

        init_map = np.full((rows, cols), np.inf)
        ant_avoidance = self.avoidance_map(init_map, ant_locations, self.walls)

        init_map = np.full((rows, cols), np.inf)
        # if (significant_advantage):
        #     self.gather_map = self.generate_dijkstra_map(init_map, goals, self.walls)
        # else:
        #     self.gather_map = self.generate_dijkstra_map(ant_avoidance, goals, self.walls) # works really bad for some reason?

        self.gather_map = self.generate_dijkstra_map(init_map, goals, self.walls)

        defend_goals = np.zeros((rows, cols))

        for hill in vis_data["my_hills"]:
            for j in valid_neighbors(hill[0], hill[1], self.walls):
                defend_goals[j[0], j[1]] = -10

        for enemy in vis_data["enemy_ants"]:
            defend_goals[enemy] = -5 
                    

        init_map = np.full((rows, cols), np.inf)
        self.attack_map = self.generate_dijkstra_map(init_map, attack_goals, self.walls)

        init_map = np.full((rows, cols), np.inf)
        self.defend_map = self.generate_dijkstra_map(init_map, defend_goals, self.walls)

        threat_map = self.get_threat_map(vis_data)
        if (not significant_advantage):
            self.gather_map += ant_avoidance
            self.attack_map += threat_map

        reserved = set(vis_data["my_hills"]) # avoid stepping on my own hills
        next_turn_roles = {} 

        ants_to_move = vis_data["ants"]
        random.shuffle(ants_to_move)

        cnt = 0

        # if (len(ants_to_move) < len(vis_data["ants"])):
            # print(ants_to_move, vis_data["ants"])
            # print("BAD")

        for ant in ants_to_move:
            role = self.ant_roles.get(ant)
            step = None

            if role == "gatherer":
                step = self.behave_gatherer(ant, vis_data, reserved)
            if role == "attacker":
                step = self.behave_attacker(ant, vis_data, reserved)
            if role == "defender":
                step = self.behave_defender(ant, vis_data, reserved)

            if step and step not in reserved:
                out_moves.add((ant, step))
                reserved.add(step)
                next_turn_roles[step] = role
            elif ant not in reserved:
                reserved.add(ant)
                next_turn_roles[ant] = role
            else:
                # cnt += 1
                # in case somehow original position is claimed already, find any valid neighbor to stay alive
                escaped = False
                for nxt in valid_neighbors(*ant, self.walls):
                    if nxt not in reserved:
                        out_moves.add((ant, nxt))
                        reserved.add(nxt)
                        next_turn_roles[nxt] = role
                        escaped = True
                        break
                if (escaped == False):
                    cnt += 1
                    # print("uhoh")
        
        # print(f"dead: {cnt}")

        self.ant_roles = next_turn_roles
        return out_moves
    
    def avoidance_map(self, init_map, goals, walls):
        # print("hi")
        a_map = self.generate_dijkstra_map(init_map, goals, walls)
        # print("hi")
        a_map *= -1.2
       # print("avoiding")
        a_map[np.isneginf(a_map)]=np.inf
        return self.generate_dijkstra_map(a_map, goals, walls)

    
    def generate_dijkstra_map(self, init_map, goals, walls):
        rows, cols = walls.shape

        d_map = init_map
        
        queue = collections.deque()

        goal_coords = np.argwhere(goals != 0) # find all places with goals = 1
        for r, c in goal_coords:
            if 0 <= r < rows and 0 <= c < cols:
                val = goals[r][c]
                d_map[r][c] = val
                queue.append((r, c, val))

        while queue:
            r, c, dist = queue.popleft()

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # neighbors
                nr, nc = r + dr, c + dc # new coords

                nr %= rows # wraparound
                nc %= cols

                if not walls[nr][nc] and d_map[nr][nc] == float('inf'): # this should make everything thats not a wall non-inf by the end
                    d_map[nr][nc] = min(dist + 1, d_map[nr][nc])
                    queue.append((nr, nc, dist + 1))
        return d_map

    # ant class behavior
    def behave_gatherer(self, ant, vis_data, reserved):
        # print("behaving")
        d_map = self.gather_map 
        r, c = ant 
        
        best_step = None 
        min_score = d_map[r, c] 

        neighbors_list = valid_neighbors(r, c, self.walls)
        random.shuffle(neighbors_list) # create some variance here
        scores = []

        # print(neighbors_list)

        for n in neighbors_list:
            score = d_map[n[0], n[1]]
            if (score != score + 1): # not infinity
                scores.append(score)

            # print(score)

            # if n in reserved:
            #     continue
            
            # if score < min_score:
            #     min_score = score
            #     best_step = n

        np_scores = np.array(scores)
        np_scores -= min(np_scores)
        exp_scores = np.exp(-4*np_scores)
        #print(exp_scores)
        probs = exp_scores / np.sum(exp_scores)
        
        #print(probs)

        # pick one neighbor based on the probability distribution e^-2x
        idx = np.random.choice(len(neighbors_list), p=probs)

        #print(idx)

        return neighbors_list[idx]

    def behave_defender(self, ant, vis_data, reserved):
        d_map = self.defend_map
        r, c = ant
        
        best_step = None
        min_score = d_map[r, c]

        scores = []
        
        neighbors_list = valid_neighbors(r, c, self.walls)
        random.shuffle(neighbors_list)

        for n in neighbors_list:
            score = d_map[n[0], n[1]]
            if (score != score + 1): # not infinity
                scores.append(score)

            # print(score)

            # if n in reserved:
            #     continue
            
            # if score < min_score:
            #     min_score = score
            #     best_step = n

        np_scores = np.array(scores)
        np_scores -= min(np_scores)
        exp_scores = np.exp(-2*np_scores)
        #print(exp_scores)
        
        probs = exp_scores / np.sum(exp_scores)
        
        #print(probs)

        idx = np.random.choice(len(neighbors_list), p=probs)

        #print(idx)

        return neighbors_list[idx]

    def behave_attacker(self, ant, vis_data, reserved):
        d_map = self.attack_map
        r, c = ant
        
        best_step = None
        min_score = d_map[r, c]
        
        neighbors_list = valid_neighbors(r, c, self.walls)
        random.shuffle(neighbors_list)

        scores = []

        for n in neighbors_list:
            score = d_map[n[0], n[1]]
            if (score != score + 1): # not infinity
                scores.append(score)

            # print(score)

            # if n in reserved:
            #     continue
            
            # if score < min_score:
            #     min_score = score
            #     best_step = n

        np_scores = np.array(scores)
        np_scores -= min(np_scores)
        exp_scores = np.exp(-2*np_scores)
        #print(exp_scores)
        probs = exp_scores / np.sum(exp_scores) # do a probability distribution
        
        #print(probs)

        idx = np.random.choice(len(neighbors_list), p=probs) 

        #print(idx)

        return neighbors_list[idx]

    def assign_permanent_roles(self, my_ants, my_hills):
        current_ant_set = set(my_ants)
        num_ants = len(my_ants)
        self.ant_roles = {pos: role for pos, role in self.ant_roles.items() if pos in current_ant_set}

        for ant in my_ants:
            if ant not in self.ant_roles:

                num_defenders = list(self.ant_roles.values()).count("defender")
                num_attackers = list(self.ant_roles.values()).count("attacker")
                num_gatherers = list(self.ant_roles.values()).count("gatherer")

                # print(num_defenders, num_attackers, num_gatherers)

                # self.ant_roles[ant] = "gatherer"
                rando = random.random()

                percentage_defend = 0.05
                percentage_gather = 0.1
                if (num_ants > 2):
                    percentage_gather = max(1-1/np.log(num_ants/2), 0)
                percentage_attack = 1-percentage_gather


                num_ants_not_defending = num_ants-num_defenders


                if num_defenders < len(my_hills) * percentage_defend * (num_ants-10): 
                    self.ant_roles[ant] = "defender"
                elif (num_attackers < (percentage_attack-0.1)*(num_ants_not_defending-20)): # deficit of attackers
                    self.ant_roles[ant] = "attacker"
                elif (num_gatherers < (percentage_gather-0.1)*num_ants_not_defending): # deficit of gatherers
                    self.ant_roles[ant] = "gatherer"
                else: # if reasonably close to distrib
                    if rando < percentage_attack:
                        self.ant_roles[ant] = "attacker"
                    else:
                        self.ant_roles[ant] = "gatherer"

    def get_threat_map(self, vis_data):
        rows, cols = self.walls.shape
        threat_map = np.zeros((rows, cols))
        
        enemies = vis_data["enemy_ants"]
        friends = vis_data["ants"]
        
        for enemy in enemies:
            er, ec = enemy
            
            local_enemies = 0
            local_friends = 0
            
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    nr, nc = (er + dr) % rows, (ec + dc) % cols
                    if (nr, nc) in enemies: local_enemies += 1
                    if (nr, nc) in friends: local_friends += 1
            
            danger_level = max(0, local_enemies - local_friends) # find numerical advantage
            
            # apply the threat to a small 'engagement zone' around the enemy
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    tr, tc = (er + dr) % rows, (ec + dc) % cols

                    threat_map[tr, tc] += danger_level * 5.0 
                    
        return threat_map

    # def get_reinforcement_map(self, init_map, vis_data):
    #     rows, cols = self.walls.shape
    #     re_map = init_map
        
    #     attackers = [a for a in vis_data["ants"] if self.ant_roles.get(a) == "attacker"] 
        
    #     for ant in attackers:
    #         ar, ac = ant
            
            
    #         for dr in range(-3, 4):
    #             for dc in range(-3, 4):

    #                 if abs(dr) + abs(dc) <= 3:
    #                     nr, nc = (ar + dr) % rows, (ac + dc) % cols
    #                     if not self.walls[nr, nc]:

    #                         re_map[nr, nc] -= 1 # encourage them to move in clusters, lone front ants will wait/go a bit back
    #     return re_map