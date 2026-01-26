import numpy as np
import time

import environment
# from RRTTree import RRTTree
from RRTTree_mod import RRTTree


class RRT_STAR(object):
    def __init__(self, max_step_size, max_itr, bb):
        self.max_step_size = max_step_size
        self.max_itr = max_itr
        self.bb = bb
        self.tree = RRTTree(bb)
        # testing variables
        self.t_curr = 0
        self.itr_no_goal_limit = 250
        self.goal_prob = 0.1
        self.sample_rotation = 0.1 #TODO - what is this ??
        # self.TWO_PI = 2 * math.pi
        self.last_cost = -1 #TODO - what is this ??
        self.last_ratio = 1 #TODO - what is this ??
        # --- STORAGE ---
        self.success_list = []
        self.cost_list = []
        self.goal_id = None
        self.stop_on_goal = True


    def find_path(self, start_conf, goal_conf):
        # TODO: HW3 3
        self.start = start_conf
        self.goal = goal_conf
        """
                Compute and return the plan.
                """
        self.tree.add_vertex(self.start)
        print(self.max_itr)

        for i in range(self.max_itr):
            # 2. Sample
            q_rand = self.bb.sample_random_config(self.goal_prob, self.goal)

            # 3. Nearest Neighbor
            near_id, q_near = self.tree.get_nearest_config(q_rand)

            # 4. Extend
            q_new, dist_new = self.extend(q_near, q_rand)

            # 5. Validity Checks
            if self.bb.config_validity_checker(q_new):
                if self.bb.edge_validity_checker(q_near, q_new):

                    # Check if this configuration is the goal
                    is_at_goal = np.allclose(q_new, self.goal, atol=1e-6)

                    # --- RRT* Dynamic k ---
                    n = len(self.tree.vertices)
                    k = int(np.ceil(2 * np.log(n)))
                    if k < 1: k = 1
                    if k >= n: k = n - 1

                    # --- BRANCH 1: GOAL ALREADY EXISTS ---
                    if is_at_goal and self.goal_id is not None:
                        # 1. Query neighbors (Goal IS in the tree, so it will return itself)
                        neighbor_ids, neighbor_configs = self.tree.get_k_nearest_neighbors(q_new, k)

                        # 2. FILTER: Remove the goal itself to prevent self-loops
                        valid_neighbors = [
                            (nid, nconf) for nid, nconf in zip(neighbor_ids, neighbor_configs)
                            if nid != self.goal_id
                        ]

                        if valid_neighbors:
                            filt_ids, filt_confs = zip(*valid_neighbors)

                            # 3. Find best parent among these VALID neighbors
                            best_parent_id, edge_cost = self.connect_shortest_valid(
                                q_new, filt_ids, filt_confs, near_id, q_near, dist_new
                            )

                            # 4. If best parent is different (and cheaper), rewire existing goal
                            # Note: connect_shortest_valid returns near_id if no better option found.
                            # We only rewire if we actually found a valid, better parent.
                            current_goal_cost = self.tree.vertices[self.goal_id].cost
                            new_path_cost = self.tree.vertices[best_parent_id].cost + edge_cost

                            if new_path_cost < current_goal_cost:
                                self.tree.rewire(self.goal_id, best_parent_id, edge_cost)

                    # --- BRANCH 2: REGULAR NODE (OR FIRST TIME GOAL) ---
                    else:
                        # 1. Query neighbors (Standard)
                        neighbor_ids, neighbor_configs = self.tree.get_k_nearest_neighbors(q_new, k)

                        # 2. Connect
                        best_parent_id, edge_cost = self.connect_shortest_valid(
                            q_new, neighbor_ids, neighbor_configs, near_id, q_near, dist_new
                        )

                        new_id = self.tree.add_vertex(q_new)
                        self.tree.add_edge(best_parent_id, new_id, edge_cost)

                        # If this is the first time we found the goal, save the ID
                        if is_at_goal:
                            self.goal_id = new_id

                        # 3. Rewire Neighbors (Standard RRT* step)
                        self.rewire_neighbors(new_id, q_new, neighbor_ids, neighbor_configs)
                        # ------------------------------------------------------
                        #  NEW: STOP ON GOAL CHECK
                        # ------------------------------------------------------
                        if self.stop_on_goal and self.goal_id is not None:
                            # Log the success for this iteration before returning (optional but good for data)
                            cost = self.tree.vertices[self.goal_id].cost
                            self.success_list.append(1)
                            self.cost_list.append(cost)
                            print(f"Goal found at iter {i}. Stopping early.")
                            return self._reconstruct_path(self.goal_id)
                        # ------------------------------------------------------
            # --- METRICS (Every 400 iterations) ---
            if (i + 1) % 1 == 0:
                # print(f"Iter {i + 1}: Recording metrics...")
                if self.goal_id is not None:
                    cost = self.tree.vertices[self.goal_id].cost
                    self.success_list.append(1)
                    self.cost_list.append(cost)
                    # print(f"  -> Path found! Cost: {cost:.4f}")
                else:
                    self.success_list.append(0)
                    self.cost_list.append(np.inf)
                    # print(f"  -> No path yet.")

        # --- END OF LOOP ---
        if self.goal_id is not None:
            return self._reconstruct_path(self.goal_id)

        print("Max iterations reached. No plan found.")
        return None


    def extend(self, x_near, x_rand):
        diff = x_rand - x_near
        dist = np.linalg.norm(diff)
        step_size = self.max_step_size

        if dist <= step_size:
            return x_rand, dist

        q_new = x_near + (diff / dist) * step_size
        return q_new, step_size


    def connect_shortest_valid(self, q_new, neighbor_ids, neighbor_configs, near_id, q_near, dist_to_near):
        min_total_cost = self.tree.vertices[near_id].cost + dist_to_near
        best_parent = near_id
        best_dist = dist_to_near

        for i, neighbor_id in enumerate(neighbor_ids):
            if neighbor_id == near_id: continue

            dist = np.linalg.norm(q_new - neighbor_configs[i])
            potential_cost = self.tree.vertices[neighbor_id].cost + dist

            if potential_cost < min_total_cost:
                if self.bb.edge_validity_checker(neighbor_configs[i], q_new):
                    min_total_cost = potential_cost
                    best_parent = neighbor_id
                    best_dist = dist

        return best_parent, best_dist

    def rewire_neighbors(self, new_id, q_new, neighbor_ids, neighbor_configs):
        new_node_cost = self.tree.vertices[new_id].cost

        for i, neighbor_id in enumerate(neighbor_ids):
            # Standard check: don't rewire node to itself
            if neighbor_id == new_id: continue

            dist = np.linalg.norm(q_new - neighbor_configs[i])

            if (new_node_cost + dist) < self.tree.vertices[neighbor_id].cost:
                if self.bb.edge_validity_checker(q_new, neighbor_configs[i]):
                    self.tree.rewire(neighbor_id, new_id, dist)

    def compute_cost(self, plan):
        if plan is None or len(plan) < 2:
            return 0.0
        diffs = np.diff(plan, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cost = np.sum(dists)
        return cost

    def _reconstruct_path(self, goal_id):
        path = []
        curr_id = goal_id
        while curr_id != self.tree.get_root_id():
            curr_config = self.tree.vertices[curr_id].config
            path.append(curr_config)
            curr_id = self.tree.edges[curr_id]
        path.append(self.tree.vertices[self.tree.get_root_id()].config)
        final_path = np.array(path[::-1])
        return final_path