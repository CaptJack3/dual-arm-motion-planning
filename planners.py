import numpy as np
from RRTTree_mod import RRTTree
import time

class RRT(object):
    """
    Plain RRT (no rewiring).
    - Stops as soon as the goal is connected.
    - Returns a path ONLY if the final node is exactly goal_conf.
    """

    def __init__(self, max_step_size, max_itr, bb):
        self.max_step_size = max_step_size
        self.max_itr = max_itr
        self.bb = bb
        self.tree = RRTTree(bb)

        # sampling
        self.goal_prob = 0.1  # keep your goal bias
        self.stop_on_goal = True

        self.goal_id = None

    def find_path(self, start_conf, goal_conf, manipulator=None):
        """
        Returns: (path, cost)
        - path is an (N x dof) numpy array, ending exactly at goal_conf.
        - cost is total Euclidean cost along the joint path.
        """
        self.start = np.array(start_conf, dtype=float)
        self.goal = np.array(goal_conf, dtype=float)

        # reset tree each call
        self.tree = RRTTree(self.bb)
        self.goal_id = None

        root_id = self.tree.add_vertex(self.start)

        print(self.max_itr)
        self.max_itr = int(self.max_itr * 2)
        for i in range(self.max_itr):
            # 1) Sample
            q_rand = self.bb.sample_random_config(self.goal_prob, self.goal)

            # 2) Nearest
            near_id, q_near = self.tree.get_nearest_config(q_rand)

            # 3) Steer/Extend
            q_new, _ = self.extend(q_near, q_rand)

            # 4) Validity
            if not self.bb.config_validity_checker(q_new):
                continue
            if not self.bb.edge_validity_checker(q_near, q_new):
                continue

            # 5) Add new node
            new_id = self.tree.add_vertex(q_new)
            edge_cost = np.linalg.norm(q_new - q_near)
            self.tree.add_edge(near_id, new_id, edge_cost)

            # 6) IMPORTANT: connect to GOAL exactly (not "near goal")
            # If the goal is reachable from q_new with one step AND edge valid,
            # we explicitly add a GOAL vertex with config == goal.
            dist_to_goal = np.linalg.norm(self.goal - q_new)
            if dist_to_goal <= self.max_step_size and self.bb.edge_validity_checker(q_new, self.goal):
                goal_id = self.tree.add_vertex(self.goal)
                self.tree.add_edge(new_id, goal_id, dist_to_goal)
                self.goal_id = goal_id

                path = self._reconstruct_path(self.goal_id)
                cost = self.compute_cost(path)
                print("RRT solution found !")
                time.sleep(5)
                return path, cost

            # (Optional) if you ever sample the goal exactly and it's valid,
            # you’ll still go through the same check above.

        # no plan
        time.sleep(4)
        print("Max iterations reached. No plan found.")

        return None, np.inf

    def extend(self, x_near, x_rand):
        diff = x_rand - x_near
        dist = np.linalg.norm(diff)

        if dist <= self.max_step_size:
            return x_rand, dist

        q_new = x_near + (diff / dist) * self.max_step_size
        return q_new, self.max_step_size

    def compute_cost(self, plan):
        if plan is None or len(plan) < 2:
            return 0.0
        diffs = np.diff(plan, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        return float(np.sum(dists))

    def _reconstruct_path(self, goal_id):
        path = []
        curr_id = goal_id
        while curr_id != self.tree.get_root_id():
            path.append(self.tree.vertices[curr_id].config)
            curr_id = self.tree.edges[curr_id]
        path.append(self.tree.vertices[self.tree.get_root_id()].config)
        return np.array(path[::-1], dtype=float)


# ---- OPTIONAL: for minimal changes in the rest of your project ----
# If other code imports RRT_STAR, you can alias it:
RRT_STAR = RRT
