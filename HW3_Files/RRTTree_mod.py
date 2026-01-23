import operator
import numpy as np

class RRTTree(object):

    def __init__(self, bb, task="mp"):

        self.bb = bb
        self.task = task
        self.vertices = {}
        self.edges = {}

        # inspecion planning properties
        if self.task == "ip":
            self.max_coverage = 0
            self.max_coverage_id = 0

    def get_root_id(self):
        '''
        Returns the ID of the root in the tree.
        '''
        return 0

    def add_vertex(self, config, inspected_points=None):
        '''
        Add a state to the tree.
        @param config Configuration to add to the tree.
        '''
        vid = len(self.vertices)
        self.vertices[vid] = RRTVertex(config=config, inspected_points=inspected_points)



        return vid

    def add_edge(self, sid, eid, edge_cost=0):
        '''
        Adds an edge in the tree.
        @param sid start state ID
        @param eid end state ID
        '''
        self.edges[eid] = sid
        self.vertices[eid].set_cost(cost=self.vertices[sid].cost + edge_cost)
        self.vertices[sid].children.append(eid)

    def is_goal_exists(self, config):
        '''
        Check if goal exists.
        @param config Configuration to check if exists.
        '''
        goal_idx = self.get_idx_for_config(config=config)
        if goal_idx is not None:
            return True
        return False

    def get_vertex_for_config(self, config):
        '''
        Search for the vertex with the given config and return it if exists
        @param config Configuration to check if exists.
        '''
        v_idx = self.get_idx_for_config(config=config)
        if v_idx is not None:
            return self.vertices[v_idx]
        return None

    def get_idx_for_config(self, config):
        '''
        Search for the vertex with the given config and return the index if exists
        @param config Configuration to check if exists.
        '''
        valid_idxs = [v_idx for v_idx, v in self.vertices.items() if (v.config == config).all()]
        if len(valid_idxs) > 0:
            return valid_idxs[0]
        return None

    def rewire(self, child_id, new_parent_id, edge_cost):
        '''
        CHANGE 3: Encapsulated Rewire Logic
        Removes child from old parent, attaches to new, and propagates cost.
        '''
        # 1. Remove from old parent
        if child_id in self.edges:
            old_parent_id = self.edges[child_id]
            if child_id in self.vertices[old_parent_id].children:
                self.vertices[old_parent_id].children.remove(child_id)

        # 2. Update pointers
        self.edges[child_id] = new_parent_id
        self.vertices[new_parent_id].children.append(child_id)

        # 3. Update Cost and Propagate
        self.vertices[child_id].set_cost(self.vertices[new_parent_id].cost + edge_cost)
        self._propagate_cost_to_leaves(child_id)

    def _propagate_cost_to_leaves(self, root_id):
        '''
        CHANGE 4: Recursive cost update helper
        '''
        parent_config = self.vertices[root_id].config
        parent_cost = self.vertices[root_id].cost

        for child_id in self.vertices[root_id].children:
            child_vertex = self.vertices[child_id]

            # Recalculate distance (or store it if memory permits, calculation is cheap enough)
            dist = np.linalg.norm(child_vertex.config - parent_config)

            # Update child
            child_vertex.set_cost(parent_cost + dist)

            # Recurse
            self._propagate_cost_to_leaves(child_id)




    def get_nearest_config(self, config):
        '''
        Find the nearest vertex for the given config and returns its state index and configuration
        @param config Sampled configuration.
        '''
        # compute distances from all vertices
        dists = []
        for _, vertex in self.vertices.items():
            dists.append(self.bb.compute_distance(config, vertex.config))

        # retrieve the id of the nearest vertex
        vid, _ = min(enumerate(dists), key=operator.itemgetter(1))

        return vid, self.vertices[vid].config

    def get_edges_as_states(self):
        '''
        Return the edges in the tree as a list of pairs of states (positions)
        '''

        return [[self.vertices[val].config,self.vertices[key].config] for (key, val) in self.edges.items()]

    def get_k_nearest_neighbors(self, config, k):
        '''
        Return k-nearest neighbors
        @param state Sampled state.
        @param k Number of nearest neighbors to retrieve.
        '''
        dists = []
        for _, vertex in self.vertices.items():
            dists.append(self.bb.compute_distance(config, vertex.config))

        dists = np.array(dists)
        knn_ids = np.argpartition(dists, k)[:k]
        #knn_dists = [dists[i] for i in knn_ids]

        return knn_ids.tolist(), [self.vertices[vid].config for vid in knn_ids]


class RRTVertex(object):

    def __init__(self, config, cost=0, inspected_points=None):
        self.config = config
        self.cost = cost
        self.inspected_points = inspected_points
        self.children = []

    def set_cost(self, cost):
        '''
        Set the cost of the vertex.
        '''
        self.cost = cost