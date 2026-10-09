from modules.diffusion.DiffusionModel import DiffusionModel
import future.utils


class ParameterThresholdModel(DiffusionModel):
    """
        Node Parameters to be specified via ModelConfig

       :param threshold: The node threshold. If not specified otherwise a value of 0.1 is assumed for all nodes.
    """

    def __init__(self, graph, seed=None):
        """
             Model Constructor

             :param graph: A networkx graph object
         """
        super(self.__class__, self).__init__(graph, seed)
        self.available_statuses = {
            "Susceptible": 0,
            "Infected": 1
        }

        self.parameters = {
            "methods": {},
            "nodes": {
                "threshold": {
                    "descr": "Node threshold",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.1
                }
            },
            "edges": {
                "prob": {
                    "descr": "Edge threshold",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.1
                }
            },
        }

        self.edge_probs = None

        self.name = "PLTThreshold"
        self.cascades = []

    def iteration(self, node_status=True):
        """
        Execute a single methods iteration

        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        self.clean_initial_status(list(self.available_statuses.values()))

        actual_status = {node: nstatus for node, nstatus in future.utils.iteritems(self.status)}

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(actual_status)
            if node_status:
                return {"iteration": 0, "status": actual_status.copy(),
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        for u in self.graph.nodes:
            cas_u = []
            if actual_status[u] == 1:
                continue

            neighbors = list(self.graph.neighbors(u))
            if self.graph.directed:
                neighbors = list(self.graph.predecessors(u))

            infected = 0
            for v in neighbors:
                if self.status[v] == 1:
                    cas_u.append((v, u))
                # infected += self.status[v]
                    infected += self.edge_probs[(v, u)]

            if len(neighbors) > 0:
                # infected_ratio = float(infected)/len(neighbors)
                infected_ratio = infected
                if infected_ratio >= self.params['nodes']['threshold'][u]:
                    self.cascades += cas_u
                    actual_status[u] = 1

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1

        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": delta.copy(),
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {},
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}

