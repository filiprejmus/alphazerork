class Node(object):
    """
            A class used to implement the object Node fot the MCTS Tree.
            his is the Node explained on page 17 of the paper.

            attributes
            ----------
            visit_count : int
                number of times the node was visited.
            to_play : int
                the player playing in the node.
            prior : float
                the prior probability of the node being selected estimated by the NN.
            value_sum : int
                the sum of the values accumulated in the node.
            children : dict
                dict encoding the children of the the node. format: {action,node}.
    """
    def __init__(self, prior: float):
        """
            Method for initialisation.

            Parameters
            ----------
            prior : float
                prior probability of the node.
        """
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        """
            Method to check if Node is already expended.
            expanded = has children

            returned value
            ----------
            bool: 'True' if node is expanded otherwise 'False'
        """
        return len(self.children) > 0

    def value(self):
        """
            Method to get the mean value of the node: accumulated value divided by number of visits

            returned value
            ----------
            float: the mean value
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
