#!/usr/bin/env python3
import random
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and sends
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        """
        while True:
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):
    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        """
        # Generate the first message (do not remove this line)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Use minimax with alpha-beta pruning to find the best move
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Send the best move to the game
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Search for the best next move using Iterative Deepening Search (IDS) and the Minimax algorithm with alpha-beta pruning.
        Args:
            initial_tree_node (TreeNode): The initial node of the game tree from which to start the search.
        Returns:
            str: The best move as a string representation.
        """

        # Best move and score
        best_move = -1
        best_score = float("-inf")

        # Iterative Deepening Search (IDS)
        for depth in range(1, 20):

            # Obtain the score of the node
            score = self.minimax_alpha_beta_search(initial_tree_node, depth)

            # Get the best move
            if score > best_score:
                best_score = score
                best_move = initial_tree_node.move

        return ACTION_TO_STR[best_move]

    def minimax_alpha_beta_search(self, node, depth):
        """
        Perform the minimax search with alpha-beta pruning.
        Args:
            node: The current state of the game.
            depth: The maximum depth to search in the game tree.
        Returns:
            The best score for the current player.
        """
        
        # Alpha and Beta for pruning
        alpha = float("-inf")
        beta = float("inf")

        # Best move and score
        value = self.max_value(node, depth, alpha, beta)

        return value
        
    def max_value(self, node, depth, alpha, beta):
        """
        Computes the maximum value for the given node using the minimax algorithm with alpha-beta pruning.
        Args:
            node (Node): The current node in the game tree.
            depth (int): The current depth in the game tree.
            alpha (float): The alpha value for alpha-beta pruning.
            beta (float): The beta value for alpha-beta pruning.
        Returns:
            float: The maximum value computed for the given node.
        """

        # Terminal state
        if depth == 0 or len(node.state.get_fish_positions()) == 0:
            return self.heuristic(node)
        
        # value <- -∞
        value = float("-inf")

        # Move Ordering 
        children = node.compute_and_get_children()
        children.sort(key=lambda child: self.heuristic(child), reverse=True)

        for child in children:
            # Recursively call min_value to evaluate the child node
            value = max(value, self.min_value(child, depth - 1, alpha, beta))
            # If the value is greater than or equal to beta, prune the remaining branches
            if value >= beta:
                return value
            # Update alpha with the maximum value
            alpha = max(alpha, value)

        return value

    def min_value(self, node, depth, alpha, beta):
        """
        Computes the minimum value for the given node using the minimax algorithm with alpha-beta pruning.
        Args:
            node (Node): The current node in the game tree.
            depth (int): The current depth in the game tree.
            alpha (float): The alpha value for alpha-beta pruning.
            beta (float): The beta value for alpha-beta pruning.
        Returns:
            float: The minimum value computed for the given node.
        """

        # Terminal state
        if depth == 0 or len(node.state.get_fish_positions()) == 0:
            return self.heuristic(node)
        
        # value <- +∞
        value = float("inf")

        # Move Ordering 
        children = node.compute_and_get_children()
        children.sort(key=lambda child: self.heuristic(child))

        for child in children:
            # Recursively call max_value to evaluate the child node
            value = min(value, self.max_value(child, depth - 1, alpha, beta))
            # If the value is less than or equal to alpha, prune the remaining branches
            if value <= alpha:
                return value
            # Update beta with the minimum value
            beta = min(beta, value)

        return value

    def heuristic(self, currentNode):
        state = currentNode.state
        score_player, score_opp = state.get_player_scores()

        difference = score_player - score_opp

        hook_positions = state.get_hook_positions()
        hook_player = hook_positions[0]
        hook_opp = hook_positions[1]

        fish_positions = state.get_fish_positions()
        fish_values = state.get_fish_scores()

        for fish_number, fish_pos in fish_positions.items():
            distance_player = self.calculate_distance(hook_player, fish_pos)#distance from player's hook to fish
            distance_opp = self.calculate_distance(hook_opp, fish_pos)#distance from opponent's hook to fish

            fish_value = fish_values[fish_number]

            if distance_player == 0: #fish caught by player
                difference += fish_value
            elif distance_opp == 0: #fish caught by opponent
                difference -= fish_value
            else:
                if fish_value > 0:
                    difference += (fish_value /distance_player) + 3
                    difference -= fish_value / distance_opp
                else:
                    difference -= abs(fish_value)/ distance_player
                    difference += abs(fish_value)/ distance_opp
            
            if distance_player == distance_opp:
                difference -= abs(fish_value) /distance_player

        caught_fish = state.get_caught()
        #adding the values of the fish caught to the difference in score
        if caught_fish[0] is not None:
            difference += fish_values[caught_fish[0]] 
        if caught_fish[1] is not None:
            difference -= fish_values[caught_fish[1]]

        return difference #finally, the the score difference is returned, taking into consideration the heuristic value

    #auxiliary method to calculate distance between two positions
    def calculate_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
