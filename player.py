#!/usr/bin/env python3

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
import time

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
        self.start_time = None
        self.timeout_duration = 0.06 # Lower than 0.075
        self.states_stored = {}

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        """
        first_msg = self.receiver()

        while True:
            msg = self.receiver()
            if msg["game_over"]:		
                return

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Initialize variables
            self.start_time = time.time()
            self.states_stored = {}

            # Use minimax with iterative deepening to find the best move
            best_move = self.search_best_next_move(node)

            # Send the best move to the game
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use your minimax model to find the best possible next move for player 0 (green boat).
        """
        best_score = float("-inf")
        best_move = 0

        # Compute children for move ordering
        children = initial_tree_node.compute_and_get_children()
        children.sort(key=lambda child: self.heuristic(child), reverse=True)

        for max_depth in range(1, 20):
            for child in children:
                score = self.minimax(
                    child, alpha=float("-inf"), beta=float("inf"),
                    player=False, max_depth=max_depth
                )
                if score > best_score:
                    best_move = child.move
                    best_score = score

                # Check timeout
                if time.time() - self.start_time > self.timeout_duration:
                    return ACTION_TO_STR[best_move]

        return ACTION_TO_STR[best_move]

    def minimax(self, current_node, alpha, beta, player, max_depth):
        """
        Perform the minimax search with alpha-beta pruning.
        """
        current_node.compute_and_get_children()

        # Hash the state for the states table
        state_key = self.state_to_string(current_node)

        # If the state is already stored
        if state_key in self.states_stored:

            # We retrive the stored value
            cached = self.states_stored[state_key]
            cached_value, cached_alpha, cached_beta = cached[0], cached[1], cached[2]

            # We know that the value is bounded by alpha
            if cached_value == cached_alpha:
                alpha = cached_alpha

            # We know that the value is bounded by beta
            if cached_value == cached_beta:
                beta = cached_beta

            # We know that the value is bounded by alpha and beta
            if cached_value > cached_alpha and cached_value < cached_beta:
                return cached_value

        # Check terminal conditions
        if (
            current_node.depth == max_depth
            or not current_node.children
            or time.time() - self.start_time > self.timeout_duration
        ):
            return self.heuristic(current_node)

        # Sort children for move ordering, if it is a max player we sort in descending order
        children = current_node.children
        children.sort(key=lambda child: self.heuristic(child), reverse=player)

        # Perform the minimax search with alpha-beta pruning
        if player:
            value = float("-inf")
            for child in children:
                value = max(value, self.minimax(child, alpha, beta, not player, max_depth))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            self.states_stored[state_key] = [value, alpha, beta]
            return value
        else:
            value = float("inf")
            for child in children:
                value = min(value, self.minimax(child, alpha, beta, not player, max_depth))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            self.states_stored[state_key] = [value, alpha, beta]
            return value

    def heuristic(self, node):
        """
        Calculate the heuristic value of a given node.
        """
        state = node.state
        p0, p1 = state.get_player_scores()
        hook_positions = state.get_hook_positions()
        fish_positions = state.get_fish_positions()

        score_diff = p0 - p1
        fish_value = 0

        if fish_positions:
            for fish, pos in fish_positions.items():
                fish_score = state.fish_scores[fish]
                my_distance = self.manhattan_distance(hook_positions[0], pos)
                opp_distance = self.manhattan_distance(hook_positions[1], pos)

                # Calculate fish value
                fish_value += fish_score / (my_distance + 1) - fish_score / (2 * opp_distance + 1)

        return score_diff + fish_value

    def manhattan_distance(self, pos1, pos2, width = 20):
        """
        Calculate the Manhattan distance between two positions.
        """
        # Handle the case where the distance wraps around the board (x-axis)
        x_dist = abs(pos1[0] - pos2[0])
        x_dist = min(x_dist, width - x_dist)
        y_dist = abs(pos1[1] - pos2[1])
        return x_dist + y_dist

    def state_to_string(self, node):
        """
        Create a string representation of the game state for hashing.
        """
        state = node.state
        key = (
            f"{state.player}{state.player_scores[0]}{state.player_scores[1]}"
            f"{state.hook_positions[0][0]}{state.hook_positions[0][1]}"
            f"{state.hook_positions[1][0]}{state.hook_positions[1][1]}"
        )
        for fish in state.fish_positions:
            key += f"{state.fish_positions[fish][0]}{state.fish_positions[fish][1]}"
        for score in state.fish_scores.values():
            key += str(score)
        return key