import numpy as np


def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points (represented as NumPy arrays).
    """
    return np.linalg.norm(point1 - point2)


def generate_preference_lists(teammate_positions, formation_positions):
    """
    Generates ranked preference lists for players and roles based on proximity.
    """
    num_players = len(teammate_positions)
    players_preferences = {}
    roles_preferences = {}

    # Generate Player Preferences (each player ranks roles)
    for i in range(num_players):
        distances_to_roles = []
        for j in range(len(formation_positions)):
            dist = euclidean_distance(teammate_positions[i], formation_positions[j])
            distances_to_roles.append((dist, j))
        distances_to_roles.sort()
        players_preferences[i] = [role_idx for dist, role_idx in distances_to_roles]

    # Generate Role Preferences (each role ranks players)
    for j in range(len(formation_positions)):
        distances_to_players = []
        for i in range(num_players):
            dist = euclidean_distance(formation_positions[j], teammate_positions[i])
            distances_to_players.append((dist, i))
        distances_to_players.sort()
        roles_preferences[j] = [player_idx for dist, player_idx in distances_to_players]

    return players_preferences, roles_preferences


def role_assignment(teammate_positions, formation_positions):
    """
    Assigns each player to a formation position using the Gale-Shapley stable
    matching algorithm.

    Args:
        teammate_positions (list): A list of NumPy arrays for each player's position.
        formation_positions (list): A list of NumPy arrays for each formation position.

    Returns:
        dict: A dictionary mapping player unum (1-5) to their assigned
              formation position (NumPy array).
    """
    # Step 1: Define Preference Lists
    players_preferences, roles_preferences = generate_preference_lists(
        teammate_positions, formation_positions
    )

    num_players = len(teammate_positions)

    # Step 2: Initialize All Players and Roles as Free
    unmatched_players = list(range(num_players))
    # current_matches maps role_index -> player_index
    current_matches = {role_idx: None for role_idx in range(len(formation_positions))}

    # Keeps track of the next role a player should propose to
    next_proposal_for_player = {player_idx: 0 for player_idx in range(num_players)}

    # Step 3 & 4: Proposal Loop
    while unmatched_players:
        proposing_player_idx = unmatched_players.pop(0)

        # Get the player's preference list
        player_pref_list = players_preferences[proposing_player_idx]

        # Get the index of the next role to propose to
        proposal_rank = next_proposal_for_player[proposing_player_idx]
        target_role_idx = player_pref_list[proposal_rank]

        current_partner_idx = current_matches[target_role_idx]

        # Case 1: Role is free
        if current_partner_idx is None:
            current_matches[target_role_idx] = proposing_player_idx
        # Case 2: Role is already matched
        else:
            # The role checks its own preference list to decide
            role_pref_list = roles_preferences[target_role_idx]
            rank_current_partner = role_pref_list.index(current_partner_idx)
            rank_new_proposer = role_pref_list.index(proposing_player_idx)

            # If the new proposer is preferred, the role swaps partners
            if rank_new_proposer < rank_current_partner:
                # The old partner is now unmatched
                unmatched_players.append(current_partner_idx)
                # The role accepts the new proposal
                current_matches[target_role_idx] = proposing_player_idx
            # Otherwise, the new proposer is rejected and remains unmatched
            else:
                unmatched_players.append(proposing_player_idx)

        # The player has now proposed to this role, so next time they'll try the next one
        next_proposal_for_player[proposing_player_idx] += 1

    # Step 5: Format and Return Final Matches
    point_preferences = {}
    for role_idx, player_idx in current_matches.items():
        # The unum is the player index + 1 (e.g., player 0 is unum 1)
        unum = player_idx + 1
        # The value is the assigned formation position
        assigned_position = formation_positions[role_idx]
        point_preferences[unum] = assigned_position

    return point_preferences