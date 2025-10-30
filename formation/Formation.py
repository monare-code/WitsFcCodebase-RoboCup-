import numpy as np

def GenerateBasicFormation():


    formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-7, -2]),  # Left Defender
        np.array([-0, 3]),   # Right Defender
        np.array([7, 1]),    # Forward Left
        np.array([12, 0])      # Forward Right
    ]



    # formation = [
    #     np.array([-13, 0]),    # Goalkeeper
    #     np.array([-10, -2]),  # Left Defender
    #     np.array([-11, 3]),   # Center Back Left
    #     np.array([-8, 0]),    # Center Back Right
    #     np.array([-3, 0]),   # Right Defender
    #     np.array([0, 1]),    # Left Midfielder
    #     np.array([2, 0]),    # Center Midfielder Left
    #     np.array([3, 3]),     # Center Midfielder Right
    #     np.array([8, 0]),     # Right Midfielder
    #     np.array([9, 1]),    # Forward Left
    #     np.array([12, 0])      # Forward Right
    # ]

    return formation


def GenerateStrategicFormation(ball_position, position_history):
    """
    Generates a new formation by shifting the basic formation based on the
    ball's position (x, y).

    - Field players shift relative to the ball.
    - Goalkeeper adjusts y-position to cover the goal.
    - All positions are clamped to the field boundaries.

    Args:
        ball_position (np.array): A numpy array [x, y] for the ball's location.

    Returns:
        list: A list of np.array coordinates for the new formation.
    """

    # Get the default formation (for ball at 0,0)
    base_formation = GenerateBasicFormation()

    # Define field boundaries
    MIN_X, MAX_X = -15, 15
    MIN_Y, MAX_Y = -10, 10

    # --- Goalkeeper Logic ---
    # We assume the goalkeeper's x-position is fixed.
    # Their y-position tracks the ball's y, but is clamped to a "goal area".
    # Let's assume the goalposts are at y=-4 and y=4 for clamping.
    GOAL_MIN_Y, GOAL_MAX_Y = -4, 4

    # Get base goalkeeper position
    base_gk_pos = base_formation[0]

    # New y is the ball's y, clamped to the goal area
    new_gk_y = np.clip(ball_position[1], GOAL_MIN_Y, GOAL_MAX_Y)

    # New gk pos (x is fixed from base, y is new)
    # We also clamp to the main field boundaries, just in case.
    new_gk_x_clamped = np.clip(base_gk_pos[0], MIN_X, MAX_X)
    new_gk_y_clamped = np.clip(new_gk_y, MIN_Y, MAX_Y)

    new_gk_pos = np.array([new_gk_x_clamped, new_gk_y_clamped])

    # Start the new formation list
    new_formation = [new_gk_pos]

    # --- Field Player Logic ---
    # The other players shift their entire formation based on the ball's position.
    # The base_formation[i] is their position when the ball is at (0,0).
    # The new position is simply base_pos + ball_pos.

    base_field_players = base_formation[1:]

    for base_pos in base_field_players:
        # New position is the base position offset by the ball's current position
        new_pos = base_pos + ball_position

        # Clamp the new position to the field boundaries
        clamped_x = np.clip(new_pos[0], MIN_X, MAX_X)
        clamped_y = np.clip(new_pos[1], MIN_Y, MAX_Y)

        new_formation.append(np.array([clamped_x, clamped_y]))

    return new_formation
