import numpy as np

def knapsack_dp(values, weights, capacity):
    """
    Classic 0/1 Knapsack Dynamic Programming implementation.
    Selects the optimal shots to maximize score without exceeding the 15% time limit.
    """
    n = len(values)
    # Convert weights and capacity to integers for DP matrix indexing
    weights = np.array(weights, dtype=np.int32)
    capacity = int(capacity)
    
    dp = np.zeros((n + 1, capacity + 1), dtype=np.float32)
    
    # Build the DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
                
    # Backtrack to find which shots were selected
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i-1)
            w -= weights[i-1]
            
    return selected[::-1] # Return in chronological order

def generate_summary(frame_scores, change_points, n_frames, limit_ratio=0.15):
    """
    Converts frame-level scores into a final binary key-shot summary.
    """
    # 1. Calculate shot-level weights (durations) and values (importance)
    shot_weights = []
    shot_values = []
    
    for shot in change_points:
        start, end = shot[0], shot[1]
        # Length of the shot
        shot_weights.append(end - start + 1)
        # The value is the sum of the frame scores inside this shot
        shot_values.append(np.sum(frame_scores[start:end+1]))
        
    # 2. Define the capacity (maximum 15% of the total video length)
    capacity = int(n_frames * limit_ratio)
    
    # 3. Run the Knapsack algorithm to select the optimal shots
    selected_shots = knapsack_dp(shot_values, shot_weights, capacity)
    
    # 4. Generate the final binary frame array
    machine_summary = np.zeros(n_frames, dtype=np.int32)
    for shot_idx in selected_shots:
        start, end = change_points[shot_idx][0], change_points[shot_idx][1]
        machine_summary[start:end+1] = 1 # Mark these frames as selected
        
    return machine_summary

def evaluate_summary(machine_summary, user_summary):
    """
    Calculates the F1-score between the AI's summary and human annotations.
    SumMe and TVSum have multiple human annotators, so we average the score.
    """
    machine_summary = np.array(machine_summary, dtype=np.float32)
    user_summary = np.array(user_summary, dtype=np.float32)
    
    # user_summary is usually shape (num_users, n_frames)
    if len(user_summary.shape) == 1:
        user_summary = user_summary.reshape(1, -1)
        
    num_users = user_summary.shape[0]
    f_scores = []
    
    for user_idx in range(num_users):
        gt_summary = user_summary[user_idx]
        
        # Calculate overlap
        overlap = np.sum(machine_summary * gt_summary)
        
        # Calculate Precision and Recall
        precision = overlap / (np.sum(machine_summary) + 1e-8)
        recall = overlap / (np.sum(gt_summary) + 1e-8)
        
        # Calculate F1-Score
        if precision == 0 and recall == 0:
            f_score = 0.0
        else:
            f_score = (2 * precision * recall) / (precision + recall)
            
        f_scores.append(f_score)
        
    # The final score for the video is the average across all human annotators
    return np.mean(f_scores)