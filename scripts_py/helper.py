import random
import time 
def random_grid_search(*dimensions, n):
    """
    Generates n different combinations of random grid searches given a number of lists (dimensions).

    Args:
    *dimensions: Variable number of lists representing dimensions.
    n (int): Number of combinations to generate.

    Returns:
    list: List of n different combinations of random grid searches.
    """
    combinations = set()  # Use a set to keep track of unique combinations
    start_time = time.time()
    while len(combinations) < n:
        grid_search = tuple(random.choice(dimension) for dimension in dimensions)
        combinations.add(grid_search)

        if time.time() - start_time > 1:
            print(f"Time limit reached: Found {len(combinations)} combinations out of {n} requested. Exiting loop.")
            break
        
    return list(combinations)