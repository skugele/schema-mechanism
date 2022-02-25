import random

RANDOM_SEED = 8675309

# For reproducibility, we we also need to set PYTHONHASHSEED=8675309 in the environment
random.seed(RANDOM_SEED)
