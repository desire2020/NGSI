# experiment directories
RESULTS_PATH = 'data/results'
CACHE_PATH = 'data/cached'
REPORT_PATH = 'data/reports'
TRAIN_NUM = 1000
VAL_NUM = 30
CANDIDATE_NUM = 9
LOAD = True

# 'single_process' to run in a single process, 'parallel' to use GNU Parallel
SCHEDULER = 'parallel'

# How much improvement would be considered acceptable

THRESHOLD = 0.3

# additional options for GNU Parallel
DEFAULT_NUM_JOBS = 4
JOBS_PATH = 'data/job_info'
