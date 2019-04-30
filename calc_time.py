import os
import time

l = [
#  'synthetic_1.0_bctf',
#  'synthetic_1.0_bmf',
#  'synthetic_1.0_chain',
#  'synthetic_1.0_gsm',
#  'synthetic_1.0_ibp',
#  'synthetic_1.0_irm',
#  'synthetic_1.0_kf',
#  'synthetic_1.0_mog',
#  'synthetic_1.0_pmf',
  'synthetic_1.0_sparse'
]
time_dict = {}
template = "python3 experiments.py everything %s"

for title in l:
    start_time = time.perf_counter()
    os.system(template % title)
    end_time = time.perf_counter() - start_time
    time_dict[title] = end_time

print(time_dict)
