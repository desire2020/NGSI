import os
import subprocess


def run(script, jobs):
    for job in jobs:
        subprocess.call(['python3', '-W', 'ignore', script] + list(job))


