#!/usr/bin/env python
import subprocess
import time
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
NUMBER_RUN = 1

for i in range(NUMBER_RUN):
    seed = str(time.time())
    out_file = open('./output/result1.txt', 'a')
    # print(dir_path)
    command = ['python2', 'IMP_test.py']
    process = subprocess.Popen(command, stdout=out_file)
    time_start = time.time()
    process.wait()
    time_end = time.time()
    print(time_end - time_start)
