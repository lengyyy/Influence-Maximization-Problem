#!/usr/bin/env python
import subprocess
import time
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
NUMBER_RUN = 1
INSTANCES = ['network', 'NetHEPT']
k = ['1', '5', '10', '20', '50', '100']
m = ['IC', 'LT']
t = ['60', '200', '600', '1200']
for i in range(NUMBER_RUN):
    for instance in INSTANCES:
        for kk in k:
            for mm in m:
                for tt in t:
                    in_file = DIR_PATH + '/test_data/%s.txt' % instance
                    out_file = open('./output/imp/hastime/%s-%s-%s-%s.txt' % (instance, kk, mm, tt), 'a')
                    # print(dir_path)
                    command = ['python', DIR_PATH + '/IMP.py',
                               '-i', in_file, '-k', kk, '-m', mm, '-b', '1', '-t', tt, '-r', str(time.time())]
                    process = subprocess.Popen(command, stdout=out_file)
                    time_start = time.time()
                    process.wait()
                    time_end = time.time()
                    print(instance, kk, mm, tt, time_end - time_start)


