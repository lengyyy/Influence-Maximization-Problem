#!/usr/bin/env python
import subprocess
import time
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
NUMBER_RUN = 1
INSTANCES = ['network', 'NetHEPT']
INSTANCES2_1 = []
INSTANCES2_2 =[]
k = ['1', '5', '10', '20', '50']
m = ['IC', 'LT']
t = ['60', '200', '600', '1200']


for kk in k:
    for mm in m:
        for tt in t:
            s = "network"+"-"+str(kk)+"-"+str(mm)+"-"+str(tt)
            INSTANCES2_1.append(s)
for kk in k:
    for mm in m:
        for tt in t:
            s = "NetHEPT" + "-" + str(kk) + "-" + str(mm) + "-" + str(tt)
            INSTANCES2_2.append(s)
print INSTANCES2_1
print INSTANCES2_2

for i in range(NUMBER_RUN):
    for instance in INSTANCES:
        if instance == 'network':
            i2 = INSTANCES2_1
        else:
            i2 = INSTANCES2_2
        for instance2 in i2:
            for mm in m:
                in_file = DIR_PATH + '/test_data/%s.txt' % instance
                in_file2 = DIR_PATH + '/output2/imp/hastime/%s.txt' % instance2
                out_file = open('./output2/ise/%s.txt' % instance2, 'a')
                # print(dir_path)
                command = ['python', DIR_PATH + '/ISE.py',
                           '-i', in_file, '-s', in_file2, '-m', mm, '-b', '1', '-t', '60', '-r', str(time.time())]
                process = subprocess.Popen(command, stdout=out_file)
                time_start = time.time()
                process.wait()
                time_end = time.time()
                print(instance, instance2, mm, time_end - time_start)


