import os
import multiprocessing as mp

epsilon = 1000
epochs = 10
trials = 1
distance = 8
filename = '/home/jnear/co/temp/auditing-dpsgd/datasets/p100/p100_8.npy'

NUM_THREADS = 1

def exp_run(cmd):
    print(cmd)
    os.system(cmd)

commands = []
for _ in range(NUM_THREADS):
    cmd = f'python data_poisoning_experiment.py {epsilon} {epochs} {trials} {distance} {filename}'
    commands.append(cmd)

pool = mp.Pool(NUM_THREADS)
pool.map(exp_run, commands)
