import os
import multiprocessing as mp

epsilon = 1
epochs = 1
trials = 1
distance = 8
filename = 'auditing-dpsgd/datasets/p100/p100_8.npy'

NUM_THREADS = 1

def exp_run(cmd):
    print(cmd)
    os.system(cmd)

commands = []
for _ in range(NUM_THREADS):
    cmd = f'python data_poisoning_experiment_baseline.py {epsilon} {epochs} {trials} {distance} {filename}'
    commands.append(cmd)

pool = mp.Pool(NUM_THREADS)
pool.map(exp_run, commands)
