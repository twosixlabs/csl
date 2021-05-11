import os
import multiprocessing as mp

epsilon = 1
epochs = 24
trials = 30
distance = 8
filename = '/home/jnear/co/temp/auditing-dpsgd/datasets/fmnist/clipbkd-new-8.npy'

NUM_THREADS = 32

def exp_run(cmd):
    print(cmd)
    os.system(cmd)

commands = []
for _ in range(NUM_THREADS):
    cmd = f'python data_poisoning_experiment.py {epsilon} {epochs} {trials} {distance} {filename}'
    commands.append(cmd)

pool = mp.Pool(NUM_THREADS)
pool.map(exp_run, commands)
