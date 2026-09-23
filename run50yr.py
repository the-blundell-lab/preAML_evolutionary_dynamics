"""Resumable driver for the -50yr Fig 5d run. Reproduces the SAME 4 worker streams
(seeds 700000..700003, 10000 sims each) as the main panel by checkpointing each stream's
random + np.random state between chunks. Run repeatedly (each call does CHUNK sims); on the
final call it assembles the cases in the same order as the main panel and draws the figure."""
import os, sys, pickle
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import random
import numpy as np
from multiprocessing import Pool
from sim_fast import (mutation_fitness_generator, mutate_population, mean_fitness,
                      select_population, purge, aml_diagnosis)
from figure5d_50yr_pipeline import make_figure, YEARS, variant_trajectories

N, T, DT, UB = 10**5, 850, 0.1, 1.0e-5
S1, R, P, OFF = 0.16, 2.5, 3, 0.0
NSTREAM, PER_STREAM = 4, 10000
STATE = 'run50_state.pkl'


def worker(task):
    idx, nsims, rs, nps, seed = task
    if rs is None:
        random.seed(seed); np.random.seed(seed)
    else:
        random.setstate(rs); np.random.set_state(nps)
    dfes = [mutation_fitness_generator(P, S1 * R**k, OFF) for k in range(4)]
    got = []
    for _ in range(nsims):
        root = {'clone_size_trajectory': np.zeros(T), 'current_clone_size': N,
                'mutations': {}, 'fitness': 0.0, 'children': [], 'aml_clone': False}
        root['clone_size_trajectory'][0] = N
        clones = {0: root}; counter = {'last_mutation': 0}; aml = False
        for t in range(T):
            mutate_population(clones, counter, UB, dfes, t, DT, T)
            select_population(clones, mean_fitness(clones), t, DT)
            purge(clones)
            if aml_diagnosis(clones):
                aml = True; break
        if not aml:
            continue
        dx = t * DT; variants = []
        for m in variant_trajectories(clones).values():
            ys, vals = [], []
            for yr in YEARS:
                j = int(round((dx + yr) / DT))
                if 0 <= j < T:
                    ys.append(yr); vals.append(float(m['cf'][j]) * 100)
            if vals and max(vals) > 0.5:
                variants.append({'yr': ys, 'cf': vals, 'k': int(m['k'])})
        got.append(variants)
    return idx, got, random.getstate(), np.random.get_state(), nsims


def main():
    chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 6500
    per_call = max(1, chunk // NSTREAM)
    if os.path.exists(STATE):
        st = pickle.load(open(STATE, 'rb'))
    else:
        st = {'cases': [[] for _ in range(NSTREAM)], 'rs': [None]*NSTREAM,
              'nps': [None]*NSTREAM, 'done': [0]*NSTREAM}
    tasks = []
    for i in range(NSTREAM):
        n = min(per_call, PER_STREAM - st['done'][i])
        if n > 0:
            tasks.append((i, n, st['rs'][i], st['nps'][i], 700000 + i))
    if tasks:
        with Pool(len(tasks)) as pool:
            for idx, got, rs, nps, ran in pool.map(worker, tasks):
                st['cases'][idx].extend(got); st['rs'][idx] = rs
                st['nps'][idx] = nps; st['done'][idx] += ran
        pickle.dump(st, open(STATE, 'wb'))
    total = sum(st['done']); ncase = sum(len(c) for c in st['cases'])
    print(f"PROGRESS {total}/{NSTREAM*PER_STREAM} sims, {ncase} AML cases")
    if total >= NSTREAM * PER_STREAM:
        cases = [c for stream in st['cases'] for c in stream]   # same order as main panel
        pickle.dump(cases, open('cases50.pkl', 'wb'))
        make_figure(cases, 'Figure5d_trajectories_50yr', S1, R, P)
        print("ALL DONE")


if __name__ == '__main__':
    main()
