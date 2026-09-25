"""
Fig 5d equivalent for virtual CONTROLS (individuals with no AML by age 85). Controls have no
diagnosis, so — as in the paper's matched-control panels — each control is aligned to a matched
AML diagnosis age sampled from the case distribution, and its variant trajectories are drawn over
the 15 years before that age. Same style as Fig 5d (coloured by driver number, 2000-4000x
sequencing depth, white-outlined lines) but no highlighted trajectories.

Usage:  python figure5d_controls.py --s 0.16 --r 2.5 --p 3 --sims 8000 --out Figure5d_controls
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import random
import numpy as np
from multiprocessing import Pool
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from sim_fast import (mutation_fitness_generator, mutate_population, mean_fitness,
                      select_population, purge, aml_diagnosis)

N, T, DT, UB = 10**5, 850, 0.1, 1.0e-5
STORE_AGES = list(range(45, 85))                 # ages sampled/stored per control (annual)
MUT = {1: '#4292c6', 2: '#74c476', 3: '#feb24c', 4: '#ef3b2c'}
_S1 = _R = _P = _OFF = None
CTRL_CAP = 120                                   # controls stored per worker


def variant_trajectories(clones):
    pop = np.zeros(T)
    for e in clones.values():
        pop = pop + e['clone_size_trajectory']
    traj = {}
    for e in clones.values():
        k = len(e['mutations'])
        frac = np.divide(e['clone_size_trajectory'], pop, out=np.zeros(T), where=pop > 0)
        for mid in e['mutations']:
            if mid in traj:
                traj[mid]['cf'] += frac; traj[mid]['k'] = min(traj[mid]['k'], k)
            else:
                traj[mid] = {'cf': frac.copy(), 'k': k}
    return traj


def worker(args):
    nsim, seed = args
    random.seed(seed); np.random.seed(seed)
    dfes = [mutation_fitness_generator(_P, _S1 * _R**k, _OFF) for k in range(4)]
    case_dx = []; controls = []
    for _ in range(nsim):
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
        if aml:
            case_dx.append(t * DT)                       # for the matched-age distribution
        elif len(controls) < CTRL_CAP:
            variants = []
            for m in variant_trajectories(clones).values():
                cf = [round(float(m['cf'][int(round(ag/DT))]) * 100, 3) for ag in STORE_AGES]
                if max(cf) > 0.5:
                    variants.append({'cf': cf, 'k': int(m['k'])})
            controls.append(variants)
    return case_dx, controls


def main():
    global _S1, _R, _P, _OFF
    ap = argparse.ArgumentParser()
    ap.add_argument('--s', type=float, default=0.16); ap.add_argument('--r', type=float, default=2.5)
    ap.add_argument('--p', type=int, default=3); ap.add_argument('--offset', type=float, default=0.0)
    ap.add_argument('--sims', type=int, default=8000); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--n', type=int, default=50, help='controls to plot')
    ap.add_argument('--out', type=str, default='Figure5d_controls')
    a = ap.parse_args(); _S1, _R, _P, _OFF = a.s, a.r, a.p, a.offset
    per = a.sims // a.workers
    print(f"running {a.sims} sims at s={a.s}, r={a.r}, p={a.p} ...")
    with Pool(a.workers) as pool:
        parts = pool.map(worker, [(per, 800000 + i) for i in range(a.workers)])
    case_dx = np.array([d for cd, _ in parts for d in cd])
    controls = [c for _, cc in parts for c in cc]
    # matched diagnosis ages must leave a full 15-yr window inside the stored range
    dxs = case_dx[(case_dx >= STORE_AGES[0] + 15) & (case_dx <= STORE_AGES[-1] + 1)]
    print(f"{len(controls)} controls stored, {len(case_dx)} case diagnosis ages ({len(dxs)} usable for matching)")

    rng = np.random.default_rng(1)
    sample = controls[:a.n] if len(controls) >= a.n else controls
    age_to_i = {ag: i for i, ag in enumerate(STORE_AGES)}

    MUT_LIGHT = {k: '#%02x%02x%02x' % tuple(int(round(x*255)) for x in
                 (np.array([int(h[i:i+2], 16) for i in (1, 3, 5)])/255 * 0.5 + 0.5)) for k, h in MUT.items()}
    def seq(vals):
        out = []
        for f in vals:
            d = int(rng.integers(2000, 4001)); out.append(rng.binomial(d, min(max(f/200, 0), 1))/d*200)
        return out

    fig, ax = plt.subplots(figsize=(13, 5.2))
    for variants in sample:
        ref = float(rng.choice(dxs))                    # matched diagnosis age for this control
        yrs = list(range(-15, 0))
        idx = [age_to_i[int(round(ref + yr))] for yr in yrs]
        for v in variants:
            vals = [v['cf'][i] for i in idx]
            if max(vals) < 1:
                continue
            ax.plot(yrs, seq(vals), color=MUT_LIGHT[v['k']], lw=1.8, alpha=0.85, zorder=2,
                    solid_capstyle='round', path_effects=[pe.Stroke(linewidth=3.0, foreground='white'), pe.Normal()])
    ax.set_xlim(-15, -1); ax.set_ylim(0, 105); ax.set_xticks(range(-15, 0, 2)); ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel('years before matched AML diagnosis'); ax.set_ylabel('fraction of cells (%)')
    ax.legend(handles=[Patch(color=MUT[k], label=f"{k} driver{'s' if k > 1 else ''}") for k in [1, 2, 3, 4]],
              loc='lower center', bbox_to_anchor=(0.5, 1.005), frameon=False, ncol=4, handlelength=1.4, columnspacing=1.6)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    with PdfPages(a.out+'.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    fig.savefig(a.out+'.png', dpi=150, bbox_inches='tight')
    print(f"plotted {len(sample)} controls; wrote {a.out}.pdf / .png")


if __name__ == '__main__':
    main()
