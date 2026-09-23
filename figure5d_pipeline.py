"""
figure5d_pipeline.py -- reproduce Figure 5d: variant-frequency trajectories in the 15 years
before AML, coloured by number of driver mutations (blue 1 / green 2 / orange 3 / red 4), with
a faint 50-case ensemble and a few bold focal trajectories that show characteristic behaviours
(single-mutant sweeps, a double-mutant up-down reversal, a triple-mutant classic sweep and an
up-down-up double reversal, and an emerging AML clone). Trajectories are sampled at 2000-4000x
sequencing depth. Model = sim_fast.py; default s=0.16, r=2.5, p=3, no offset.

Usage:  python figure5d_pipeline.py --sims 40000 --workers 4 --out Figure5d_trajectories
(Rare reversal patterns need a large pool, hence the default 40k simulations.)
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
from scipy.signal import find_peaks
from sim_fast import (mutation_fitness_generator, mutate_population, mean_fitness,
                      select_population, purge, aml_diagnosis)

N, T, DT, UB = 10**5, 850, 0.1, 1.0e-5
YEARS = list(range(-15, 0))                      # -15 .. -1
MUT = {1: '#4292c6', 2: '#74c476', 3: '#feb24c', 4: '#ef3b2c'}     # colours from original panel d
_S1 = _R = _P = _OFF = None


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
    out = []
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
        if not aml:
            continue
        dx = t * DT; variants = []
        for m in variant_trajectories(clones).values():
            ys, vals = [], []
            for yr in YEARS:
                idx = int(round((dx + yr) / DT))
                if 0 <= idx < T:
                    ys.append(yr); vals.append(float(m['cf'][idx]) * 100)
            if vals and max(vals) > 0.5:
                variants.append({'yr': ys, 'cf': vals, 'k': int(m['k'])})
        out.append(variants)
    return out


# ---------- behaviour detectors (on the true annual %-trajectory; idx0=-15yr) ----------
def already_swept(y):
    y = np.asarray(y, float); return y.min() if (y.min() >= 82 and y.max()-y.min() < 16) else None
def finishing_early(y):
    y = np.asarray(y, float)
    if y[-1] < 95 or not (35 <= y[0] < 90): return None
    hi = np.where(y >= 95)[0]
    return (100-hi[0]) + (90-y[0])*0.2 if (len(hi) and hi[0] <= 8 and y[hi[0]:].min() >= 85) else None
def updown(y):
    y = np.asarray(y, float); n = len(y); pi = int(np.argmax(y))
    if y.max() < 15 or pi in (0, n-1): return None
    return y[pi]-y[pi:].min() if (y[pi]-y[pi:].min() >= 12 and y[-1] < y[pi]-10 and y[:pi+1].min() < y[pi]-8) else None
def orange_sweep(y):
    y = np.asarray(y, float); n = len(y); pi = int(np.argmax(y))
    return y[-1] if (y.max() >= 55 and y[-1] >= 0.82*y.max() and pi >= n-4 and y.max()-y[0] >= 25) else None
def orange_udu(y):
    y = np.asarray(y, float); n = len(y)
    if y.max() < 25 or y[0] >= 25: return None
    mx, _ = find_peaks(y, prominence=5); mn, _ = find_peaks(-y, prominence=5); best = None
    for i in mx:
        for j in mn:
            if j > i and y[j] > 2:
                dip = y[i]-y[j]; kmax = j+int(np.argmax(y[j:])); kick = y[kmax]-y[j]
                if dip >= 8 and kick >= 12 and kmax >= n-4 and y[i]-y[0] >= 15:
                    best = max(best if best is not None else -1, kick + dip*0.2)
    return best
def red_aml(y):                      # emerges late (stays ~0 until the last point, clears the orange re-expansion)
    y = np.asarray(y, float); n = len(y)
    return y[-1] if (10 <= y[-1] <= 20 and y[:n-2].max() < 2.5) else None

DETECTORS = [('blue_old', 1, already_swept), ('blue_fin', 1, finishing_early),
             ('green_ud', 2, updown), ('orange_sw', 3, orange_sweep),
             ('orange_udu', 3, orange_udu), ('red', 4, red_aml)]


def lighten(h, f=0.5):
    c = np.array([int(h[i:i+2], 16) for i in (1, 3, 5)])/255; c = c*(1-f)+f
    return '#%02x%02x%02x' % tuple(int(round(x*255)) for x in c)


def make_figure(cases, out, s1, r, p):
    MUT_LIGHT = {k: lighten(v) for k, v in MUT.items()}
    best = {}
    for variants in cases:
        for v in variants:
            for key, kk, fn in DETECTORS:
                if v['k'] == kk:
                    sc = fn(v['cf'])
                    if sc is not None and sc > best.get(key, (-1e9, None))[0]:
                        best[key] = (sc, v)
    focal = [best[k][1] for k, _, _ in DETECTORS if k in best]
    rng = np.random.default_rng(7)
    def seq(cf):                       # sample each point at 2000-4000x depth (VAF = cf/2)
        out = []
        for f in cf:
            d = int(rng.integers(2000, 4001))
            out.append(rng.binomial(d, min(max(f/200.0, 0.0), 1.0)) / d * 200.0)
        return out
    sample = [cases[i] for i in rng.choice(len(cases), min(50, len(cases)), replace=False)]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for variants in sample:
        for v in variants:
            if max(v['cf']) < 1: continue
            ax.plot(v['yr'], seq(v['cf']), color=MUT_LIGHT[v['k']], lw=1.8, alpha=0.85, zorder=2,
                    solid_capstyle='round', path_effects=[pe.Stroke(linewidth=3.0, foreground='white'), pe.Normal()])
    for v in focal:
        ax.plot(v['yr'], seq(v['cf']), color=MUT[v['k']], lw=4.5, zorder=6, solid_capstyle='round',
                path_effects=[pe.Stroke(linewidth=8.0, foreground='white'), pe.Normal()])
    ax.set_xlim(-15, -1); ax.set_ylim(0, 105)
    ax.set_xticks(range(-15, 0, 2)); ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel('years before AML diagnosis', fontsize=13)
    ax.set_ylabel('fraction of cells (%)', fontsize=13)
    ax.legend(handles=[Patch(color=MUT[k], label=f"{k} driver{'s' if k > 1 else ''}") for k in [1, 2, 3, 4]],
              loc='lower center', bbox_to_anchor=(0.5, 1.005), frameon=False, ncol=4, fontsize=11,
              handlelength=1.4, columnspacing=1.6)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    with PdfPages(out+'.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    fig.savefig(out+'.png', dpi=150, bbox_inches='tight')
    print(f"highlighted: {[k for k,_,_ in DETECTORS if k in best]}")
    print(f"wrote {out}.pdf / .png")


def main():
    global _S1, _R, _P, _OFF
    ap = argparse.ArgumentParser()
    ap.add_argument('--s', type=float, default=0.16); ap.add_argument('--r', type=float, default=2.5)
    ap.add_argument('--p', type=int, default=3); ap.add_argument('--offset', type=float, default=0.0)
    ap.add_argument('--sims', type=int, default=40000); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', type=str, default='Figure5d_trajectories')
    a = ap.parse_args(); _S1, _R, _P, _OFF = a.s, a.r, a.p, a.offset
    per = a.sims // a.workers
    print(f"running {a.sims} sims at s={a.s}, r={a.r}, p={a.p} ...")
    with Pool(a.workers) as pool:
        parts = pool.map(worker, [(per, 700000 + i) for i in range(a.workers)])
    cases = [c for pp in parts for c in pp]
    print(f"{len(cases)} AML cases")
    make_figure(cases, a.out, a.s, a.r, a.p)


if __name__ == '__main__':
    main()
