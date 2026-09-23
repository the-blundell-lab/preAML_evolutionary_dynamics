"""
fig5e_pipeline.py -- reproduce Figure 5e end-to-end (joint VAF scatter at ages 40/50/60/70
for future AML cases vs controls) AND compute, at each age:
  * the odds ratio of future AML for a somatic sweep (largest variant VAF > 30%);
  * the positive predictive value (PPV) of a somatic sweep PLUS a second driver
    (largest VAF > 30% AND second-largest VAF > 10%) for future AML -- the quantity
    quoted in the main text.

Model = sim_fast.py. Published parameters: s0 = 0.16, r = 2.5, p = 3, DFE offset a = 0.

Usage:
    python fig5e_pipeline.py --s 0.16 --r 2.5 --p 3 --sims 200000 --workers 4 --out Figure5e
Outputs: <out>.pdf, <out>.png, <out>_odds_ratios.json (contains both the ORs and the PPVs)

The text quotes OR = 98.8 (UK Biobank hotspot sweeps, Fig 5f). This script's simulated OR
is strongly age-dependent; the Mantel-Haenszel age-adjusted value is the single-number summary.

PPV is computed over ALL simulated individuals still undiagnosed at the sampling age
(cases + controls), not over the 100-case sample plotted in the scatter:
    PPV(age) = cases meeting the criterion / (cases + controls meeting the criterion).
Published values (200,200 sims, s0 = 0.16, r = 2.5): 88% (44/50) at age 40, 73% (214/292)
at 50, 49% (564/1152) at 60, 22% (719/3313) at 70.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, json
import random
import numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sim_fast import (mutation_fitness_generator, mutate_population, mean_fitness,
                      select_population, purge, aml_diagnosis)

N, T, DT, UB, R, P = 10**5, 850, 0.1, 1.0e-5, 3, 3
AGES = [40, 50, 60, 70]; AIDX = [int(a/DT) for a in AGES]
SWEEP = 0.30       # 'somatic sweep': largest variant VAF > 30%
SECOND = 0.10      # 'second driver': second-largest variant VAF > 10%
PINK = (1.00, 0.18, 0.33); CTRL = (0.45, 0.45, 0.45)
_S1 = 0.10   # set in main()


def vaf_summary(clones, tA):
    pop = sum(e['clone_size_trajectory'][tA] for e in clones.values())
    if pop <= 0:
        return None                      # not alive (case already diagnosed)
    mut = {}
    for e in clones.values():
        s = e['clone_size_trajectory'][tA]
        if s > 0:
            for m in e['mutations']:
                mut[m] = mut.get(m, 0.0) + s
    if not mut:
        return (0.0, 0.0)                # alive, no surviving variant -> 'no sweep'
    vafs = sorted((0.5*v/pop for v in mut.values()), reverse=True)
    return vafs[0], (vafs[1] if len(vafs) > 1 else 0.0)


def worker(args):
    nsim, seed, s1, offset = args
    random.seed(seed); np.random.seed(seed)
    DFEs = [mutation_fitness_generator(P, s1, offset), mutation_fitness_generator(P, s1*R, offset),
            mutation_fitness_generator(P, s1*R*R, offset), mutation_fitness_generator(P, s1*R*R*R, offset)]
    case_records = []; ctrl = {a: [[], []] for a in AGES}; ncase = 0
    for _ in range(nsim):
        ce = {'clone_size_trajectory': np.zeros(T), 'mutations': {}, 'current_clone_size': N,
              'fitness': 0.0, 'children': [], 'parent': []}
        ce['clone_size_trajectory'][0] = N
        clones = {0: ce}; last = {'last_mutation': 0}; aml = False
        for t in range(T):
            mutate_population(clones, last, UB, DFEs, t, DT, T)
            select_population(clones, mean_fitness(clones), t, DT)
            purge(clones)
            if aml_diagnosis(clones):
                aml = True; break
        ncase += aml
        if aml:                              # keep per-case identity so we can sample cases
            case_records.append({a: vaf_summary(clones, tA) for a, tA in zip(AGES, AIDX)})
        else:
            for a, tA in zip(AGES, AIDX):
                r = vaf_summary(clones, tA)
                if r is not None:
                    ctrl[a][0].append(r[0]); ctrl[a][1].append(r[1])
    return nsim, ncase, case_records, ctrl


def odds_ratio(cL, kL):
    A = int((cL > SWEEP).sum()); B = len(cL)-A
    C = int((kL > SWEEP).sum()); D = len(kL)-C
    a, b, c, d = (A, B, C, D) if min(A, B, C, D) else (A+.5, B+.5, C+.5, D+.5)
    se = np.sqrt(1/a+1/b+1/c+1/d)
    orr = (a*d)/(b*c)
    return orr, np.exp(np.log(orr)-1.96*se), np.exp(np.log(orr)+1.96*se), (A, B, C, D)


def ppv(cL, cS, kL, kS):
    """PPV of future AML given a somatic sweep (largest VAF > SWEEP) *and* a second driver
    (second-largest VAF > SECOND) at this age. Denominator = every simulated individual
    still undiagnosed at this age who meets the criterion (cases + controls)."""
    hit_case = int(((cL > SWEEP) & (cS > SECOND)).sum())
    hit_ctrl = int(((kL > SWEEP) & (kS > SECOND)).sum())
    tot = hit_case + hit_ctrl
    return (hit_case/tot if tot else float('nan')), hit_case, hit_ctrl, tot


def main():
    global _S1, R, P
    ap = argparse.ArgumentParser()
    ap.add_argument('--s', type=float, default=0.16)
    ap.add_argument('--r', type=float, default=2.5)
    ap.add_argument('--p', type=int, default=3)
    ap.add_argument('--offset', type=float, default=0.0, help='DFE offset (0 = paper eq 3)')
    ap.add_argument('--sims', type=int, default=100000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=60000)
    ap.add_argument('--out', type=str, default='Figure5e')
    a = ap.parse_args(); _S1 = a.s; R = a.r; P = a.p
    per = a.sims // a.workers
    print(f"running {a.sims} sims at s={a.s}, offset={a.offset} on {a.workers} workers ...")
    with Pool(a.workers) as pool:
        parts = pool.map(worker, [(per, a.seed+i, a.s, a.offset) for i in range(a.workers)])
    nsim = sum(p[0] for p in parts); ncase = sum(p[1] for p in parts)
    case_records = [rec for _, _, cr, _ in parts for rec in cr]
    ctrl = {ag: [[], []] for ag in AGES}
    for _, _, _, ct in parts:
        for ag in AGES:
            ctrl[ag][0] += ct[ag][0]; ctrl[ag][1] += ct[ag][1]

    # ---- odds ratios + Mantel-Haenszel (from ALL cases and controls) ----
    summary = {'nsim': nsim, 'ncase': ncase, 's1': a.s, 'per_age': {}, 'quoted_OR_biobank': 98.8}
    mh_num = mh_den = 0.0
    print(f"\n{'age':>4} {'OR':>8}  {'95% CI':>16}  case>30%   ctrl>30%")
    for ag in AGES:
        cL = np.array([rec[ag][0] for rec in case_records if rec[ag] is not None])
        cS = np.array([rec[ag][1] for rec in case_records if rec[ag] is not None])
        kL = np.array(ctrl[ag][0]); kS = np.array(ctrl[ag][1])
        orr, lo, hi, (A, B, C, D) = odds_ratio(cL, kL)
        nn = A+B+C+D; mh_num += A*D/nn; mh_den += B*C/nn
        pv, hit_case, hit_ctrl, hit_tot = ppv(cL, cS, kL, kS)
        summary['per_age'][ag] = {'OR': orr, 'CI': [lo, hi], 'case_sweep': [A, A+B], 'ctrl_sweep': [C, C+D],
                                  'ppv': pv, 'ppv_cases': hit_case, 'ppv_controls': hit_ctrl,
                                  'ppv_total': hit_tot,
                                  'n_cases_alive': int(len(cL)), 'n_controls_alive': int(len(kL))}
        print(f"{ag:>4} {orr:>8.1f}  [{lo:>6.1f},{hi:>7.1f}]  {A:>4}/{A+B:<4} {C:>6}/{C+D}")
    mh = mh_num/mh_den; summary['mantel_haenszel_OR'] = mh
    print(f"\nMantel-Haenszel age-adjusted OR = {mh:.1f}   (text quotes 98.8, Biobank)")

    # ---- PPV of a sweep PLUS a second driver (the number quoted in the main text) ----
    summary['ppv_criterion'] = {'largest_VAF_gt': SWEEP, 'second_VAF_gt': SECOND}
    print(f"\nPPV of future AML | largest VAF > {SWEEP:.0%} AND second VAF > {SECOND:.0%}")
    print(f"{'age':>4} {'PPV':>7}   cases/total   (cases, controls meeting criterion)")
    for ag in AGES:
        v = summary['per_age'][ag]
        print(f"{ag:>4} {100*v['ppv']:>6.1f}%   {v['ppv_cases']}/{v['ppv_total']:<7} "
              f"({v['ppv_cases']} cases, {v['ppv_controls']} controls)")

    # ---- scatter: sample of 100 cases (pink) + controls (grey); n labelled per age ----
    rng = np.random.default_rng(0)
    sample = [case_records[i] for i in rng.choice(len(case_records), min(100, len(case_records)), replace=False)]
    with PdfPages(a.out+'.pdf') as pdf:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.3))
        for ax, ag in zip(axes, AGES):
            kL = np.array(ctrl[ag][0]); kS = np.array(ctrl[ag][1]); km = kL > 0
            kLx, kSx = kL[km], kS[km]
            if len(kLx) > 5000:
                idx = rng.choice(len(kLx), 5000, replace=False); kLx, kSx = kLx[idx], kSx[idx]
            cpts = [rec[ag] for rec in sample if rec[ag] is not None and rec[ag][0] > 0]
            ax.scatter(kLx, kSx, color=CTRL, alpha=0.30, s=45, linewidths=0, zorder=1)
            ax.scatter([l for l, s in cpts], [s for l, s in cpts], color=PINK, alpha=1.0, s=95,
                       edgecolors='w', linewidths=0.8, zorder=3)
            ax.plot([SWEEP, SWEEP], [-0.02, 0.52], color='k', lw=1.2, zorder=2)
            ax.set_xlim(-0.02, 0.52); ax.set_ylim(-0.02, 0.52); ax.set_aspect('equal')
            ax.set_xticks([0, .1, .2, .3, .4, .5]); ax.set_yticks([0, .1, .2, .3, .4, .5])
            ax.set_xticklabels(['.0', '.1', '.2', '.3', '.4', '.5'], fontsize=13)
            ax.set_yticklabels(['.0', '.1', '.2', '.3', '.4', '.5'], fontsize=13)
            ax.set_xlabel('largest VAF', fontsize=15)
            if ag == AGES[0]:
                ax.set_ylabel('second largest VAF', fontsize=15)
            ax.set_title(f'{ag} years', fontsize=15); ax.spines[['top', 'right']].set_visible(False)
            ax.text(0.335, 0.47, f"OR = {summary['per_age'][ag]['OR']:.0f}", fontsize=13, fontweight='bold')
            ax.text(0.335, 0.40, f"n = {len(cpts)}", fontsize=12, color=PINK)
        fig.suptitle(f'Figure 5e - {nsim:,} sims (s={a.s*100:.0f}%): sample of {len(sample)} of {ncase} '
                     f'future AML cases (pink) vs controls (grey); age-adjusted OR = {mh:.0f}', fontsize=13, y=1.02)
        plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight')
        fig.savefig(a.out+'.png', dpi=130, bbox_inches='tight'); plt.close(fig)
    json.dump(summary, open(a.out+'_odds_ratios.json', 'w'), indent=2)
    print(f"wrote {a.out}.pdf / .png / _odds_ratios.json")


if __name__ == '__main__':
    main()
