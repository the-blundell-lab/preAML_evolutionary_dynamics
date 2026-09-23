"""
figure5_pipeline.py  --  end-to-end reproduction of the Figure 5 simulation panels
(evolutionary-pattern bar chart 5b + age-incidence curve 5c).

Model: stochastic branching model of pre-leukaemic clonal evolution (see sim_fast.py).
Published parameters:
    s0 = 0.16 (single-mutant fitness scale), r = 2.5, p = 3, N = 1e5, U = 1e-5,
    dt = 0.1, T = 850, DFE offset a = 0.  AML = a 4-driver clone exceeding 50% cell
    fraction before age 85.

Classification (mutually exclusive; 'late' assigned first):
  * late     : the AML lineage's FIRST driver only crosses VAF 0.1% (0.2% cell fraction)
               within 2 yr of diagnosis.
  * branched : any off-trunk clone with >=1 driver ever exceeds 10% cell fraction.
  * linear   : neither of the above.

Usage:
    python figure5_pipeline.py --s 0.16 --r 2.5 --p 3 --sims 200000 --workers 4 --out Figure5_bc
Outputs: <out>.pdf, <out>.png, <out>_summary.json

Reproduces the published numbers up to Monte-Carlo noise; pass --seed to fix the stream.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')       # avoid numpy thread oversubscription
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse, json
import numpy as np
from multiprocessing import Pool
from scipy.interpolate import make_smoothing_spline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from sim_fast import run

T, DT = 850, 0.1
LATE_WINDOW = 2.0          # years
CELLFRAC_LIMIT = 2*0.001   # VAF 0.1% -> cell fraction 0.2%
BRANCH_THRESH = 0.10       # off-trunk clone >10% cell fraction => branch

# ---- observed data ----------------------------------------------------------
# CRUK AML age-specific incidence (per 100k/yr), 5-yr bands centred 2.5..82.5
# (male & female rates, then averaged).
CRUK_MALE   = np.array([1.1,0.4,0.6,0.7,1.0,0.9,1.2,1.3,1.6,2.1,3.3,4.8,7.5,12.6,18.2,26.7,34.0])
CRUK_FEMALE = np.array([0.9,0.4,0.5,0.7,0.8,1.0,1.3,1.2,1.7,2.1,2.8,3.5,5.5,7.9,10.9,13.6,20.4])
CRUK_COMB   = 0.5*(CRUK_MALE + CRUK_FEMALE)
AGE_CENTRES = np.linspace(2.5, 82.5, 17)
# Observed evolutionary-pattern counts (paper, n = 47 pre-AML cases)
OBS_PATTERN = {'linear': 26, 'branched': 12, 'late': 9}

PINK = (1.00, 0.18, 0.33)
GREY = (0.55, 0.55, 0.55)
GREY3 = '#969696'


def classify(clones, dx_time):
    pop = np.zeros(T)
    for e in clones.values():
        pop = pop + e['clone_size_trajectory']
    pop = pop + 0.1
    trunk = []
    for cid, e in clones.items():
        if len(e['mutations']) == 4 and e.get('aml_clone'):
            trunk = list(e['mutations'].keys()); break
    m1 = trunk[0]; trunk_set = set(trunk)
    cf = np.zeros(T)
    for e in clones.values():
        if m1 in e['mutations']:
            cf = cf + e['clone_size_trajectory']
    cf = cf/pop
    first_detect = np.argmax(cf >= CELLFRAC_LIMIT)*DT
    if dx_time - first_detect <= LATE_WINDOW:
        return 'late'
    for cid, e in clones.items():
        if len(e['mutations']) > 0 and cid not in trunk_set:
            if np.max(e['clone_size_trajectory']/pop) > BRANCH_THRESH:
                return 'branched'
    return 'linear'


def _worker(args):
    nsim, seed, s1, r, p, offset = args
    res = run(s1=s1, r=r, p=p, number_of_sims=nsim, seed=seed, offset=offset, keep_case_clones=True)
    cats, dxs = [], []
    for v in res['simulated_cases'].values():
        cats.append(classify(v['clones'], v['diagnosis_time']))
        dxs.append(v['diagnosis_time'])
    return res['case_count'], cats, dxs


def run_pipeline(s1, r, p, total_sims, workers, seed_base, offset):
    per = total_sims // workers
    jobs = [(per, seed_base + i, s1, r, p, offset) for i in range(workers)]
    with Pool(workers) as pool:
        parts = pool.map(_worker, jobs)
    nsim = per*workers
    cats = [c for _, cs, _ in parts for c in cs]
    dxs = np.array([d for _, _, ds in parts for d in ds])
    return nsim, cats, dxs


def make_figure(s1, r, p, nsim, cats, dxs, out):
    n = len(cats)
    counts = {k: cats.count(k) for k in ['linear', 'branched', 'late']}
    bins = np.linspace(0, 85, 18)
    inc = 100000*np.histogram(dxs, bins)[0]/(5*nsim)
    cum_sim = inc.sum()*5/1e5*100
    cum_obs = CRUK_COMB.sum()*5/1e5*100

    fig, (axB, axC) = plt.subplots(1, 2, figsize=(13, 5))
    # --- evolutionary pattern ---
    cat_order = ['linear', 'branched', 'late']
    ypos = np.arange(3)[::-1]; h = 0.36
    for i, c in enumerate(cat_order):
        y = ypos[i]
        op = 100*OBS_PATTERN[c]/47; sp = 100*counts[c]/n
        axB.barh(y+h/2+0.02, op, height=h, color=GREY, zorder=3)
        axB.barh(y-h/2-0.02, sp, height=h, color=PINK, zorder=3)
        axB.text(op+1.5, y+h/2+0.02, f"{OBS_PATTERN[c]}/47 ({op:.0f}%)", va='center', fontsize=9.5, color=(0.35,)*3)
        axB.text(sp+1.5, y-h/2-0.02, f"{counts[c]}/{n} ({sp:.0f}%)", va='center', fontsize=9.5, color=PINK)
    axB.set_yticks(ypos); axB.set_yticklabels(cat_order, fontsize=15)
    axB.set_xlim(0, 100); axB.set_xticks([0,25,50,75,100]); axB.set_xticklabels([0,25,50,75,100], fontsize=13)
    axB.set_xlabel('%', fontsize=15); axB.set_title('Evolutionary pattern', fontsize=15)
    axB.spines[['top','right']].set_visible(False); axB.tick_params(width=1.4, color=GREY3, length=5)
    axB.legend(handles=[Patch(color=GREY, label='observed (n=47)'),
                        Patch(color=PINK, label=f'simulated s={s1*100:.0f}% (n={n})')],
               fontsize=11, loc='lower right', frameon=False)
    # --- age-incidence, spline-smoothed pink curve, no cap line ---
    m = AGE_CENTRES >= 30
    spl = make_smoothing_spline(AGE_CENTRES[m], np.sqrt(inc[m]), lam=8.0)
    xf = np.linspace(30, 82.5, 400); yf = np.clip(spl(xf), 0, None)**2
    axC.plot(xf, yf, color=PINK, lw=4, zorder=2,
             label=f'simulated s={s1*100:.0f}% ({nsim:,} sims; cum. risk {cum_sim:.2f}%)')
    axC.scatter(AGE_CENTRES, CRUK_COMB, color=GREY, s=95, edgecolors='white', linewidths=1.2,
                zorder=3, label=f'observed CRUK (cum. risk {cum_obs:.2f}%)')
    axC.set_xlim(30, 85); axC.set_ylim(0, max(55, yf.max()*1.1))
    axC.set_xticks([30,40,50,60,70,80]); axC.set_xticklabels([30,40,50,60,70,80], fontsize=13)
    axC.set_xlabel('age (years)', fontsize=15); axC.set_ylabel('incidence (per 100,000 / yr)', fontsize=15)
    axC.set_title('Age-incidence of AML', fontsize=15)
    axC.spines[['top','right']].set_visible(False); axC.tick_params(width=1.4, color=GREY3, length=5)
    axC.legend(fontsize=10.5, loc='upper left', frameon=False)
    plt.tight_layout()
    with PdfPages(out+'.pdf') as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    fig.savefig(out+'.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    summary = {'s1': s1, 'r': r, 'p': p, 'nsim': nsim, 'n_AML': n, 'case_fraction': n/nsim,
               'pattern_counts': counts,
               'pattern_pct': {k: 100*counts[k]/n for k in counts},
               'observed_pattern_counts': OBS_PATTERN,
               'cum_risk_sim_pct': cum_sim, 'cum_risk_obs_pct': cum_obs,
               'incidence_sim_per100k': inc.tolist(), 'age_centres': AGE_CENTRES.tolist()}
    json.dump(summary, open(out+'_summary.json', 'w'), indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s', type=float, default=0.16, help='single-mutant fitness scale s0')
    ap.add_argument('--r', type=float, default=2.5, help='fold-increase in DFE scale per driver')
    ap.add_argument('--p', type=int, default=3, help='DFE stretch exponent')
    ap.add_argument('--offset', type=float, default=0.0, help='DFE offset a (0 = paper eq 3)')
    ap.add_argument('--sims', type=int, default=200000, help='total simulations')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1000, help='base seed (workers use seed..seed+W-1)')
    ap.add_argument('--out', type=str, default='Figure5_bc')
    a = ap.parse_args()
    print(f"running {a.sims} sims at s={a.s}, r={a.r}, p={a.p}, offset={a.offset} on {a.workers} workers ...")
    nsim, cats, dxs = run_pipeline(a.s, a.r, a.p, a.sims, a.workers, a.seed, a.offset)
    s = make_figure(a.s, a.r, a.p, nsim, cats, dxs, a.out)
    print(f"AMLs: {s['n_AML']}  fraction {s['case_fraction']*100:.3f}%")
    print(f"pattern: linear {s['pattern_pct']['linear']:.1f}% | branched "
          f"{s['pattern_pct']['branched']:.1f}% | late {s['pattern_pct']['late']:.1f}%")
    print(f"cum. risk: sim {s['cum_risk_sim_pct']:.2f}% vs obs {s['cum_risk_obs_pct']:.2f}%")
    print(f"wrote {a.out}.pdf / .png / _summary.json")


if __name__ == '__main__':
    main()
