"""Stochastic branching model of pre-leukaemic clonal evolution used for all Figure 5
panels. For speed, mutate_population iterates a lightweight snapshot of
(id, size, mutations) taken before adding children, rather than deep-copying the whole
clones dict every step: existing clones' sizes/mutations are not modified inside
mutate_population, and children are not iterated in the step they are born. RNG call
order is preserved, so a run is fully reproducible from its seed."""
import numpy as np
import random
import scipy.integrate as integrate

def cumulative_function(target_s, p, sb, offset):
    continuous_DFE = lambda s: np.exp(-((s-offset)/sb)**p)
    s_lim = 100*sb
    normalization = integrate.quad(continuous_DFE, 0.0, s_lim)[0]
    cumulative = integrate.quad(continuous_DFE, 0.0, target_s)[0]
    return cumulative/normalization

def mutation_fitness_generator(p, sb, offset=0.1):
    nq = 1000; d = {}; s_lim = 100*sb; prev = 0
    for target_s in np.linspace(0, s_lim, 1001):
        cur = int(np.round(cumulative_function(target_s, p, sb, offset)*nq))
        for q in range(prev, cur):
            d[q] = target_s
        prev = cur
        if cur == nq:
            d[cur] = target_s
            break
    return d

def random_mut_fit_generator(gen):
    n = len(gen)-1
    return gen[int(np.round(random.random()*n))]

def mutate_population(clones, last, mut_rate, DFEs, t, dt, T):
    mc = last['last_mutation']
    # lightweight snapshot (equivalent to iterating the deepcopy in the original)
    snapshot = [(cid, e['current_clone_size'], e['mutations']) for cid, e in clones.items()]
    for clone_id, size, mutations in snapshot:
        nmut = len(mutations)
        if nmut < 4 and size > 0:
            n_children = np.random.poisson(size*mut_rate*dt)
            DFE = DFEs[nmut]
        else:
            n_children = 0
        for _ in range(n_children):
            mc += 1
            child_id = mc
            new_fit = random_mut_fit_generator(DFE)
            child = {}
            child['clone_size_trajectory'] = np.zeros(T)
            child['clone_size_trajectory'][t] = 1
            child['parent'] = clone_id
            child['children'] = []
            child['aml_clone'] = False
            child['current_clone_size'] = 1
            child['occurence_time'] = t*dt
            child['mutations'] = dict(mutations)  # copy parent's mutations
            child['mutations'][mc] = new_fit
            child['fitness'] = sum(child['mutations'].values())
            clones[child_id] = child
            clones[clone_id]['children'].append(child_id)
    last['last_mutation'] = mc

def mean_fitness(clones):
    tot = 0
    for e in clones.values():
        tot += e['current_clone_size']
    mf = 0.0
    for e in clones.values():
        mf += e['fitness']*e['current_clone_size']/tot
    return mf

def select_population(clones, mf, t, dt):
    for e in clones.values():
        ne = e['current_clone_size']*np.exp((e['fitness']-mf)*dt)
        ns = np.random.poisson(ne) if ne > 0 else 0
        e['current_clone_size'] = ns
        e['clone_size_trajectory'][t] = ns

def aml_diagnosis(clones):
    tot = 0
    for e in clones.values():
        tot += e['current_clone_size']
    aml = False
    for e in clones.values():
        if len(e['mutations']) == 4 and e['current_clone_size']/tot > 0.5:
            aml = True
            e['aml_clone'] = True
    return aml

def purge(clones):
    rm = [cid for cid, e in clones.items()
          if e['current_clone_size'] == 0 and len(e['children']) == 0
          and np.max(e['clone_size_trajectory']) < 10]
    for cid in rm:
        del clones[cid]

def run(s1=0.16, r=2.5, number_of_sims=1000, N=10**5, T=850, dt=0.1,
        Ub=1.0e-5, p=3, offset=0.0, seed=0, keep_case_clones=True):
    random.seed(seed); np.random.seed(seed)
    s2, s3, s4 = s1*r, s1*r*r, s1*r*r*r
    DFEs = [mutation_fitness_generator(p, s1, offset),
            mutation_fitness_generator(p, s2, offset),
            mutation_fitness_generator(p, s3, offset),
            mutation_fitness_generator(p, s4, offset)]
    dx_ages = []
    case_count = 0
    cases = {}
    for sim in range(number_of_sims):
        root = {'clone_size_trajectory': np.zeros(T), 'mutations': {},
                'current_clone_size': N, 'fitness': 0.0, 'children': [], 'parent': []}
        root['clone_size_trajectory'][0] = N
        clones = {0: root}
        last = {'last_mutation': 0}
        aml = False
        for t in range(T):
            mutate_population(clones, last, Ub, DFEs, t, dt, T)
            mf = mean_fitness(clones)
            select_population(clones, mf, t, dt)
            purge(clones)
            aml = aml_diagnosis(clones)
            if aml:
                dx_ages.append(t*dt)
                break
        if aml:
            case_count += 1
            if keep_case_clones:
                cases[sim] = {'clones': clones, 'diagnosis_time': t*dt}
    return {'case_count': case_count, 'number_of_sims': number_of_sims,
            'fraction': case_count/number_of_sims, 'dx_ages': dx_ages,
            'simulated_cases': cases}

def conditioned_fitnesses(cases):
    s, d, tr = [], [], []
    for v in cases.values():
        for kk, vv in v['clones'].items():
            if kk != 0 and vv.get('aml_clone'):
                for cc, (mid, fit) in enumerate(vv['mutations'].items()):
                    if cc == 0: s.append(fit)
                    elif cc == 1: d.append(fit)
                    elif cc == 2: tr.append(fit)
    return s, d, tr
