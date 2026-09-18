# Evolutionary dynamics in the decades preceding acute myeloid leukaemia

Code accompanying the manuscript **"Evolutionary dynamics in the decades preceding acute myeloid
leukaemia"**, in which annual blood samples from 47 women who went on to develop AML and 46 matched
controls (UKCTOCS) were sequenced with TETRIS-seq and used to reconstruct clonal evolution in the
decades before diagnosis.

The upstream pipeline — processing raw sequencing data into duplex consensus reads and applying the
TETRIS-seq *in silico* noise-correction model — is in a separate repository:
<https://github.com/the-blundell-lab/TETRIS-seq>.

If a notebook does not render on GitHub, view it on
[nbviewer](https://nbviewer.org/github/the-blundell-lab/preAML_evolutionary_dynamics/tree/main/).

## Getting started

Python 3.11, with:

```
numpy  pandas  scipy  matplotlib  statsmodels  scikit-learn  seaborn
pysam  biopython  openpyxl  jupyter
```

```bash
git clone https://github.com/the-blundell-lab/preAML_evolutionary_dynamics.git
cd preAML_evolutionary_dynamics
jupyter notebook
```

Every notebook opens with a **Data availability** table listing the files it needs and where each one
comes from. Notebooks whose inputs are all in `Data_files/` run as they are; the rest need data under
controlled access (see [Data availability](#data-availability)).

## How the analysis fits together

1. **`Post_processing_variant_calls.ipynb`** — filters the raw beta-binomial variant calls into the
   final somatic and germline call sets, and writes the per-individual trajectory files.
2. **`Code_for_inferring_acquisition_age_and_fitness_v16.py`** — maximum-likelihood inference of the
   fitness and establishment age of every clone, run per individual from 25 random seeds:
   `python Code_for_inferring_acquisition_age_and_fitness_v16.py --sample_name C92_002 --seeds 25`
3. **`Supplementary_Fig_46.ipynb`** — summarises the seed-to-seed spread into the uncertainty
   estimates (standard deviation and 95% optimiser range) reported in Supplementary Table 9.
4. **The figure notebooks** — each reads those outputs and reproduces one or more manuscript figures.

## Figures

| figure | notebook |
|---|---|
| Fig. 1 | `Figure_1.ipynb` |
| Fig. 2 | `Figure_2.ipynb` |
| Fig. 3a–c | `Figure_3a-c.ipynb` |
| Fig. 3d–g (trajectories) | `Figure_3d-g.ipynb` |
| Fig. 3d–g (phylogenies) | `Figure_3d_g_trees.ipynb` |
| Fig. 4 | `Figure_4.ipynb` |
| Fig. 5b–e | `Figure_5b_e.ipynb` |
| Fig. 5f | `Figure_5f.ipynb` |
| Extended Data Fig. 1 | `Extended_Data_Figure_1.ipynb` |
| Extended Data Fig. 2 | `Extended_Data_Figure_2.ipynb` |
| Supplementary Fig. 1 | `Supplementary_Fig_1.ipynb` |
| Supplementary Figs. 2–5 | `Supplementary_Fig_2-5.ipynb` |
| Supplementary Fig. 6 | `Supplementary_Fig_6.ipynb` |
| Supplementary Fig. 7 | `Supplementary_Fig_7.ipynb` |
| Supplementary Fig. 8 | `Supplementary_Fig_8.ipynb`, `Supplementary_Fig_8c.ipynb` |
| Supplementary Fig. 9 | `Supplementary_Fig_9.ipynb` |
| Supplementary Fig. 10 | `Supplementary_Fig_10.ipynb` |
| Supplementary Fig. 11 | `Supplementary_Fig_11.ipynb`, `Supplementary_Fig_11_simulating_beta_binomial_dataset.ipynb` |
| Supplementary Fig. 12 | `Supplementary_Fig_12.ipynb` |
| Supplementary Fig. 13 | `Supplementary_Fig_13.ipynb` |
| Supplementary Fig. 14 | `Supplementary_Fig_14.ipynb` |
| Supplementary Fig. 15 | `Supplementary_Fig_15.ipynb` |
| Supplementary Fig. 22 | `Supplementary_Fig_22.ipynb` (rearrangement caller) |
| Supplementary Fig. 23 | `Supplementary_Fig_23.ipynb` |
| Supplementary Fig. 24 | `Supplementary_Fig_24.ipynb` |
| Supplementary Fig. 25 | `Supplementary_Fig_25.ipynb` |
| Supplementary Figs. 26–30 | `Supplementary_Fig_26-30_Simulating_mCA_samples.ipynb`, `Supplementary_Fig_26.ipynb`, `Supplementary_Fig_26_mCA_unphased_caller_on_simulated_samples.py`, `Supplementary_Fig_27.ipynb`, `Supplementary_Fig_28-29-mCA_phased_caller_benchmarking_on_simulated_samples.ipynb`, `Supplementary_Fig_28.ipynb`, `Supplementary_Fig_29.ipynb`, `Supplementary_Fig_30.ipynb` |
| Supplementary Figs. 31–38 | `Supplementary_Fig_31-38-mCA_calling_unphased.ipynb`, `Supplementary_Fig_31-38-mCA_calling_phased_timepoints.ipynb` |
| Supplementary Fig. 46 | `Supplementary_Fig_46.ipynb` |

## Data availability

`Data_files/` holds everything needed to reproduce the analyses that do not rest on individual-level
data: the panel design, the aggregated variant calls with ages given as completed units (rounded
down), the inferred fitness and establishment ages, the clonal structures, the simulated datasets and
the published CH / COSMIC reference data.

Not in this repository:

| data | where |
|---|---|
| raw and duplex-consensus sequencing data; per-sample VCFs, annotated call files and BAF/LRR files | European Genome-phenome Archive (controlled access, via a Data Access Committee administered by the corresponding authors) |
| participant-level sample information with exact ages and sample timings | as above |
| somatic VCFs | Zenodo, DOI 10.5281/zenodo.22262496 |
| hg19 reference genome, ANNOVAR `humandb/`, fgbio | third parties; set the path in the notebook that uses them |

Each notebook's own Data availability table says which of these it needs, and several fall back to a
summary file so the figure can still be drawn without the controlled-access data.

## Licence and citation

The code in this repository is released under the **GNU General Public License v3.0**
(see [LICENSE](LICENSE)), as is the upstream
[TETRIS-seq](https://github.com/the-blundell-lab/TETRIS-seq) pipeline. The data files under
`Data_files/` are provided for reproducing the analyses presented in the manuscript.

If you use this code, please cite the manuscript.
