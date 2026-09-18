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
3. **`Supplementary_Fig_47.ipynb`** — summarises the seed-to-seed spread into the uncertainty
   estimates (standard deviation and 95% optimiser range) reported in Supplementary Table 9.
4. **The figure notebooks** — each reads those outputs and reproduces one or more manuscript figures.

## Figures

| figure | what it shows | notebook |
|---|---|---|
| Fig. 1 | Deep sequencing of serial blood samples in the decades preceding AML | `Figure_1.ipynb` |
| Fig. 2 | Reconstruction of clonal evolutionary histories in AML | `Figure_2.ipynb` |
| Fig. 3a–c | Quantitative dynamics of driver mutations in the decades before AML diagnosis (schematic) | `Figure_3a-c.ipynb` |
| Fig. 3d–g | Measured and inferred cell-fraction trajectories | `Figure_3d-g.ipynb` |
| Fig. 3d–g | The inferred clonal phylogenies beside them | `Figure_3d_g_trees.ipynb` |
| Fig. 4 | Fitness and occurrence time estimates of pre-leukaemic driver events | `Figure_4.ipynb` |
| Fig. 5b–e | A unifying framework for pre-leukaemic clonal dynamics (simulated panels) | `Figure_5b_e.ipynb` |
| Fig. 5f | The same plot for CH hotspot variants in UK Biobank | `Figure_5f.ipynb` |
| Extended Data Fig. 1 | Longitudinal blood samples pre-AML diagnosis | `Extended_Data_Figure_1.ipynb` |
| Extended Data Fig. 2 | Classes of mutations detected in pre-AML and control samples | `Extended_Data_Figure_2.ipynb` |
| Supp. Fig. 1 | UKCTOCS cell type deconvolution | `Supplementary_Fig_1.ipynb` |
| Supp. Figs. 2–5 | Gene regions targeted by the TETRIS-seq SNV/indel panel | `Supplementary_Fig_2-5.ipynb` |
| Supp. Fig. 6 | Custom panel coverage of chromosomal rearrangement breakpoint regions | `Supplementary_Fig_6.ipynb` |
| Supp. Fig. 7 | Error-corrected sequencing metrics | `Supplementary_Fig_7.ipynb` |
| Supp. Fig. 8 | DCS variant calling using Myeloid Reference Standard DNA | `Supplementary_Fig_8.ipynb` |
| Supp. Fig. 8c | Mutational signatures of those reference-standard calls | `Supplementary_Fig_8c.ipynb` |
| Supp. Fig. 9 | Expected and observed number of DCS variants | `Supplementary_Fig_9.ipynb` |
| Supp. Fig. 10 | Distribution of sample DCS `VAFs' (error rates) at example positions | `Supplementary_Fig_10.ipynb` |
| Supp. Fig. 11 | Choosing a p-value threshold for calling real variants | `Supplementary_Fig_11.ipynb` |
| Supp. Fig. 11 | Simulating the beta-binomial dataset used to choose it | `Supplementary_Fig_11_simulating_beta_binomial_dataset.ipynb` |
| Supp. Fig. 12 | Position-specific distributions of errors | `Supplementary_Fig_12.ipynb` |
| Supp. Fig. 13 | Distribution of final position-specific error rates, grouped by base change | `Supplementary_Fig_13.ipynb` |
| Supp. Fig. 14 | Observed vs expected VAF across Horizon Myeloid Reference Standard dilutions | `Supplementary_Fig_14.ipynb` |
| Supp. Fig. 15 | Effect of the in silico noise correction method on the number of variant calls | `Supplementary_Fig_15.ipynb` |
| Supp. Fig. 22 | VAF concordance in simulated samples (and the rearrangement caller itself) | `Supplementary_Fig_22.ipynb` |
| Supp. Fig. 23 | Detection of known AML-associated rearrangements across serial dilutions | `Supplementary_Fig_23.ipynb` |
| Supp. Fig. 24 | Effect of KMT2A-PTD on exon 3 : exon 27 read depth ratios | `Supplementary_Fig_24.ipynb` |
| Supp. Fig. 25 | LRR and BAF deviations for mCA detection | `Supplementary_Fig_25.ipynb` |
| Supp. Figs. 26–30 | Generating the simulated mCA samples | `Supplementary_Fig_26-30_Simulating_mCA_samples.ipynb` |
| Supp. Fig. 26 | Performance of the unphased mCA caller on simulated test data | `Supplementary_Fig_26.ipynb` |
| Supp. Fig. 26 | The unphased mCA caller, run over those samples | `Supplementary_Fig_26_mCA_unphased_caller_on_simulated_samples.py` |
| Supp. Fig. 27 | Phasing SNPs for detection of low cell fraction mCAs | `Supplementary_Fig_27.ipynb` |
| Supp. Figs. 28–29 | Benchmarking the longitudinal phased mCA caller | `Supplementary_Fig_28-29-mCA_phased_caller_benchmarking_on_simulated_samples.ipynb` |
| Supp. Fig. 28 | Performance of the longitudinal phased mCA caller on simulated test data | `Supplementary_Fig_28.ipynb` |
| Supp. Fig. 29 | False positive rate of the longitudinal phased mCA caller | `Supplementary_Fig_29.ipynb` |
| Supp. Fig. 30 | Selection of control samples for mCA simulation | `Supplementary_Fig_30.ipynb` |
| Supp. Figs. 31–38 | Longitudinal mCA detection: calls in the index samples | `Supplementary_Fig_31-38-mCA_calling_unphased.ipynb` |
| Supp. Figs. 31–38 | Longitudinal mCA detection: phased calls at earlier timepoints | `Supplementary_Fig_31-38-mCA_calling_phased_timepoints.ipynb` |
| Supp. Fig. 47 | Quantification of uncertainty for fitness and establishment time estimates | `Supplementary_Fig_47.ipynb` |

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
