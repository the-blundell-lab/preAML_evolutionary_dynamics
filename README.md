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
| Fig.&nbsp;1 | Deep sequencing of serial blood samples in the decades preceding AML | [`Figure_1.ipynb`](Figure_1.ipynb) |
| Fig.&nbsp;2<br>Supp.&nbsp;Figs.&nbsp;39–46, 57–60 | Reconstruction of clonal evolutionary histories in AML | [`Figure_2.ipynb`](Figure_2.ipynb) |
| Fig.&nbsp;3a–c | Quantitative dynamics of driver mutations in the decades before AML diagnosis (schematic) | [`Figure_3a-c.ipynb`](Figure_3a-c.ipynb) |
| Fig.&nbsp;3d–g<br>ED&nbsp;Figs.&nbsp;3–9<br>Supp.&nbsp;Figs.&nbsp;48–55 | Measured and inferred cell-fraction trajectories | [`Figure_3d-g.ipynb`](Figure_3d-g.ipynb) |
| Fig.&nbsp;3d–g<br>ED&nbsp;Figs.&nbsp;3–9<br>Supp.&nbsp;Figs.&nbsp;48–55 | The inferred clonal phylogenies beside them | [`Figure_3d_g_trees.ipynb`](Figure_3d_g_trees.ipynb) |
| Fig.&nbsp;4<br>Supp.&nbsp;Fig.&nbsp;56 | Fitness and occurrence time estimates of pre-leukaemic driver events | [`Figure_4.ipynb`](Figure_4.ipynb) |
| Fig.&nbsp;5b–e | A unifying framework for pre-leukaemic clonal dynamics (simulated panels) | [`Figure_5b_e.ipynb`](Figure_5b_e.ipynb) |
| Fig.&nbsp;5f | The same plot for CH hotspot variants in UK Biobank | [`Figure_5f.ipynb`](Figure_5f.ipynb) |
| Extended&nbsp;Data Fig.&nbsp;1 | Longitudinal blood samples pre-AML diagnosis | [`Extended_Data_Figure_1.ipynb`](Extended_Data_Figure_1.ipynb) |
| Extended&nbsp;Data Fig.&nbsp;2 | Classes of mutations detected in pre-AML and control samples | [`Extended_Data_Figure_2.ipynb`](Extended_Data_Figure_2.ipynb) |
| Supp.&nbsp;Fig.&nbsp;1 | UKCTOCS cell type deconvolution | [`Supplementary_Fig_1.ipynb`](Supplementary_Fig_1.ipynb) |
| Supp.&nbsp;Fig.&nbsp;2–5 | Gene regions targeted by the TETRIS-seq SNV/indel panel | [`Supplementary_Fig_2-5.ipynb`](Supplementary_Fig_2-5.ipynb) |
| Supp.&nbsp;Fig.&nbsp;6 | Custom panel coverage of chromosomal rearrangement breakpoint regions | [`Supplementary_Fig_6.ipynb`](Supplementary_Fig_6.ipynb) |
| Supp.&nbsp;Fig.&nbsp;7 | Error-corrected sequencing metrics | [`Supplementary_Fig_7.ipynb`](Supplementary_Fig_7.ipynb) |
| Supp.&nbsp;Fig.&nbsp;8 | DCS variant calling using Myeloid Reference Standard DNA | [`Supplementary_Fig_8.ipynb`](Supplementary_Fig_8.ipynb) |
| Supp.&nbsp;Fig.&nbsp;8c | Mutational signatures of those reference-standard calls | [`Supplementary_Fig_8c.ipynb`](Supplementary_Fig_8c.ipynb) |
| Supp.&nbsp;Fig.&nbsp;9 | Expected and observed number of DCS variants | [`Supplementary_Fig_9.ipynb`](Supplementary_Fig_9.ipynb) |
| Supp.&nbsp;Fig.&nbsp;10 | Distribution of sample DCS `VAFs' (error rates) at example positions | [`Supplementary_Fig_10.ipynb`](Supplementary_Fig_10.ipynb) |
| Supp.&nbsp;Fig.&nbsp;11 | Choosing a p-value threshold for calling real variants | [`Supplementary_Fig_11.ipynb`](Supplementary_Fig_11.ipynb) |
| Supp.&nbsp;Fig.&nbsp;11 | Simulating the beta-binomial dataset used to choose it | [`Supp_Fig_11…beta_binomial.ipynb`](Supplementary_Fig_11_simulating_beta_binomial_dataset.ipynb) |
| Supp.&nbsp;Fig.&nbsp;12 | Position-specific distributions of errors | [`Supplementary_Fig_12.ipynb`](Supplementary_Fig_12.ipynb) |
| Supp.&nbsp;Fig.&nbsp;13 | Distribution of final position-specific error rates, grouped by base change | [`Supplementary_Fig_13.ipynb`](Supplementary_Fig_13.ipynb) |
| Supp.&nbsp;Fig.&nbsp;14 | Observed vs expected VAF across Horizon Myeloid Reference Standard dilutions | [`Supplementary_Fig_14.ipynb`](Supplementary_Fig_14.ipynb) |
| Supp.&nbsp;Fig.&nbsp;15 | Effect of the in silico noise correction method on the number of variant calls | [`Supplementary_Fig_15.ipynb`](Supplementary_Fig_15.ipynb) |
| Supp.&nbsp;Fig.&nbsp;22 | VAF concordance in simulated samples (and the rearrangement caller itself) | [`Supplementary_Fig_22.ipynb`](Supplementary_Fig_22.ipynb) |
| Supp.&nbsp;Fig.&nbsp;23 | Detection of known AML-associated rearrangements across serial dilutions | [`Supplementary_Fig_23.ipynb`](Supplementary_Fig_23.ipynb) |
| Supp.&nbsp;Fig.&nbsp;24 | Effect of KMT2A-PTD on exon 3 : exon 27 read depth ratios | [`Supplementary_Fig_24.ipynb`](Supplementary_Fig_24.ipynb) |
| Supp.&nbsp;Fig.&nbsp;25 | LRR and BAF deviations for mCA detection | [`Supplementary_Fig_25.ipynb`](Supplementary_Fig_25.ipynb) |
| Supp.&nbsp;Fig.&nbsp;26–30 | Generating the simulated mCA samples | [`Supp_Fig_26-30_Simulating_mCA.ipynb`](Supplementary_Fig_26-30_Simulating_mCA_samples.ipynb) |
| Supp.&nbsp;Fig.&nbsp;26 | Performance of the unphased mCA caller on simulated test data | [`Supplementary_Fig_26.ipynb`](Supplementary_Fig_26.ipynb) |
| Supp.&nbsp;Fig.&nbsp;26 | The unphased mCA caller, run over those samples | [`Supp_Fig_26…unphased_caller.py`](Supplementary_Fig_26_mCA_unphased_caller_on_simulated_samples.py) |
| Supp.&nbsp;Fig.&nbsp;27 | Phasing SNPs for detection of low cell fraction mCAs | [`Supplementary_Fig_27.ipynb`](Supplementary_Fig_27.ipynb) |
| Supp.&nbsp;Fig.&nbsp;28–29 | Benchmarking the longitudinal phased mCA caller | [`Supp_Fig_28-29…benchmarking.ipynb`](Supplementary_Fig_28-29-mCA_phased_caller_benchmarking_on_simulated_samples.ipynb) |
| Supp.&nbsp;Fig.&nbsp;28 | Performance of the longitudinal phased mCA caller on simulated test data | [`Supplementary_Fig_28.ipynb`](Supplementary_Fig_28.ipynb) |
| Supp.&nbsp;Fig.&nbsp;29 | False positive rate of the longitudinal phased mCA caller | [`Supplementary_Fig_29.ipynb`](Supplementary_Fig_29.ipynb) |
| Supp.&nbsp;Fig.&nbsp;30 | Selection of control samples for mCA simulation | [`Supplementary_Fig_30.ipynb`](Supplementary_Fig_30.ipynb) |
| Supp.&nbsp;Fig.&nbsp;31–38 | Longitudinal mCA detection: calls in the index samples | [`Supp_Fig_31-38…unphased.ipynb`](Supplementary_Fig_31-38-mCA_calling_unphased.ipynb) |
| Supp.&nbsp;Fig.&nbsp;31–38 | Longitudinal mCA detection: phased calls at earlier timepoints | [`Supp_Fig_31-38…phased_timepoints.ipynb`](Supplementary_Fig_31-38-mCA_calling_phased_timepoints.ipynb) |
| Supp.&nbsp;Fig.&nbsp;47 | Quantification of uncertainty for fitness and establishment time estimates | [`Supplementary_Fig_47.ipynb`](Supplementary_Fig_47.ipynb) |

## Data availability

`Data_files/` holds everything needed to reproduce the analyses that do not rest on individual-level
data: the panel design, the aggregated variant calls with ages given as completed units (rounded
down), the inferred fitness and establishment ages, the clonal structures, the simulated datasets and
the published CH / COSMIC reference data.

Not in this repository:

| data | where |
|---|---|
| <ul><li>raw sequencing reads</li><li>all-positions VCFs for single-strand (SSCS) and duplex (DCS) consensus reads</li><li>pre-error-model annotated call files (DCS)</li><li>germline SNV and indel calls</li><li>the combined somatic SNV, indel and mCA call table (the exact-age version of the rounded table provided here)</li><li>longitudinal mCA calls</li><li>per-sample BAF and LRR files</li><li>participant sample and clinical annotation files with exact ages</li></ul> | European Genome-phenome Archive (controlled access, via a Data Access Committee administered by the corresponding authors) |
| The final filtered somatic variant calls (hg19):<ul><li>965 SNVs and indels as per-sample VCF records across 385 files</li><li>the <em>FLT3</em>-ITD as the Pindel caller's own output</li></ul>No ages or clinical data; the same calls with annotation and rounded ages are Supplementary Table 6 | Zenodo, DOI 10.5281/zenodo.22262496 |
| <ul><li>hg19 reference genome</li><li>ANNOVAR <code>humandb/</code></li><li>fgbio</li></ul> | third parties; set the path in the notebook that uses them |

Each notebook's own Data availability table says which of these it needs, and several fall back to a
summary file so the figure can still be drawn without the controlled-access data.

## Licence and citation

The code in this repository is released under the **GNU General Public License v3.0**
(see [LICENSE](LICENSE)), as is the upstream
[TETRIS-seq](https://github.com/the-blundell-lab/TETRIS-seq) pipeline. The data files under
`Data_files/` are provided for reproducing the analyses presented in the manuscript.

If you use this code, please cite the manuscript.
