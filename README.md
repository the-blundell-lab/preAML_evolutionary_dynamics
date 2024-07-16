# Evolutionary dynamics in the decade preceding acute myeloid leukaemia
This repository contains the code and necessary files to accompany the manuscript **"Evolutionary dynamics in the decade preceding acute myeloid leukaemia"**. Code was written using Python. 

TETRIS-seq code, for analysing the raw sequencing data and processing the duplex output files with the TETRIS-seq _in silico_ noise correction model can be found here: https://github.com/the-blundell-lab/TETRIS-seq

If there are problems with the code (.ipynb) pages rendering, please try viewing them on [jupyter nbviewer](https://nbviewer.org/github/the-blundell-lab/preAML_evolutionary_dynamics/tree/main/).

## Navigation:
### Code & data for generation of figures/ analyses in main text:
#### Figure 1c (Deep sequencing of serial blood samples in the decades preceding AML):
- Github file: _Figure 1 - All trajectories and burden of mutations.ipynb_

#### Figure 2 (Reconstruction of clonal evolutionary histories pre-AML):
- Github file: _Figure 2 - Supplement - Muller_plots.ipynb_ 
    
#### Figure 3 (Quantitative dynamics of driver mutations in the decades before AML diagnosis):
- **Figures 3a-c**:
    - Github file: _Figure 3 - Schematic for clonal behaviour.ipynb_
- **Figures 3d-g**:
    - Github file: _Code_for_inferring_acquisition_age_and_fitness_v14.py_
          - (N.B. Code requires VCF files for each sample, containing total depth information at each position).
    - Github file: _Figure 3 - Extended Data Figs - Supplement - Plotting cell fraction trajectories from s and t inferences.ipynb_
    - Github file: _Figure 3 - Extended Data Figs - Supplement - Phylogeny plots.ipynb_ 
    
#### Figure 4 (Fitness and occurrence time estimates of pre-leukaemic driver events):
- Github file: _Figures 4 and 5 - Acquisition_age_fitness_summary_plot.ipynb_
  
#### Figure 5 (Fitness effects of driver mutations in pre-AML cases):
- Github file: _Figures 4 and 5 - Acquisition_age_fitness_summary_plot.ipynb_
  
#### Figure 6 (A unifying framework for pre-leukaemic clonal dynamics):
- **Figure 2b-e**:
    - Github folder: _Clonal dynamic simulations/UKCTOCS_Figure_5_simulations.ipynb_
- **Figure 2f**:
    - Github folder: _UK Biobank hotspots_

 
### Code & data for generation of Extended Data figures:
#### Extended Data Figure 1a (Longitudinal blood samples pre-AML diagnosis):
- Github file: _Extended Data Fig - UKCTOCS samples.ipynb_
 
#### Extended Data Figure 2 (Classes of mutations detected in pre-AML and control samples):
- Github file: _Extended Data Fig - Classes of mutations detected.ipynb_

#### Extended Data Figures 3-9 (Quantitative dynamics of driver mutations in the decades before AML diagnosis):
- Github file: _Code_for_inferring_acquisition_age_and_fitness_v14.py_
          - (N.B. Code requires VCF files for each sample, containing total depth information at each position).
- Github file: _Figure 3 - Extended Data Figs - Supplement - Plotting cell fraction trajectories from s and t inferences.ipynb_
- Github file: _Figure 3 - Extended Data Figs - Supplement - Phylogeny plots.ipynb_ 
 
#### Extended Data Figure 10 (Simulated pre-AML cases and controls):
- Github folder: _Clonal dynamic simulations/UKCTOCS_Figure_5_simulations.ipynb_


### Code & data for generation of figures/ analyses in supplement:
#### Supplementary Figures 1-4 (Gene regions targeted by TETRIS-seq SNV/ indel panel):
- Github file: _TETRIS-seq targeted regions/Gene regions targeted by TETRIS-seq SNV_indel panel.ipynb_
    
#### Supplementary Figure 5 (Custom panel coverage of chromosomal rearrangement breakpoint regions):
- Github file: _TETRIS-seq targeted regions/Chromosomal rearrangement breakpoint regions targeted by TETRIS-seq.ipynb_

#### Supplementary Figure 12 (LRR and BAF deviations for mCA detection (schematic):
- Github file: _TETRIS-seq performance/Supplement - mCA simulations.ipynb_

#### Supplementary Figure 13 (Phasing SNPs for detection of low cell fraction mCAs (schematic):
- Github file: _TETRIS-seq performance/Supplement - mCA simulations.ipynb_

#### Supplementary Figure 14 (Schematic showing the effect of KMT2A-PTD on exon 3: exon 27 read depth ratios (schematic):
- Github file: _TETRIS-seq performance/Supplement - KMT2A PTD simulation.ipynb_

#### Supplementary Figures 15-21 (Reconstruction of clonal evolutionary histories):
- Github file: _Figure 2 - Supplement - Muller_plots.ipynb_ 

#### Supplementary Figures 22-30 (Quantitative dynamics of driver mutations in the decades before AML diagnosis):
- Github file: _Code_for_inferring_acquisition_age_and_fitness_v14.py_
     - (N.B. Code requires VCF files for each sample, containing total depth information at each position).
- Github file: _Figure 3 - Extended Data Figs - Supplement - Plotting cell fraction trajectories from s and t inferences.ipynb_
- Github file: _Figure 3 - Extended Data Figs - Supplement - Phylogeny plots.ipynb_ 
  
## LICENSE
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)
