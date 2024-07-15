#!/usr/bin/env python

'''''
Watson code for inferring acquisition age and fitness effect of variants
Version 14.0 (June 2024)

Input:
    1) Sample name (e.g. 'C92_002')
    2) Seeds (number of times to run the optimiser)

    Also requires the following files in the Data_files folder:
    - UKCTOCS_watson_all_germline_variants_calls_April_2024.csv
    - UKCTOCS_watson_non-germline_variants_calls_SNVs_indels_mCAs_April_2024.csv
    - Clonal_structure_for_each_sample.csv (inferred using hierarchical method as described in manuscript)
    - UKCTOCS_samples_processed_information.csv
    - DCS +/- SSCS VCF files (to get total depth at any position on panel to calculate limit of detection for variants not detected at certain timepoints)

Outputs:
    1) PDF plot of each of the initial crude gradient fits to the trajectories (to estimate initial fitness and establishment time)
    2) PDF plot of the likelihood results of each seed
    3) PDF plot of the optimiser results (fitness vs establishment time) for each seed (all seeds on one plot)
    4) PDF plots of the inferred trajectories (version with 'bounded variants' shown and version without) - version in terms of VAF/cell fraction and version just in terms of cell fraction
    5) PNG and PDF image of germline variants table
    6) Combined PDF of all of the above
    7) Text file of optimiser likelihoods, inferred fitnesses and establishment times
    8) CSV file of phylogenies for making Muller plots (without missing drivers)
    9) CSV file of phylogenies for drawing phylogenies (including missing drivers)

Usage:
Code_for_inferring_acquisition_age_and_fitness_v14.py  --sample_name --seeds

'''''
version = '14'

# imported packages
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as ticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.patches import Polygon
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm
import scipy.special
import scipy.integrate as it
from scipy.special import loggamma
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.stats import kde
from scipy.stats import betabinom
from scipy.stats import binom
import copy
import glob, os
import re
from sklearn import datasets, linear_model
import pandas as pd
from decimal import *
from operator import itemgetter
from collections import OrderedDict
import datetime
from datetime import datetime
import timeit
import time
import csv
from pyfaidx import Fasta
from argparse import ArgumentParser
import pysam
import sys
import matplotlib
from array import array
from ast import literal_eval
import dataframe_image as dfi
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter, PdfMerger
from fpdf import FPDF

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

start_time = time.time()

#define the colors from colorbrewer2
grey1 = '#f7f7f7'
grey2 = '#cccccc'
grey3 = '#969696'
grey4 = '#636363'
grey5 = '#252525'

mutation_classes = {'NPM1': 'NPM1',
                   'DNMT3A': 'DNA methylation',
                   'TET2': 'DNA methylation',
                   'IDH1': 'DNA methylation',
                   'IDH2': 'DNA methylation',
                   'ASXL1': 'Chromatin modifiers',
                   'EZH2': 'Chromatin modifiers',
                   'RUNX1': 'Transcription factors',
                   'CEBPA': 'Transcription factors',
                   'GATA2': 'Transcription factors',
                   'BCOR': 'Transcriptional corepressors',
                   'BCORL1': 'Transcriptional corepressors',
                   'TP53': 'Tumour suppressor',
                   'PPM1D': 'Tumour suppressor',
                   'CHEK2': 'Tumour suppressor',
                   'WT1': 'Tumour suppressor',
                   'CBL': 'Tumour suppressor',
                   'DDX41': 'Tumour suppressor',
                   'SRSF2': 'Spliceosome',
                   'SF3B1': 'Spliceosome',
                   'U2AF1': 'Spliceosome',
                   'ZRSR2': 'Spliceosome',
                   'RAD21': 'Cohesin',
                   'STAG2': 'Cohesin',
                   'FLT3': 'Cell signalling',
                   'KIT': 'Cell signalling',
                   'JAK2': 'Cell signalling',
                   'KRAS': 'Cell signalling',
                   'NRAS': 'Cell signalling',
                   'PTPN11': 'Cell signalling',
                   'CSF3R': 'Cell signalling',
                   'GNB1': 'Cell signalling',
                   'GNAS': 'Cell signalling',
                   'MPL': 'Cell signalling',
                    'mCA': 'mCA',
                   '15q CNLOH': 'mCA',
                   '4q CNLOH': 'mCA',
                   '4 gain?': 'mCA',
                   'X gain': 'mCA',
                    'chrX': 'mCA',
                    'chr19p': 'mCA',
                    'chr9p': 'mCA',
                    'chr7q': 'mCA',
                    'chr4': 'mCA',
                    'chr4q': 'mCA',
                    'chr15q': 'mCA',
                   '19p CNLOH': 'mCA',
                   '9p CNLOH': 'mCA',
                   '7q CNLOH': 'mCA',
                   '15q CNLOH': 'mCA',
                   'X': 'missing driver',
                   'Z': 'missing driver 2'}

mutation_number_colors = {1: ['#2171b5', '#6baed6', '#9ecae1', '#deebf7', '#08306b'],
                          2: ['#41ab5d', '#a1d99b', '#c7e9c0', '#e5f5e0', '#238b45'],
                          3: ['#fd8d3c', '#feb24c', '#fed976', '#ffeda0', '#ffffcc'],
                          4: ['#cb181d', '#ef3b2c', '#fb6a4a', '#fc9272'],
                          5: ['#980043', '#7a0177'],
                         6: ['#67001f']}

multiple_mutant_clone_colors = {1: {1: ['#6ba3d6'],
                                   2: ['#2171b5', '#9ecae1'],
                                   3: ['#08306b', '#2171b5', '#9ecae1'],
                                   4: ['#08306b', '#2171b5', '#9ecae1', '#deebf7'],
                                   5: ['#08306b', '#08519c', '#4292c6', '#9ecae1', '#deebf7']},
                               2: {1: ['#41ab5d'],
                                  2: ['#006d2c', '#74c476'],
                                  3: ['#00441b', '#238b45', '#a1d99b'],
                                  4: ['#00441b', '#238b45', '#a1d99b', '#e5f5e0']},
                               3: {1: ['#fd8d3c'],
                                  2: ['#ec7014', '#fed976'],
                                  3: ['#cc4c02', '#fe9929', '#fee391'],
                                  4: ['#993404', '#ec7014', '#fec44f', '#ffeda0'],
                                  5: ['#662506', '#cc4c02', '#fe9929', '#fed976', '#fff7bc'],
                                  6: ['#662506', '#993404', '#cc4c02', '#fe9929', '#fed976', '#fff7bc']},
                               4: {1: ['#da2b2b'],
                                  2: ['#bc2221', '#f0a1a1'],
                                  3: ['#bc2221', '#e9807f', '#fad4d4'],
                                  4: ['#9b191c', '#da2b2b', '#f0a1a1', '#fad4d4'],
                                  5: ['#791614', '#b9181f', '#eb5770', '#f3a0a3', '#fad4d4']},
                               5: {1: ['#834594'],
                                  2: ['#834594', '#ba9bc9']},
                               6: {1: ['#4e4d4d']}}


def dictionary_sample_details(sample_ID):


    sample_ages = {} #e.g. {'C92_002_s1': 73.73, 'C92_002_s2': 75.0, 'C92_002_s3': 75.83...}
    sample_ages_dict = {} #e.g. C92_002 = [73.73, 75.0, 75.83...]
    sample_ages_dict_names = {} #e.g. {C92_002: {73.73: C92_002_s1, 75.0: C92_002_s2...}}
    sample_diagnosis_age = {} #e.g. {'C92_002': 81.1, 'C92_003': 75.21, 'C92_005': 70.09...}
    # sample_DNA_amount = {} #e.g. {'C92_002_s1': '45', 'C92_002_s2': '50', 'C92_002_s3': '50'...}
    matched_sample = {} #e.g. {'C92_002': 'CNTRL_169', 'C92_003': 'CNTRL_002'...}

    with open('Data_files/UKCTOCS_samples_processed_information.csv') as csvfile:
        readreader = csv.reader(csvfile)
        row_count=0
        for row in readreader:
            if row_count>0:
                sample_name = row[1].split('_')[0]+'_'+row[1].split('_')[1]
                timepoint = row[1]
                sample_ages[timepoint]=float(row[6]) #dictionary of e.g. C92_007_s2 = 76

                if sample_name in sample_ages_dict.keys():
                    sample_ages_dict[sample_name].append(float(row[6]))
                else:
                    sample_ages_dict[sample_name]=[float(row[6])]

                if sample_name in sample_ages_dict_names.keys():
                    sample_ages_dict_names[sample_name][float(row[6])]=timepoint
                else:
                    sample_ages_dict_names[sample_name]={float(row[6]): timepoint}

                if row[7]!='':
                    if '_' in row[7]:
                        matched_sample_name = row[7].split('_')[0]+'_'+row[7].split('_')[1]
                        matched_sample[sample_name]=matched_sample_name

                if row[0]=='Case':
                    sample_diagnosis_age[sample_name]=float(row[5])

                if row[0]=='Control':
                    if sample_name in matched_sample.keys():
                        sample_diagnosis_age[sample_name]=sample_diagnosis_age[matched_sample[sample_name]]

            row_count+=1

    sample_ages_lists = {}
    for k, v in sample_ages_dict.items():
        sample_ages_lists[k] = sorted(v)

    diagnosis_age = sample_diagnosis_age[sample_ID]
    ages_dict_names = sample_ages_dict_names[sample_ID]

    return sample_ages, ages_dict_names, sample_ages_lists, diagnosis_age

def sample_trajectories(person_ID, df): #extract trajectories of each variant for a given sample, e.g. 'C92_002'

    filtered_df = df[df['sample name'].str.startswith(person_ID)]

#     person_variant_dict={}
    var_dict={}
    variant_details = {}
#     person_variant_dict[person_ID]=var_dict

    gene_variant = ''

    for index, row in filtered_df.iterrows():
        # Access data for each column by column name
        variant_type=row['variant_type']
#         if row['VAF (cell fraction for mCAs)'] != 'not processed':
        if variant_type=='SNV':
            variant=row['AA_change']
            gene=row['gene']
            VAF=float(row['VAF (cell fraction for mCAs)'])
            age=row['age_sample_taken']
            gene_variant=gene+'_'+variant
        elif (variant_type=='CNLOH' or variant_type=='gain' or variant_type=='gain?' or variant_type=='loss'):
            VAF=row['VAF (cell fraction for mCAs)']
            age=row['age_sample_taken']
            chromosome=row['chromosome']
            gene_variant=chromosome+'_'+variant_type
        elif (variant_type=='Deletion' or variant_type=='Insertion' or variant_type=='Complex'):
            VAF=float(row['VAF (cell fraction for mCAs)'])
            variant=row['AA_change']
            age=row['age_sample_taken']
            gene=row['gene']
            gene_variant=gene+'_'+variant
        elif (variant_type=='ITD'):
            VAF=float(row['VAF (cell fraction for mCAs)'])
            variant=row['AA_change']
            age=row['age_sample_taken']
            gene=row['gene']
            gene_variant=gene+'_'+variant

        if gene_variant in var_dict.keys():
            if VAF != 'not processed':
                var_dict[gene_variant][age]=float(VAF)
        else:
            if VAF != 'not processed':
                var_dict[gene_variant]={age: float(VAF)}

        if gene_variant != '':
            if variant_type not in ('CNLOH', 'gain', 'gain?', 'loss'):
                chromosome = row['chromosome']
                start = int(row['start'])
                ref = row['REF']
                variant_details[gene_variant]=(chromosome, start, ref)
            else:
                variant_details[gene_variant]=(chromosome, '', '')

    return var_dict, variant_details

def germline_variants(person_ID, df_germline): #extract trajectories of each variant for a given sample, e.g. 'C92_002'

    filtered_df_germline = df_germline[df_germline['sample name'].str.startswith(person_ID)]

    germline_dict={}
    germline_list = []
    significant_germline = []
    significant_germline_with_hom_het = []
    all_germline = {}

    for index, row in filtered_df_germline.iterrows():
        # Access data for each column by column name
        variant_type=row['variant_type']
        variant=row['AA_change']
        gene=row['gene']
        VAF=float(row['VAF (cell fraction for mCAs)'])
        age=row['age_sample_taken']
        gene_variant=gene+'_'+variant
        exac = row['exac_all']
        clin_sig = row['clin_sig']
        clinvar_dn = row['clinvar_dn']
        cosmic_haem = row['cosmic_haem_lymphoid']
        cosmic_total = row['cosmic_total']

        if gene_variant in germline_dict.keys():
            germline_dict[gene_variant][age]=float(VAF)
        else:
            germline_dict[gene_variant]={age: float(VAF)}

        if exac == '.':
            if gene_variant not in significant_germline:
                significant_germline.append(gene_variant)
        else:
            if float(exac)<0.01:
                if clin_sig != 'Benign':
                    if gene_variant not in significant_germline:
                        significant_germline.append(gene_variant)

        all_germline[gene_variant]={'exac': exac, 'clin_sig': clin_sig, 'clinvar_dn': clinvar_dn, 'cosmic haem': cosmic_haem, 'cosmic total': cosmic_total}

    for k, v in germline_dict.items():
        all_VAFs = []
        for age, VAF in v.items():
            all_VAFs.append(VAF)
        if np.mean(VAF)>0.6:
            het_homo = '(1/1)'
        else:
            het_homo = '(0/1)'

        variant_summary = k.split('_')[0]+' '+k.split('_')[1]+' '+het_homo
        germline_list.append(variant_summary)
        if k in significant_germline:
            significant_germline_with_hom_het.append(variant_summary)

    germline_df_summary = pd.DataFrame.from_dict(all_germline, orient = 'index')

    germline_df_summary = germline_df_summary.sort_values(by=['exac'], ascending = False)

    return germline_dict, sorted(germline_list), sorted(significant_germline), germline_df_summary

#Inferred clonal structure to use in MLE
def convert_to_tuple(string_of_variants):
    tuple_variants = tuple(string_of_variants.split(', '))
    return tuple_variants

def inferred_clonal_structure(sample_name):
    df_clonal = pd.read_csv('Data_files/Clonal_structure_for_each_sample.csv')
    df_clonal_sample = df_clonal[df_clonal['Sample name']==sample_name]
    df_clonal_sample['Variants in clone']=df_clonal_sample['Variants in clone'].apply(convert_to_tuple)
    df_clonal_dict = pd.DataFrame.to_dict(df_clonal_sample, orient = 'index')

    sample_clonal_structure = {}
    manual_initial_guesses = {}
    coarse_optimisation_attempts = 0
    upper_freq = 1.0

    for k, v in df_clonal_dict.items():
        if v['Sample name'] in sample_clonal_structure.keys():
            sample_clonal_structure[v['Sample name']].append(v['Variants in clone'])
        else:
            sample_clonal_structure[v['Sample name']]=[v['Variants in clone']]

        if np.isnan(v['Fitness_estimate']) == False:
            manual_initial_guesses[v['Variants in clone']]={'fitness': v['Fitness_estimate'], 'est_time': v['Est_time_estimate']}

        if np.isnan(v['Coarse']) == False:
            coarse_optimisation_attempts=int(v['Coarse'])

        if np.isnan(v['Upper_freq']) == False:
            upper_freq=float(v['Upper_freq'])

    return sample_clonal_structure[sample_name], manual_initial_guesses, coarse_optimisation_attempts, upper_freq

#Put a bound on VAF for mutations that dissappear or only appear late
def extract_variant_reads(info):
    variant_reads = int(info.split(';')[4].split('=')[1])
    return variant_reads

def extract_depth(info):
    depth = int(info.split(';')[2].split('=')[1])
    return depth

def bounded_VAF(sample_name, sample_timepoint, chromosome, position, ref, variant): #get the total read depth at a position from the all positions vcf file
    #e.g. sample_name = C92_002, sample_timepoint = C92_002_s7, chromosome = 'chr1', position = 1747196, ref = 'T'
    #retrieve depth
    if variant != 'FLT3_ITD':
        try:
            df= pd.read_csv('Data_files/VCF_files/DCS'+sample_timepoint+'_SNV_SNV_watson_code_DCS_variants_MUFs_3_all_positions.vcf', comment = '#', sep = '\t', header = None, names = ['chromosome', 'position', 'ID', 'REF','ALT', 'FILTER', 'INFO', 'FORMAT', 'SAMPLE'])
        except FileNotFoundError:
            df= pd.read_csv('Data_files/VCF_files/DCS'+sample_timepoint+'_SNV_watson_code_DCS_variants_MUFs_3_all_positions.vcf', comment = '#', sep = '\t', header = None, names = ['chromosome', 'position', 'ID', 'REF','ALT', 'FILTER', 'INFO', 'FORMAT', 'SAMPLE'])
    else: #if FLT3-ITD, look at the SSCS depth...
        try:
            df= pd.read_csv('Data_files/VCF_files/SSCS'+sample_timepoint+'_SNV_SNV_watson_code_SSCS_variants_MUFs_3_all_positions.vcf', comment = '#', sep = '\t', header = None, names = ['chromosome', 'position', 'ID', 'REF','ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'SAMPLE'])
        except FileNotFoundError:
            df= pd.read_csv('Data_files/VCF_files/SSCS'+sample_timepoint+'_SNV_watson_code_SSCS_variants_MUFs_3_all_positions.vcf', comment = '#', sep = '\t', header = None, names = ['chromosome', 'position', 'ID', 'REF','ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'SAMPLE'])
    df = df[(df['chromosome']==chromosome) & (df['position']==position)]
    df['depth'] = df['INFO'].apply(extract_depth)
    total_depth = df['depth'].tolist()[0]
    highest_VAF = 1/total_depth #i.e. upper bound on what the VAF could be

    df['variant reads'] = df['INFO'].apply(extract_variant_reads)
    variant_reads = df['variant reads'].tolist()[0]
    VAF = variant_reads/total_depth

    return highest_VAF, (variant_reads, VAF)

def create_bounded_variant_trajectories(sample_name, sample_variants, sample_ages_lists, ages_dict_names, variant_details): #creates trajectories (with 2 timepoints) for variants with only 1 timepoint (final or first timepoint)

    bounded_sample_variants = {} #create a dictionary of just the variants for which we infer an upper bound on VAF
    all_trajectories_with_bounded = {} #include all variants

    sample_timings = sample_ages_lists[sample_name]

    for variant, timepoints in sample_variants.items():
#         print(variant)
        detected_times = list(timepoints.keys())
        first_detected_timepoint = detected_times[0] #the age at which the variant is first detected
        sample_timing_first_detected = sample_timings.index(first_detected_timepoint) #where in the list of sample timings the variant was first detected (i.e. the index/ position)
        last_detected_timepoint = detected_times[-1] #the age at which the variant is last detected
        sample_timing_last_detected = sample_timings.index(last_detected_timepoint) #where in the list of sample timings the variant was last detected (i.e. the index/ position)

        #if the variant is not detected from the beginning, get the bounded VAF of the previous 2 timepoints
        if sample_timing_first_detected > 0: #i.e. if the age at which it is first detected is not the first age
            print('variant not detected from beginning = ', variant)
            #preceding timpoint....
            sample_ID_first_detected = ages_dict_names[sample_timings[sample_timing_first_detected]] #e.g. C92_002_s8
            last_non_detected_age = sample_timings[sample_timing_first_detected-1] #i.e. the timepoint/age before 1st detected (i.e. last timepoint not detected)
            sample_ID_non_detected = ages_dict_names[last_non_detected_age] #e.g. C92_002_s7
            #get the chromosome, position and ref for the variant...
            chromosome = variant_details[variant][0]
            position = variant_details[variant][1]
            ref = variant_details[variant][2]
            if len(ref)>1:
                ref = ref[0]
            non_detected_bounded_VAF, variant_reads_VAF = bounded_VAF(sample_name, sample_ID_non_detected, chromosome, position, ref, variant)
            variant_reads = variant_reads_VAF[0]
            VAF = variant_reads_VAF[1]

            if variant_reads <=1: #i..e if the variant genuinely not detected
                bounded_sample_variants[variant] = {last_non_detected_age: float(non_detected_bounded_VAF)}

                if sample_timing_first_detected>=2: #i.e. 2 missing previous timepoints -> also had the bound for 2 timepoints before
                    two_before_non_detected_age = sample_timings[sample_timing_first_detected-2] #i.e. two timepoints/ages before 1st detected
                    sample_ID_two_non_detected = ages_dict_names[two_before_non_detected_age] #e.g. C92_002_s7
                    non_detected_bounded_VAF_two_before, variant_reads_VAF_two_before = bounded_VAF(sample_name, sample_ID_two_non_detected, chromosome, position, ref, variant)
                    variant_reads_two_before = variant_reads_VAF_two_before[0]

                    if variant_reads_two_before <=1:
                        bounded_sample_variants[variant][two_before_non_detected_age]= float(non_detected_bounded_VAF_two_before)

                if variant in bounded_sample_variants.keys():
                    for k, v in timepoints.items(): #add the actual detected VAFs to the dictionary as well
                        bounded_sample_variants[variant][k]=float(v)

        #if the variant is initially detected and then disappears
        if last_detected_timepoint < sample_timings[-1]: #i.e. when younger than final timepoint sample
            print('variant initially detected then disappears = ', variant)
            sample_ID_last_detected = ages_dict_names[sample_timings[sample_timing_last_detected]] #e.g. C92_002_s7
            first_non_detected_age = sample_timings[sample_timing_last_detected+1] #i.e. the timepoint/age when first not detected (i.e. timepoint after last detected)
            sample_ID_non_detected = ages_dict_names[first_non_detected_age] #e.g. C92_002_s8
            #get the chromosome, position and ref for the variant...
            chromosome = variant_details[variant][0]
            position = variant_details[variant][1]
            ref = variant_details[variant][2]
            if len(ref)>1:
                ref = ref[0]
            non_detected_bounded_VAF, variant_reads_VAF = bounded_VAF(sample_name, sample_ID_non_detected, chromosome, position, ref, variant)
            variant_reads = variant_reads_VAF [0]

            if variant_reads <=1:
                if variant not in bounded_sample_variants: #i.e. not a variant that is also not detected from beginning:
                    for k, v in timepoints.items(): #add the actual detected VAFs to the dictionary as well
                        if variant not in bounded_sample_variants.keys():
                            bounded_sample_variants[variant]={k: float(v)}
                        else:
                            bounded_sample_variants[variant][k]=float(v)
                    bounded_sample_variants[variant][first_non_detected_age]= float(non_detected_bounded_VAF)
                else: #i.e. missing from beginning and from end
                    bounded_sample_variants[variant][first_non_detected_age]= float(non_detected_bounded_VAF)

                if last_detected_timepoint < sample_timings[-2]: #i.e. more than 2 timepoints missing at the end:
                    second_non_detected_age = sample_timings[sample_timing_last_detected+2] #i.e. the 2nd timepoint/age when first not detected (i.e. timepoint after last detected)
                    sample_ID_second_non_detected = ages_dict_names[second_non_detected_age] #e.g. C92_002_s8
                    second_non_detected_bounded_VAF, variant_reads_VAF_second_non_detected = bounded_VAF(sample_name, sample_ID_second_non_detected, chromosome, position, ref, variant)

                    bounded_sample_variants[variant][second_non_detected_age]= float(second_non_detected_bounded_VAF)

#     print()
#     print('bounded = ', bounded_sample_variants)

    bounded_sample_variants_age_order = {}
    for variant, timepoints in bounded_sample_variants.items():
        sorted_timepoints_dict = {}
        sorted_timepoints = sorted(list(timepoints.keys()))
        for time in sorted_timepoints:
            sorted_timepoints_dict[time]=timepoints[time]
        bounded_sample_variants_age_order[variant]=sorted_timepoints_dict

#     print()
#     print('bounded age order = ', bounded_sample_variants_age_order)

    for variant, timepoints in sample_variants.items():
        if variant not in bounded_sample_variants.keys():
            all_trajectories_with_bounded[variant]=timepoints
        else:
            all_trajectories_with_bounded[variant]= bounded_sample_variants_age_order[variant]

    return bounded_sample_variants, all_trajectories_with_bounded

# Theoretical expectations based on no ecology:
# + The idea here is to assume all clones compete via the simplest possible contraint that frequencies add to 1 (no ecology)
# + Each clone has a fitness and an establishment time (s, t) which toghether will all other clones completely determines the growth over time.
# + The set of {(s1, t1), (s2, t2)...} for the clones then determine the allele trajectories that can be fit to the data
# + Each mutation defines a clone
# + Define a clonal composition and then calculate allele trjaectories based on that

def get_effective_depths_for_variant_each_timepoint(sample_name, ages_dict_names, clonal_structure, variant_details):

    depths_by_timepoint = {}
    variants_looked_at = []

    for clone in clonal_structure:
        for variant in clone:
            if variant not in variants_looked_at:
                if (variant not in ['X', 'Y', 'Z']) and ('chr' not in variant): #not an mCA or missing driver
                    chromosome = variant_details[variant][0]
                    position = variant_details[variant][1]

                    for age, sample_timepoint in ages_dict_names.items():
                        #retrieve depth
                        if variant != 'FLT3_ITD':
                            try:
                                df= pd.read_csv('Data_files/VCF_files/DCS'+sample_timepoint+'_SNV_SNV_watson_code_DCS_variants_MUFs_3_all_positions.vcf', comment = '#', sep = '\t', header = None, names = ['chromosome', 'position', 'ID', 'REF','ALT', 'FILTER', 'INFO', 'FORMAT', 'SAMPLE'])
                            except FileNotFoundError:
                                df= pd.read_csv('Data_files/VCF_files/DCS'+sample_timepoint+'_SNV_watson_code_DCS_variants_MUFs_3_all_positions.vcf', comment = '#', sep = '\t', header = None, names = ['chromosome', 'position', 'ID', 'REF','ALT', 'FILTER', 'INFO', 'FORMAT', 'SAMPLE'])
                        else: #if FLT3-ITD, look at the SSCS depth...
                            try:
                                df= pd.read_csv('Data_files/VCF_files/SSCS'+sample_timepoint+'_SNV_SNV_watson_code_SSCS_variants_MUFs_3_all_positions.vcf', comment = '#', sep = '\t', header = None, names = ['chromosome', 'position', 'ID', 'REF','ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'SAMPLE'])
                            except FileNotFoundError:
                                df= pd.read_csv('Data_files/VCF_files/SSCS'+sample_timepoint+'_SNV_watson_code_SSCS_variants_MUFs_3_all_positions.vcf', comment = '#', sep = '\t', header = None, names = ['chromosome', 'position', 'ID', 'REF','ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'SAMPLE'])

                        df = df[(df['chromosome']==chromosome) & (df['position']==position)]
                        df['depth'] = df['INFO'].apply(extract_depth)
                        total_depth = df['depth'].tolist()[0]

                        if variant in depths_by_timepoint.keys():
                            depths_by_timepoint[variant][age]=total_depth
                        else:
                            depths_by_timepoint[variant]={age: total_depth}
                else:
                    for age, sample_timepoint in ages_dict_names.items():
                        if variant in depths_by_timepoint.keys():
                            depths_by_timepoint[variant][age]=1800
                        else:
                            depths_by_timepoint[variant]={age: 1800}
                variants_looked_at.append(variant)

    return depths_by_timepoint

def log_binomial(n, k):
    return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)

def log_prob_binomial(n, k, p):
    try:
        result = log_binomial(n, k) + k * math.log(p) + (n - k) * math.log(1 - p)
    except ValueError:
        # print('ValueError in log_prob_binomial function')
        # print('n = ', n)
        # print('k = ', k)
        # print('p = ', p)
        result = -100000
    return result

#functions that maps clone trajectories back onto allele trajectories and compares to observed allele trajectories to assign likelihood
def calculate_likelihood_observed_trajectory(mut, observed_traj, upper_freq, pred_traj, variant_depths_by_age):
    #calculate likelihood of the inferred trajectory relative to the real (detected) datapoints (if the non-detected timepoint is feasible VAF)
    likelihood = 0
    for kk, vv in observed_traj.items(): #kk = age, vv = variant frequency
        if vv != 'not processed':
            observed_freq=float(vv)
            if 0.0<observed_freq<upper_freq: #only allow variants between a certain freq, e.g. <40% to contribute to likelihood
                timestep=round(kk*10)
                predicted_freq=pred_traj[timestep]
                effective_depth = variant_depths_by_age[mut][kk]
                n_eff=round(effective_depth*observed_freq)
                loglikelihood=-1.0*log_prob_binomial(effective_depth, n_eff, predicted_freq)
                likelihood+=loglikelihood

    return likelihood

def inferred_timepoint_penalisation(mut, inferred_timepoints, bounded_variant_dict, pred_traj, variant_depths_by_age):
    likelihood = 0
    for non_detected_timepoint in inferred_timepoints:
        non_detected_inferred_VAF = bounded_variant_dict[mut][non_detected_timepoint]
        #take in to account the non-detected timepoint - making sure it doesn't infer a trajectory the has VAFs that should have been detectable when they weren't
        observed_non_detected_freq=float(non_detected_inferred_VAF) #maximum frequency it could be (otherwise would have been detected))
        timestep=round(non_detected_timepoint*10)
        predicted_non_detected_freq=pred_traj[timestep]
        effective_depth = variant_depths_by_age[mut][non_detected_timepoint]
        n_eff=round(effective_depth*observed_non_detected_freq)
        if predicted_non_detected_freq > observed_non_detected_freq: #add penalty if trying to fit trajectory higher than non-detected VAF
            # print('predicted freq is in range it should have bene detected -> assign high likelihood')
            loglikelihood=-1.0*log_prob_binomial(effective_depth, n_eff, predicted_non_detected_freq)
            likelihood+=3*loglikelihood #penalise the loglikelihood if predicted >bounded VAF

    return likelihood

def likelihood_clones(sample_name, clones, T, dt, bounded_variant_dict, variant_dict, sample_ages_lists, upper_freq, diagnosis_age, variant_depths_by_age): #upper_freq = upper VAF (or cell fraction) to allow the likelihood to include
    #e.g. clones={('DNMT3A_p.Q606X'):{'fitness':0.67, 'est_time':60}, ('DNMT3A_p.Q606X', 'SF3B1_p.K700E'):{'fitness':0.67, 'est_time':62.}}
    #e.g. variant_dict={'SF3B1_p.K700E': {73.73: 0.0801915020945541, 75.0: 0.0819563780568407, 75.83: 0.2144559173947577}, 'DNMT3A_p.Q606X': {73.73: 0.0774505849132714 etc...
    #e.g. bounded_variant_dict = variant_dict but just for the variants with inferred bounded VAFs

#     print('sample name = ', sample_name)
    sample_timings = sample_ages_lists[sample_name]
#     print('sample timings = ', sample_timings)
    # print('diagnosis age = ', diagnosis_age)

    for k, v in clones.items():
        s=v['fitness']
        tau=v['est_time']
        clone_size_trajectory=np.array([(1/s)*((np.exp(s*(i*dt-tau)))-1) for i in range(T)])
        clone_size_trajectory=np.where(clone_size_trajectory < 0, 0.0, clone_size_trajectory)
        v['clone_size_trajectory']=clone_size_trajectory #adds clone_size_trajectory to the clones dictionary

    total_clone_size=np.array([1.0*10**5 for i in range(T)]) #N = 10**5
    for k, v in clones.items():
        clone_size_trajectory=v['clone_size_trajectory']
        total_clone_size+=clone_size_trajectory

    for k, v in clones.items():
        clone_size_trajectory=v['clone_size_trajectory']
        clone_frequency_trajectory=clone_size_trajectory/total_clone_size
        v['clone_frequency_trajectory']=clone_frequency_trajectory #taking in to account changing N size (from other clones)

        for age, freq in zip(np.linspace(0, T*dt, T), clone_frequency_trajectory):
            if freq >= 1/(1.0*10**5):
                establishment_age = age
                break
        try:
            v['establishment_time']=establishment_age
        except: #if trajectory doesn't reach >1*10**5 during the lifespan, just assign the est_time
            v['establishment_time']=v['est_time']

    mutation_trajectories={}
    for k, v in clones.items():
        clone_frequency_trajectory=v['clone_frequency_trajectory']

        if 'JAK2_p.V617F' in k:
            if 'chr9p_CNLOH' in k: #i.e. if clone with both JAK2 V617F and 9p CNLOH in
                for mutation in k:
                    if mutation == 'JAK2_p.V617F':
                        if mutation in mutation_trajectories:
                            mutation_trajectories[mutation]+=clone_frequency_trajectory
                        else:
                            mutation_trajectories[mutation]=clone_frequency_trajectory
                    else:
                        if mutation in mutation_trajectories:
                            if mutation[0:3]=='chr': #i.e. if = mCA
                                mutation_trajectories[mutation]+=clone_frequency_trajectory
                            else:
                                mutation_trajectories[mutation]+=0.5*clone_frequency_trajectory
                        else: #if mutation not in mutation_trajectories
                            if mutation[0:3]=='chr': #i.e. if = mCA
                                mutation_trajectories[mutation]=clone_frequency_trajectory
                            else:
                                mutation_trajectories[mutation]=0.5*clone_frequency_trajectory

            else: #i.e. clone with JAK2 V617F, but no 9p CNLOH
                for mutation in k:
                    if mutation in mutation_trajectories:
                        if mutation[0:3]=='chr': #i.e. if = mCA
                            mutation_trajectories[mutation]+=clone_frequency_trajectory
                        else:
                            mutation_trajectories[mutation]+=0.5*clone_frequency_trajectory
                    else: #if mutation not in mutation_trajectories
                        if mutation[0:3]=='chr': #i.e. if = mCA
                            mutation_trajectories[mutation]=clone_frequency_trajectory
                        else:
                            mutation_trajectories[mutation]=0.5*clone_frequency_trajectory

        elif 'chr4q_CNLOH' in k:
            for mutation in k:
                if 'TET2' in mutation:
#                     print('chr4 CNLOH + TET2 mutation')
                    if mutation in mutation_trajectories:
                        mutation_trajectories[mutation]+=clone_frequency_trajectory
                    else:
                        mutation_trajectories[mutation]=clone_frequency_trajectory
                else:
                    if mutation in mutation_trajectories:
                        if mutation[0:3]=='chr': #i.e. if = mCA
                            mutation_trajectories[mutation]+=clone_frequency_trajectory
                        else:
                            mutation_trajectories[mutation]+=0.5*clone_frequency_trajectory
                    else: #if mutation not in mutation_trajectories
                        if mutation[0:3]=='chr': #i.e. if = mCA
                            mutation_trajectories[mutation]=clone_frequency_trajectory
                        else:
                            mutation_trajectories[mutation]=0.5*clone_frequency_trajectory

        else:
            for mutation in k:
                if mutation in mutation_trajectories:
                    if mutation[0:3]=='chr': #i.e. if = mCA
                        mutation_trajectories[mutation]+=clone_frequency_trajectory
                    else:
                        mutation_trajectories[mutation]+=0.5*clone_frequency_trajectory
                else: #if mutation not in mutation_trajectories
                    if mutation[0:3]=='chr': #i.e. if = mCA
                        mutation_trajectories[mutation]=clone_frequency_trajectory
                    else:
                        mutation_trajectories[mutation]=0.5*clone_frequency_trajectory

#     print('mutation trajectories keys = ', mutation_trajectories.keys())

    variant_likelihoods=[]
    total_likelihood=0.0
    for mut, pred_traj in mutation_trajectories.items(): #mut = each variant present (as apposed to clones)
        variant_likelihood=0.0

        if mut in variant_dict: #i.e. not an unseen driver this allows for unseen mutations to influence trajectories of others
            observed_traj=variant_dict[mut]

            if mut in bounded_variant_dict: #i.e. if it is a mutation that either appears late or disappears early...
                detected_timepoints = list(variant_dict[mut].keys()) #ages it was detected

                #if only detected in final timepoint, make sure it's fitness is higher than its predecessor clone (if it has one)
                if len(detected_timepoints)==1:
                    if detected_timepoints[0]==sample_timings[-1]: #i.e.final timepoint
                        clone_final_variant_detected=0
                        for k, v in clones.items():
                            if k[-1]==mut: #if the last variant in the clone is the variant of interest, e.g. CBL
                                clone_final_variant_detected+=1
                                current_clone_fitness = v['fitness'] #of e.g. (TET2, TET2, SRSF2, CBL) clone
                                current_clone_est_time = v['est_time']
                                current_clone_establishment_age = v['establishment_time'] #where the clone frequency reaches 1/N (i.e. 1/10**5)
                                full_clone = k
                                if len(full_clone)==1: #variant with only 1 timepoint is a single mutant clone
                                    if (current_clone_est_time > diagnosis_age) or (current_clone_est_time < 0): #penalise the optimiser if it tries to fit an est_time older than the diagnosis age
                                        variant_likelihood+=10000
                                    else:
                                        timepoints_in_bounded_dict = list(bounded_variant_dict[mut].keys())
                                        inferred_timepoints = []
                                        for timepoint in timepoints_in_bounded_dict:
                                            if timepoint not in detected_timepoints:
                                                inferred_timepoints.append(timepoint)

                                        #penalise the likelihood if trying to fit trajectory higher than non-detected VAF
                                        variant_likelihood += inferred_timepoint_penalisation(mut, inferred_timepoints, bounded_variant_dict, pred_traj, variant_depths_by_age)

                                        #calculate likelihood of the inferred trajectory relative to the real (detected) datapoints (if the non-detected timepoint is feasible VAF)
                                        variant_likelihood += calculate_likelihood_observed_trajectory(mut, observed_traj, upper_freq, pred_traj, variant_depths_by_age)

                                if len(full_clone)>1: #i.e. variant with only 1 timepoint is not a single mutant clone
                                    variant_of_interest = 0
                                    preceding_clone_final_variant = k[-2] #e.g. if clone = (TET2, TET2, SRSF2, CBL) clone, preceding clone final variant is SRSF2
                                    for k, v in clones.items():
                                        if k[-1]==preceding_clone_final_variant: #i.e. found the predecessor clone (e.g. TET2, TET2, SRSF2)
                                            # print('predecessor clone = ', k)
                                            variant_of_interest+=1
                                            predecessor_clone_fitness = v['fitness']
                                            predecessor_clone_est_time = v['est_time']
                                            predecessor_clone_establishment_age = v['establishment_time'] #where the clone frequency reaches 1/N (i.e. 1/10**5)
                                            # print('predecessor clone fitness = ', predecessor_clone_fitness)
                                            if (current_clone_est_time > diagnosis_age) or (current_clone_est_time < 0) or (current_clone_fitness < predecessor_clone_fitness) or (current_clone_est_time < predecessor_clone_est_time):
                                                variant_likelihood+=10000
                                            else: #if estimated establishment time and fitness is feasible...
                                                timepoints_in_bounded_dict = list(bounded_variant_dict[mut].keys()) #timepoints where variant was not dtected
                                                inferred_timepoints = []
                                                for timepoint in timepoints_in_bounded_dict:
                                                    if timepoint not in detected_timepoints:
                                                        inferred_timepoints.append(timepoint)

                                                #penalise the likelihood if trying to fit trajectory higher than non-detected VAF
                                                variant_likelihood += inferred_timepoint_penalisation(mut, inferred_timepoints, bounded_variant_dict, pred_traj, variant_depths_by_age)

                                                #calculate likelihood of the inferred trajectory relative to the real (detected) datapoints (if the non-detected timepoint is feasible VAF)
                                                variant_likelihood += calculate_likelihood_observed_trajectory(mut, observed_traj, upper_freq, pred_traj, variant_depths_by_age)

                                    if variant_of_interest == 0: #i.e. not found predecessor clone which has the penultimate variant as its final variant
                                    #e.g. predecessor of DNMT3A, GATA2, ASXL1 is GATA2, but there is no clone ending in GATA2, only e.g. (DNMT3A)
                                        if len(full_clone)>2:
                                            preceding_clone_final_variant = k[-3] #if 2 mutations were added at once to latest clone, try seeing if can find predecessor clone
                                            for k, v in clones.items():
                                                if k[-1]==preceding_clone_final_variant: #if found predecessor clone
                                                    variant_of_interest+=1
                                                    predecessor_clone_fitness = v['fitness']
                                                    predecessor_clone_est_time = v['est_time']
                                                    predecessor_clone_establishment_age = v['establishment_time'] #where the clone frequency reaches 1/N (i.e. 1/10**5)
                                                    # print('predecessor clone fitness = ', predecessor_clone_fitness)
                                                    if (current_clone_est_time > diagnosis_age) or (current_clone_est_time < 0) or (current_clone_fitness < predecessor_clone_fitness) or (current_clone_est_time < predecessor_clone_est_time):
                                                        variant_likelihood+=10000
                                                    else: #if estimated establishment time is not <0 or after time of diagnosis...
                                                        timepoints_in_bounded_dict = list(bounded_variant_dict[mut].keys())
                                                        inferred_timepoints = []
                                                        for timepoint in timepoints_in_bounded_dict:
                                                            if timepoint not in detected_timepoints:
                                                                inferred_timepoints.append(timepoint)

                                                        #penalise the likelihood if trying to fit trajectory higher than non-detected VAF
                                                        variant_likelihood += inferred_timepoint_penalisation(mut, inferred_timepoints, bounded_variant_dict, pred_traj, variant_depths_by_age)

                                                        #calculate likelihood of the inferred trajectory relative to the real (detected) datapoints (if the non-detected timepoint is feasible VAF)
                                                        variant_likelihood += calculate_likelihood_observed_trajectory(mut, observed_traj, upper_freq, pred_traj, variant_depths_by_age)

                    else: #detected at 1 timepoint, but the timepoint it is detected at is not the final timepoint
                        current_clone_est_time = 0
                        for k, v in clones.items():
                            if k[-1]==mut: #if the last variant in the clone is the variant of interest
                                current_clone_est_time = v['est_time']
                                current_clone_establishment_age = v['establishment_time'] #where the clone frequency reaches 1/N (i.e. 1/10**5)

                        if (current_clone_est_time > diagnosis_age) or (current_clone_est_time < 0): #penalise the optimiser if it tries to assign an est_time > diangnosis age
                            # print(str(full_clone)+' clone est_time > diagnosis age -> assign high likelihood')
                            variant_likelihood+=10000

                        else: #if estimated est_time is reasonable, check to make sure the trajectory is not unrealistic given the undetected variants
                            timepoints_in_bounded_dict = list(bounded_variant_dict[mut].keys())
                            inferred_timepoints = []
                            for timepoint in timepoints_in_bounded_dict:
                                if timepoint not in detected_timepoints:
                                    inferred_timepoints.append(timepoint)

                            #penalise the likelihood if trying to fit trajectory higher than non-detected VAF
                            variant_likelihood += inferred_timepoint_penalisation(mut, inferred_timepoints, bounded_variant_dict, pred_traj, variant_depths_by_age)

                            #calculate likelihood of the inferred trajectory relative to the real (detected) datapoints (if the non-detected timepoint is feasible VAF)
                            for kk, vv in observed_traj.items(): #kk = age, vv = variant frequency
                                if vv != 'not processed':
                                    observed_freq=float(vv)
                                    if 0.0<observed_freq<upper_freq: #only allow variants between a certain freq, e.g. <40% to contribute to likelihood
                                        timestep=round(kk*10)
                                        predicted_freq=pred_traj[timestep]
                                        effective_depth = variant_depths_by_age[mut][kk]
                                        n_eff=round(effective_depth*observed_freq)
                                        loglikelihood=-1.0*log_prob_binomial(effective_depth, n_eff, predicted_freq)
                                        if len(inferred_timepoints)>=2: #i.e. if there are 2 or more undetected timepoints, but only 1 detected, don't let the undetected pull it below the detected VAF
                                            if predicted_freq<observed_freq:
                                                variant_likelihood+=9*loglikelihood #penalise the loglikelihood if predicted < observed (i.e. penalise more than the penalty for the trajectory being above the non-observed timepoints)
                                            else:
                                                variant_likelihood+=loglikelihood
                                        else: #if only 1 timepoint not observed
                                            if predicted_freq<observed_freq:
                                                variant_likelihood+=3*loglikelihood #penalise the loglikelihood if predicted < observed (i.e. penalise more than the penalty for the trajectory being above the non-observed timepoints)
                                            else:
                                                variant_likelihood+=loglikelihood

                #detected at more than 1 timepoint, but either appears late and/ or disappears early (i.e. it is still in bounded variant dict)...
                else:
                    # print('not only detected in final timepoint, but has some missing timepoints')
                    current_clone_est_time = 0 #set to 0 to cover in case there isn't a clone present with the mutation as it's final variant (e.g. if only present in triple mutant clone)
                    clone_final_variant_detected = 0
                    for k, v in clones.items():
                        if k[-1]==mut: #if the last variant in the clone is the variant of interest
                            clone_final_variant_detected+=1
                            current_clone_fitness = v['fitness']
                            current_clone_est_time = v['est_time']
                            current_clone_establishment_age = v['establishment_time'] #where the clone frequency reaches 1/N (i.e. 1/10**5)
                            full_clone = k
                            if len(full_clone)==1: #variant with only 1 timepoint is a single mutant clone
                                if (current_clone_est_time > diagnosis_age) or (current_clone_est_time < 0): #penalise the optimiser if it tries to fit an est_time older than the diagnosis age
                                    variant_likelihood+=10000
                                else:
                                    timepoints_in_bounded_dict = list(bounded_variant_dict[mut].keys())
                                    inferred_timepoints = []
                                    for timepoint in timepoints_in_bounded_dict:
                                        if timepoint not in detected_timepoints:
                                            inferred_timepoints.append(timepoint)

                                    #penalise the likelihood if trying to fit trajectory higher than non-detected VAF
                                    variant_likelihood += inferred_timepoint_penalisation(mut, inferred_timepoints, bounded_variant_dict, pred_traj, variant_depths_by_age)

                                    #calculate likelihood of the inferred trajectory relative to the real (detected) datapoints (if the non-detected timepoint is feasible VAF)
                                    variant_likelihood += calculate_likelihood_observed_trajectory(mut, observed_traj, upper_freq, pred_traj, variant_depths_by_age)

                            if len(full_clone)>1: #i.e. variant with only 1 timepoint is not a single mutant clone
                                variant_of_interest = 0
                                preceding_clone_final_variant = k[-2]
                                # print("preceding clone final variant = ", preceding_clone_final_variant)
                                for k, v in clones.items():
                                    if k[-1]==preceding_clone_final_variant:
                                        # print('predecessor clone = ', k)
                                        variant_of_interest+=1
                                        predecessor_clone_fitness = v['fitness']
                                        predecessor_clone_est_time = v['est_time']
                                        predecessor_clone_establishment_age = v['establishment_time'] #where the clone frequency reaches 1/N (i.e. 1/10**5)
                                        # print('predecessor clone fitness = ', predecessor_clone_fitness)
                                        if (current_clone_est_time > diagnosis_age) or (current_clone_est_time < 0) or (current_clone_fitness < predecessor_clone_fitness) or (current_clone_est_time < predecessor_clone_est_time):
                                            variant_likelihood+=10000
                                        else: #if estimated establishment time is not <0 or after time of diagnosis...
                                            timepoints_in_bounded_dict = list(bounded_variant_dict[mut].keys())
                                            inferred_timepoints = []
                                            for timepoint in timepoints_in_bounded_dict:
                                                if timepoint not in detected_timepoints:
                                                    inferred_timepoints.append(timepoint)

                                            #penalise the likelihood if trying to fit trajectory higher than non-detected VAF
                                            variant_likelihood += inferred_timepoint_penalisation(mut, inferred_timepoints, bounded_variant_dict, pred_traj, variant_depths_by_age)

                                            #calculate likelihood of the inferred trajectory relative to the real (detected) datapoints (if the non-detected timepoint is feasible VAF)
                                            variant_likelihood += calculate_likelihood_observed_trajectory(mut, observed_traj, upper_freq, pred_traj, variant_depths_by_age)

                                if variant_of_interest == 0: #i.e. not found predecessor clone which has the penultimate variant as its final variant
                                #e.g. predecessor of DNMT3A, GATA2, ASXL1 is GATA2, but there is no clone ending in GATA2, only e.g. (DNMT3A)
                                    if len(full_clone)>2:
                                        preceding_clone_final_variant = k[-3] #if 2 mutations were added at once to latest clone, try seeing if can find predecessor clone
                                        for k, v in clones.items():
                                            if k[-1]==preceding_clone_final_variant: #if found predecessor clone
                                                variant_of_interest+=1
                                                predecessor_clone_fitness = v['fitness']
                                                predecessor_clone_est_time = v['est_time']
                                                predecessor_clone_establishment_age = v['establishment_time'] #where the clone frequency reaches 1/N (i.e. 1/10**5)
                                                # print('predecessor clone fitness = ', predecessor_clone_fitness)
                                                if (current_clone_est_time > diagnosis_age) or (current_clone_est_time < 0) or (current_clone_fitness < predecessor_clone_fitness) or (current_clone_est_time < predecessor_clone_est_time):
                                                    variant_likelihood+=10000
                                                else: #if estimated establishment time is not <0 or after time of diagnosis...
                                                    timepoints_in_bounded_dict = list(bounded_variant_dict[mut].keys())
                                                    inferred_timepoints = []
                                                    for timepoint in timepoints_in_bounded_dict:
                                                        if timepoint not in detected_timepoints:
                                                            inferred_timepoints.append(timepoint)

                                                    #penalise the likelihood if trying to fit trajectory higher than non-detected VAF
                                                    variant_likelihood += inferred_timepoint_penalisation(mut, inferred_timepoints, bounded_variant_dict, pred_traj, variant_depths_by_age)

                                                    #calculate likelihood of the inferred trajectory relative to the real (detected) datapoints (if the non-detected timepoint is feasible VAF)
                                                    variant_likelihood += calculate_likelihood_observed_trajectory(mut, observed_traj, upper_freq, pred_traj, variant_depths_by_age)

            else: #if variant detected at all timepoints
#                 print('variant is detected at all timepoits')
                current_clone_est_time = 0 #set to 0 to cover in case there isn't a clone present with the mutation as it's final variant (e.g. if only present in triple mutant clone)
                predecessor_clone_est_time = 0 #set to 0 to cover in case there isn't a predecessor clone
                for k, v in clones.items():
                    # print('k in clones = ', k)
                    if k[-1]==mut: #if the last variant in the clone is the variant of interest
                        current_clone_est_time = v['est_time']
                        full_clone = k
                        if len(full_clone)>1: #i.e. not a single mutant clone
                            preceding_clone_final_variant = k[-2]
                            for k, v in clones.items():
                                if k[-1]==preceding_clone_final_variant:
                                    predecessor_clone_est_time = v['est_time']
                                    predecessor_clone_establishment_age = v['establishment_time'] #where the clone frequency reaches 1/N (i.e. 1/10**5)

                if (current_clone_est_time > diagnosis_age) or (current_clone_est_time < 0) or (current_clone_est_time < predecessor_clone_est_time): #penalise the optimiser if it tries to assign an est_time > diangnosis age
                    # print(str(full_clone)+' clone est_time > diagnosis age -> assign high likelihood')
                    variant_likelihood+=10000

                else:
                    #calculate likelihood of the inferred trajectory relative to the real (detected) datapoints (if the non-detected timepoint is feasible VAF)
                    variant_likelihood += calculate_likelihood_observed_trajectory(mut, observed_traj, upper_freq, pred_traj, variant_depths_by_age)

            variant_likelihoods.append((mut, variant_likelihood))
            total_likelihood+=variant_likelihood


        else: #variant is a missing driver
#             print('variant is a missing driver')
            if mut in ['X', 'Y', 'Z']: #missing driver
#                 print('mutation = ', mut)
                predecessor_clone_est_time = 0 #set to 0 to cover in case there isn't a predecessor clone
                predecessor_clone_fitness = 0 #set to 0 to cover in case there isn't a predecessor clone
                for k, v in clones.items():
                    if k[-1]==mut:
                        if len(k)>0:
                            current_clone_fitness = v['fitness']
                            current_clone_est_time = v['est_time']
                            current_clone_establishment_age = v['establishment_time'] #where the clone frequency reaches 1/N (i.e. 1/10**5)
#                             print('current clone fitness = ', current_clone_fitness)
                            full_clone = k
                            if len(full_clone)>1:
#                                 print('missing driver not a single mutant clone')
                                preceding_clone_final_variant = k[-2]
                                for k, v in clones.items():
                                    if k[-1]==preceding_clone_final_variant:
                                        predecessor_clone_fitness = v['fitness']
                                        predecessor_clone_est_time = v['est_time']
                                        current_clone_establishment_age = v['establishment_time'] #where the clone frequency reaches 1/N (i.e. 1/10**5)
            #                             print('predecessor clone fitness = ', predecessor_clone_fitness)
                if (current_clone_fitness < predecessor_clone_fitness) or (current_clone_est_time > diagnosis_age) or (current_clone_est_time < 0) or (current_clone_est_time < predecessor_clone_est_time): #make  'unlikely', otherwise don't impact the likelihood scores
                    # print('current clone fitness < predecessor clone fitness  OR '+str(full_clone)+' clone est_time > diagnosis age -> assign high likelihood')
                    variant_likelihood+=10000
                    variant_likelihoods.append((mut, variant_likelihood))
                    total_likelihood+=variant_likelihood

        # print()

    variant_likelihoods.sort(key=lambda x: x[1], reverse=True)
#     print(variant_likelihoods)
    worst_mutation=variant_likelihoods[0][0]

    return [total_likelihood, worst_mutation, mutation_trajectories]

def mutation_trajectories(sample_name, clones, T, dt): #upper_freq = upper VAF (or cell fraction) to allow the likelihood to include
    #e.g. clones={('DNMT3A_p.Q606X'):{'fitness':0.67, 'est_time':60}, ('DNMT3A_p.Q606X', 'SF3B1_p.K700E'):{'fitness':0.67, 'est_time':62.}}

    for k, v in clones.items():
        s=v['fitness']
        tau=v['est_time']
        clone_size_trajectory=np.array([(1/s)*((np.exp(s*(i*dt-tau)))-1) for i in range(T)])
        clone_size_trajectory=np.where(clone_size_trajectory < 0, 0.0, clone_size_trajectory)
        v['clone_size_trajectory']=clone_size_trajectory #adds clone_size_trajectory to the clones dictionary

    total_clone_size=np.array([1.0*10**5 for i in range(T)]) #N = 10**5
    for k, v in clones.items():
        clone_size_trajectory=v['clone_size_trajectory']
        total_clone_size+=clone_size_trajectory

    for k, v in clones.items():
        clone_size_trajectory=v['clone_size_trajectory']
        clone_frequency_trajectory=clone_size_trajectory/total_clone_size
        v['clone_frequency_trajectory']=clone_frequency_trajectory #taking in to account changing N size (from other clones)

        for age, freq in zip(np.linspace(0, T*dt, T), clone_frequency_trajectory):
            if freq >= 1/(1.0*10**5):
                establishment_age = age
                break
        v['establishment_time']=establishment_age

    mutation_trajectories={}
    for k, v in clones.items():
        clone_frequency_trajectory=v['clone_frequency_trajectory']

        if 'JAK2_p.V617F' in k:
            if 'chr9p_CNLOH' in k: #i.e. if clone with both JAK2 V617F and 9p CNLOH in
                for mutation in k:
                    if mutation == 'JAK2_p.V617F':
                        if mutation in mutation_trajectories:
                            mutation_trajectories[mutation]+=clone_frequency_trajectory
                        else:
                            mutation_trajectories[mutation]=clone_frequency_trajectory
                    else:
                        if mutation in mutation_trajectories:
                            if mutation[0:3]=='chr': #i.e. if = mCA
                                mutation_trajectories[mutation]+=clone_frequency_trajectory
                            else:
                                mutation_trajectories[mutation]+=0.5*clone_frequency_trajectory
                        else: #if mutation not in mutation_trajectories
                            if mutation[0:3]=='chr': #i.e. if = mCA
                                mutation_trajectories[mutation]=clone_frequency_trajectory
                            else:
                                mutation_trajectories[mutation]=0.5*clone_frequency_trajectory

            else: #i.e. clone with JAK2 V617F, but no 9p CNLOH
                for mutation in k:
                    if mutation in mutation_trajectories:
                        if mutation[0:3]=='chr': #i.e. if = mCA
                            mutation_trajectories[mutation]+=clone_frequency_trajectory
                        else:
                            mutation_trajectories[mutation]+=0.5*clone_frequency_trajectory
                    else: #if mutation not in mutation_trajectories
                        if mutation[0:3]=='chr': #i.e. if = mCA
                            mutation_trajectories[mutation]=clone_frequency_trajectory
                        else:
                            mutation_trajectories[mutation]=0.5*clone_frequency_trajectory

        elif 'chr4q_CNLOH' in k:
            for mutation in k:
                if 'TET2' in mutation:
#                     print('chr4 CNLOH + TET2 mutation')
                    if mutation in mutation_trajectories:
                        mutation_trajectories[mutation]+=clone_frequency_trajectory
                    else:
                        mutation_trajectories[mutation]=clone_frequency_trajectory
                else:
                    if mutation in mutation_trajectories:
                        if mutation[0:3]=='chr': #i.e. if = mCA
                            mutation_trajectories[mutation]+=clone_frequency_trajectory
                        else:
                            mutation_trajectories[mutation]+=0.5*clone_frequency_trajectory
                    else: #if mutation not in mutation_trajectories
                        if mutation[0:3]=='chr': #i.e. if = mCA
                            mutation_trajectories[mutation]=clone_frequency_trajectory
                        else:
                            mutation_trajectories[mutation]=0.5*clone_frequency_trajectory

        else:
            for mutation in k:
                if mutation in mutation_trajectories:
                    if mutation[0:3]=='chr': #i.e. if = mCA
                        mutation_trajectories[mutation]+=clone_frequency_trajectory
                    else:
                        mutation_trajectories[mutation]+=0.5*clone_frequency_trajectory
                else: #if mutation not in mutation_trajectories
                    if mutation[0:3]=='chr': #i.e. if = mCA
                        mutation_trajectories[mutation]=clone_frequency_trajectory
                    else:
                        mutation_trajectories[mutation]=0.5*clone_frequency_trajectory

    return mutation_trajectories

def clone_optimiser(sample_name, old_clones, T, dt, var_dict, var_dict_bounded, bounded_sample_variants, sample_ages_lists, upper_freq, fitness_adjustment, est_time_adjustment, steps, diagnosis_age, variant_depths_by_age):

    #e.g. old_clones={('DNMT3A_p.Q606X'):{'fitness':0.67, 'est_time':60}, ('DNMT3A_p.Q606X', 'SF3B1_p.K700E'):{'fitness':0.67, 'est_time':62.}}
    #T = total timesteps (1000)
    #dt = time interval (0.1)
    #var_dict = #e.g. {'SF3B1_p.K700E': {73.73: 0.0801915020945541, 75.0: 0.0819563780568407, 75.83: 0.2144559173947577}, 'DNMT3A_p.Q606X': {73.73: 0.0774505849132714 etc...

    likelihoods = []

    print('number of steps = ', steps)

    for i in range(steps): #run the optimiser e.g. 1000 times:

        temp=0.0001

        old = likelihood_clones(sample_name, old_clones, T, dt, bounded_sample_variants, var_dict, sample_ages_lists, upper_freq, diagnosis_age, variant_depths_by_age) #returns [total_likelihood, worst_mutation, mutation_trajectories]
        #also updates the old_clones dictionary to include the mutation trajectories for each clone, i.e. {(clone mutations): {'fitness': .., 'est_time':..., 'clone_size trajectory':..., 'clone_frequency trajectory':...}
        old_likelihood = old[0]
        old_worst_mut = old[1]

        new_clones=copy.deepcopy(old_clones)
        for clone, fit_time in new_clones.items(): #e.g. clone = ('DNMT3A_p.Q606X', 'SF3B1_p.K700E', 'TET2_p.P869Qfs*4_Deletion'), fit_time = {'fitness':0.67, 'est_time':60}
            #update the fitness in new_clones with a new fitness
            old_fitness=fit_time['fitness']
#             new_fitness=np.random.normal(old_fitness, 0.0004) #choose a new fitness from random normal distribution centered around old fitness
            new_fitness=np.random.normal(old_fitness, old_fitness*fitness_adjustment) #choose a new fitness from random normal distribution centered around old fitness
            fit_time['fitness']=new_fitness #update fitness in new_clones with new fitness

            #update the est_time in new_clones with a new est_time
            old_time=fit_time['est_time']
            new_time=np.random.normal(old_time, est_time_adjustment) #choose a new establishment time from random normal distribution centered around old establishment time
            fit_time['est_time']=new_time #update est_time in new_clones with new est time

        new=likelihood_clones(sample_name, new_clones, T, dt, bounded_sample_variants, var_dict, sample_ages_lists, upper_freq, diagnosis_age, variant_depths_by_age); #returns new [total_likelihood, worst_mutation, mutation_trajectories]
        #also updates the new_clones dictionary to include the mutation trajectories for each clone, i.e. {(clone mutations): {'fitness': .., 'est_time':..., 'clone_size trajectory':..., 'clone_frequency trajectory':...}
        new_likelihood = new[0]
        new_worst_mut=new[1]

        likelihoods.append(old_likelihood)

        if new_likelihood<old_likelihood: #the lower the likelihood number, the better the fit
#             print(i) #optimisert number
#             print(old_likelihood)
            old_clones=new_clones #accept the new clone
    #         print("move accepted and better")

        elif np.exp(-(new_likelihood-old_likelihood)/temp)> np.random.random():
            old_clones=new_clones
    #         print("move accepted and worse")

    fitness_establishment_time = {}
    for k, v in new_clones.items():
        fitness_establishment_time[k]={'fitness': v['fitness'], 'est_time': v['est_time'], 'establishment_time': v['establishment_time']}

#     print('fine tune optimiser output = ', fitness_establishment_time)
#     print('likelihood =', old_likelihood)

    return new_clones, fitness_establishment_time, likelihoods

def clone_optimiser_3_step(sample_name, old_clones, T, dt, var_dict, var_dict_bounded, bounded_sample_variants, sample_ages_lists, upper_freq, fitness_adjustment_list, est_time_adjustment_list, steps_list, diagnosis_age, variant_depths_by_age):

    fitness_adj_1 = fitness_adjustment_list[0]
    fitness_adj_2 = fitness_adjustment_list[1]
    fitness_adj_3 = fitness_adjustment_list[2]

    est_time_adj_1 = est_time_adjustment_list[0]
    est_time_adj_2 = est_time_adjustment_list[1]
    est_time_adj_3 = est_time_adjustment_list[2]

    steps_1 = steps_list[0] #coarse steps
    steps_2 = steps_list[1] #moderate steps
    steps_3 = steps_list[2] #fine steps

    # print('number of step 1 steps = ', steps_1)

    new_clones_1, fitness_est_1, likelihoods_1 = clone_optimiser(sample_name, old_clones, T, dt, var_dict, var_dict_bounded, bounded_sample_variants, sample_ages_lists, upper_freq, fitness_adj_1, est_time_adj_1, steps_1, diagnosis_age, variant_depths_by_age)

    initial_guess_2 = fitness_est_1
    new_clones_2, fitness_est_2, likelihoods_2 = clone_optimiser(sample_name, initial_guess_2, T, dt, var_dict, var_dict_bounded, bounded_sample_variants, sample_ages_lists, upper_freq, fitness_adj_2, est_time_adj_2, steps_2, diagnosis_age, variant_depths_by_age)

    initial_guess_3 = fitness_est_2
    new_clones_3, fitness_est_3, likelihoods_3 = clone_optimiser(sample_name, initial_guess_3, T, dt, var_dict, var_dict_bounded, bounded_sample_variants, sample_ages_lists, upper_freq, fitness_adj_3, est_time_adj_3, steps_3, diagnosis_age, variant_depths_by_age)

    return new_clones_3, fitness_est_3, likelihoods_1, likelihoods_2, likelihoods_3

def clone_optimiser_seeds(old_clones, T, dt, var_dict, var_dict_bounded, bounded_sample_variants, sample_ages_lists, upper_freq, sample_name, fitness_adjustment_list, est_time_adjustment_list, steps_list, seeds, diagnosis_age, variant_depths_by_age):

    #Plot the seed optimisation results
    fig, axes = plt.subplots(1, 3, figsize = (27, 6))
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    seeds_results = {}

    for i in range(1, seeds+1):
        print('seed '+str(i))
        new_clones, fitness_establishment_time, step_1_likelihoods, step_2_likelihoods, step_3_likelihoods = clone_optimiser_3_step(sample_name, old_clones, T, dt, var_dict, var_dict_bounded, bounded_sample_variants, sample_ages_lists, upper_freq, fitness_adjustment_list, est_time_adjustment_list, steps_list, diagnosis_age, variant_depths_by_age)
        seeds_results[i]={'clones': new_clones,
                         'fitness establishment times': fitness_establishment_time,
                          'likelihood': step_3_likelihoods[-1],
                         'step 1 likelihoods': step_1_likelihoods,
                         'step 2 likelihoods': step_2_likelihoods,
                         'step 3 likelihoods': step_3_likelihoods,}
        ax1.plot([i+1 for i in range(len(step_1_likelihoods))], step_1_likelihoods, label = str(i))
        ax2.plot([i+1 for i in range(len(step_2_likelihoods))], step_2_likelihoods, label = str(i))
        ax3.plot([i+1 for i in range(len(step_3_likelihoods))], step_3_likelihoods, label = str(i))


    step1_seed_likelihoods = {} #record the minimum likelihood (i.e. best likelihood) for each seed
    step2_seed_likelihoods = {} #record the minimum likelihood (i.e. best likelihood) for each seed
    step3_seed_likelihoods = {} #record the minimum likelihood (i.e. best likelihood) for each seed

    #get the results with the best likelihood overall
    for seed, results in seeds_results.items():
#         likelihood_result = min(results['step 3 likelihoods'])
        step1_seed_likelihoods[seed] = results['step 1 likelihoods'][-1]
        step2_seed_likelihoods[seed] = results['step 2 likelihoods'][-1]
        step3_seed_likelihoods[seed] = results['step 3 likelihoods'][-1]

    step1_seed_with_best_likelihood = min(step1_seed_likelihoods.items(), key=lambda x: x[1])[0]
    step1_best_likelihood = min(step1_seed_likelihoods.items(), key=lambda x: x[1])[1]
    step2_seed_with_best_likelihood = min(step2_seed_likelihoods.items(), key=lambda x: x[1])[0]
    step2_best_likelihood = min(step2_seed_likelihoods.items(), key=lambda x: x[1])[1]

    #get the seed that had the best overall likelihood in step 3
    seed_with_best_likelihood = min(step3_seed_likelihoods.items(), key=lambda x: x[1])[0]
    best_likelihood = min(step3_seed_likelihoods.items(), key=lambda x: x[1])[1]
    optimised_clones = seeds_results[seed_with_best_likelihood]['clones']
    fitness_est = seeds_results[seed_with_best_likelihood]['fitness establishment times']


    ax1.set_title('coarse optimisation\n(best likelihood = seed '+str(step1_seed_with_best_likelihood)+': '+str(int(step1_best_likelihood))+')', fontsize = 21)
    ax2.set_title('moderate optimisation\n(best likelihood = seed '+str(step2_seed_with_best_likelihood)+': '+str(int(step2_best_likelihood))+')', fontsize = 21)
    ax3.set_title('fine optimisation\n(best likelihood = seed '+str(seed_with_best_likelihood)+': '+str(int(best_likelihood))+')', fontsize = 21)

    for ax in axes.flatten():
        ax.set_yscale('log')
        ax.set_xlabel('optimisation number', fontsize = 21)
        ax.set_ylabel('likelihood', fontsize = 21)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_tick_params(width=1.5, color = grey3, length = 6)
        ax.xaxis.set_tick_params(width=1.5, color = grey3, length = 6)
        ax.tick_params(axis='both', which='major', labelsize=21)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color(grey3)

    ax1.text(0.016, 1.07, sample_name, transform=ax1.transAxes, fontsize = 16, color = 'black', zorder = 10,
                bbox=dict(facecolor='white', edgecolor=grey4, boxstyle='round, pad=0.5'), linespacing = 1.5)

    # ax3.legend(fontsize = 16, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig('Data_files/'+sample_name+'_optimiser_results_'+str(seeds)+'_seeds_v'+str(version)+'.pdf')

    final_seed_likelihoods = {} #record the minimum likelihood (i.e. best likelihood) for each seed
    #get the results with the best likelihood overall
    for seed, results in seeds_results.items():
#         likelihood_result = min(results['step 3 likelihoods'])
        likelihood_result = results['step 3 likelihoods'][-1]
        final_seed_likelihoods[seed]=likelihood_result

    #get the seed that had the best overall likelihood
    seed_with_best_likelihood = min(final_seed_likelihoods.items(), key=lambda x: x[1])[0]
    optimised_clones = seeds_results[seed_with_best_likelihood]['clones']
    fitness_est = seeds_results[seed_with_best_likelihood]['fitness establishment times']

    return seeds_results, optimised_clones, fitness_est

def plot_seed_results(optimised_seeds, sample_name, seeds, clone_colors, diagnosis_age):
    fig, ax1 = plt.subplots(1, 1, figsize = (9, 6))

    final_seed_likelihoods = {}

    for seed, results in optimised_seeds.items():
        likelihood_result = results['step 3 likelihoods'][-1]
        final_seed_likelihoods[seed]=likelihood_result

    seed_with_best_likelihood = min(final_seed_likelihoods.items(), key=lambda x: x[1])[0]
    best_likelihood = min(final_seed_likelihoods.items(), key=lambda x: x[1])[1]

    print(final_seed_likelihoods)
    print('seed with best likelihood = ', seed_with_best_likelihood)
    print('best likelihood = ', best_likelihood)

    fitnesses = []
    establishment_times = []
    clone_size_ranges = {}

    for seed, results in optimised_seeds.items():
        i = 0
        likelihood_result = final_seed_likelihoods[seed]

        #only plot if the likelihood for this seed is +/-20% of the seed that had the best overall likelihood
        if (0.8*best_likelihood)<= likelihood_result <= (1.2*best_likelihood):
            for clone, fitness_est in results['fitness establishment times'].items():
#                 print('clone = ', clone)
                fitness = fitness_est['fitness']*100 #best result from the fine-tune running of each seed
                establishment = fitness_est['establishment_time']  #best result from the fine-tune running of each seed
                mutation_color = clone_colors[clone]

                ax1.scatter(establishment, fitness, color = mutation_color, alpha = 0.5, lw = 1, edgecolor = 'k', s = 200, zorder = 20)
                if seed == seed_with_best_likelihood:
                    ax1.scatter(establishment, fitness, marker = 'X', color = mutation_color, alpha = 0.85, s = 450, lw = 2, edgecolor = 'k', zorder = 50, label = clone)
#                     ax1.annotate(clone, (establishment, fitness))

                fitnesses.append(fitness)
                establishment_times.append(establishment)
#                 print()

    # ax1.set_xscale('log')
    ax1.set_ylabel('fitness (% per year)', fontsize = 21)
    ax1.set_xlabel('establishment time (years)', fontsize = 21)
    # ax1.legend(fontsize = 16, loc='upper left', bbox_to_anchor=(1, 1))

    ax1.set_xlim(((min(establishment_times))-5), (max(establishment_times)+5), diagnosis_age)
    ax1.set_ylim(0, max(fitnesses)*1.15)

    ax1.xaxis.set_major_locator(MultipleLocator(10))
    ax1.xaxis.set_minor_locator(MultipleLocator(5))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.set_tick_params(width=1.5, color = grey3, length = 6)
    ax1.xaxis.set_tick_params(width=1.5, color = grey3, length = 6)
    ax1.tick_params(axis='both', which='major', labelsize=21)
    ax1.grid(zorder = 0, color = grey2)
    for axis in ['bottom','left']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.spines[axis].set_color(grey3)

    plt.tight_layout()
    plt.savefig('Data_files/'+sample_name+'_seed_results_'+str(seeds)+'_seeds_v'+str(version)+'.pdf')

    return final_seed_likelihoods, seed_with_best_likelihood, best_likelihood

def sort_trajectories_mutation_number_and_est_time(optimised_clones, new_mut_traj):
    #Sort the trajectories by acquisition age and number of mutations
    #Sort the optimised clones dictioanry by age of acquisition and number of mutations in clone
    #Sort by acquisition age
    acquisition_age = {}
    for k, v in optimised_clones.items():
        acquisition_age[k]=v['establishment_time']

    acquisition_age = {k: v for k, v in sorted(acquisition_age.items(), key=lambda item: item[1])}

    #Next sort by number of mutations in clone
    sorted_by_number_mutations = {}
    for k in sorted(acquisition_age, key=len, reverse=False):
        sorted_by_number_mutations[k] = acquisition_age[k]

    #Order the optimised clones dictionary
    sorted_optimised_clones = {}
    for k in sorted_by_number_mutations.keys():
        v = optimised_clones[k]
        sorted_optimised_clones[k]=v

    #Order the variant mutation trajectories by same order as clones dictionary
    new_mut_traj_ordered = {}

    variants_ordered = []
    for k in sorted_optimised_clones.keys():
        for i in k:
            if i not in variants_ordered:
                variants_ordered.append(i)

    for variant in variants_ordered:
        trajectory = new_mut_traj[variant]
        new_mut_traj_ordered[variant]=trajectory

    return new_mut_traj_ordered, sorted_optimised_clones, variants_ordered

def convert_inferred_variant_trajectories_to_cell_fractions(new_mut_traj_ordered, sorted_optimised_clones):
    #Convert the mutation trajectoris to cell fraction trajectories
    new_mut_traj_ordered_cell_fractions = {}

    for mutation, trajectory in new_mut_traj_ordered.items():
        cell_fraction_trajectory = []
        if mutation == 'JAK2_p.V617F':
            CNLOH_present = 0
            for clone in sorted_optimised_clones.keys(): #info = {'fitness': .., 'est_time': ..., 'clone_size_trajectory':..., 'clone_frequency_trajectory':...}
                if 'chr9p_CNLOH' in clone:
                    CNLOH_present+=1
            if CNLOH_present >0: #person has a clone with both JAK2 and 9p in
                timepoint=0
                for VAF in trajectory:
                    chr9p_cell_fraction = new_mut_traj_ordered['chr9p_CNLOH'][timepoint]
                    JAK2_cell_fraction = ((VAF-chr9p_cell_fraction)*2)+chr9p_cell_fraction
                    cell_fraction_trajectory.append(JAK2_cell_fraction)
                    timepoint+=1
            else: #person has JAK2 but no clone with 9p in
                for VAF in trajectory:
                    cell_fraction_trajectory.append(VAF*2)

        elif 'TET2' in mutation:
            CNLOH_present = 0
            for clone in sorted_optimised_clones.keys(): #info = {'fitness': .., 'est_time': ..., 'clone_size_trajectory':..., 'clone_frequency_trajectory':...}
                if 'chr4q_CNLOH' in clone:
                    CNLOH_present+=1
            if CNLOH_present >0: #person has a clone with both JAK2 and 9p in
                timepoint=0
                for VAF in trajectory:
                    chr4q_cell_fraction = new_mut_traj_ordered['chr4q_CNLOH'][timepoint]
                    TET2_cell_fraction = ((VAF-chr4q_cell_fraction)*2)+chr4q_cell_fraction
                    cell_fraction_trajectory.append(TET2_cell_fraction)
                    timepoint+=1
            else: #person has TET2 but no clone with 4q in
                for VAF in trajectory:
                    cell_fraction_trajectory.append(VAF*2)

        elif 'chr' in mutation:
            for VAF in trajectory:
                cell_fraction_trajectory.append(VAF) #already a cell fraction rather than a VAF

        else:
            for VAF in trajectory:
                cell_fraction_trajectory.append(VAF*2)

        new_mut_traj_ordered_cell_fractions[mutation]= cell_fraction_trajectory

    return new_mut_traj_ordered_cell_fractions

def convert_sample_variants_to_cell_fractions(sample_variants, sorted_optimised_clones):
    #Convert the sample variants VAFs to cell fractions
    sample_variants_cell_fractions = {}

    for mutation, trajectory in sample_variants.items():
        cell_fraction_trajectory = {}
        if mutation == 'JAK2_p.V617F':
            CNLOH_present = 0
            for clone in sorted_optimised_clones.keys(): #info = {'fitness': .., 'est_time': ..., 'clone_size_trajectory':..., 'clone_frequency_trajectory':...}
                if 'chr9p_CNLOH' in clone:
                    CNLOH_present+=1
            if CNLOH_present >0: #person has a clone with both JAK2 and 9p in
                for age, VAF in trajectory.items():
                    chr9p_cell_fraction = sample_variants['chr9p_CNLOH'][age]
                    JAK2_cell_fraction = ((VAF-chr9p_cell_fraction)*2)+chr9p_cell_fraction
                    cell_fraction_trajectory[age]=JAK2_cell_fraction
            else: #person has JAK2 but no clone with 9p in
                for age, VAF in trajectory.items():
                    cell_fraction_trajectory[age]=VAF*2

        elif 'TET2' in mutation:
            CNLOH_present = 0
            for clone in sorted_optimised_clones.keys(): #info = {'fitness': .., 'est_time': ..., 'clone_size_trajectory':..., 'clone_frequency_trajectory':...}
                if 'chr4q_CNLOH' in clone:
                    CNLOH_present+=1
            if CNLOH_present >0: #person has a clone with both JAK2 and 9p in
                for age, VAF in trajectory.items():
                    chr4q_cell_fraction = sample_variants['chr4q_CNLOH'][age]
                    TET2_cell_fraction = ((VAF-chr4q_cell_fraction)*2)+chr4q_cell_fraction
                    cell_fraction_trajectory[age]=TET2_cell_fraction
            else: #person has TET2 but no clone with 4q in
                for age, VAF in trajectory.items():
                    cell_fraction_trajectory[age]=VAF*2

        elif 'chr' in mutation:
            for age, VAF in trajectory.items():
                cell_fraction_trajectory[age]=VAF #already a cell fraction rather than a VAF

        else:
            for age, VAF in trajectory.items():
                cell_fraction_trajectory[age]=VAF*2

        sample_variants_cell_fractions[mutation]= cell_fraction_trajectory

    return sample_variants_cell_fractions

def convert_bounded_sample_variants_to_cell_fractions(sample_variants, bounded_sample_variants, sorted_optimised_clones):
    #Convert the sample variants VAFs to cell fractions
    bounded_sample_variants_cell_fractions = {}

    for mutation, trajectory in bounded_sample_variants.items():
        cell_fraction_trajectory = {}
        if mutation == 'JAK2_p.V617F':
            CNLOH_present = 0
            for clone in sorted_optimised_clones.keys(): #info = {'fitness': .., 'est_time': ..., 'clone_size_trajectory':..., 'clone_frequency_trajectory':...}
                if 'chr9p_CNLOH' in clone:
                    CNLOH_present+=1
            if CNLOH_present >0: #person has a clone with both JAK2 and 9p in
                for age, VAF in trajectory.items():
                    chr9p_cell_fraction = sample_variants['chr9p_CNLOH'][age] #look in full dicitonary, not just bounded dictionary
                    JAK2_cell_fraction = ((VAF-chr9p_cell_fraction)*2)+chr9p_cell_fraction
                    cell_fraction_trajectory[age]=JAK2_cell_fraction
            else: #person has JAK2 but no clone with 9p in
                for age, VAF in trajectory.items():
                    cell_fraction_trajectory[age]=VAF*2

        elif 'TET2' in mutation:
            CNLOH_present = 0
            for clone in sorted_optimised_clones.keys(): #info = {'fitness': .., 'est_time': ..., 'clone_size_trajectory':..., 'clone_frequency_trajectory':...}
                if 'chr4q_CNLOH' in clone:
                    CNLOH_present+=1
            if CNLOH_present >0: #person has a clone with both JAK2 and 9p in
                for age, VAF in trajectory.items():
                    chr4q_cell_fraction = sample_variants['chr4q_CNLOH'][age]
                    TET2_cell_fraction = ((VAF-chr4q_cell_fraction)*2)+chr4q_cell_fractionn
                    cell_fraction_trajectory[age]=TET2_cell_fraction
            else: #person has TET2 but no clone with 4q in
                for age, VAF in trajectory.items():
                    cell_fraction_trajectory[age]=VAF*2

        elif 'chr' in mutation:
            for age, VAF in trajectory.items():
                cell_fraction_trajectory[age]=VAF #already a cell fraction rather than a VAF

        else:
            for age, VAF in trajectory.items():
                cell_fraction_trajectory[age]=VAF*2

        bounded_sample_variants_cell_fractions[mutation]= cell_fraction_trajectory

    return bounded_sample_variants_cell_fractions

def plot_trajectories_cell_fractions(sample_name, optimised_clones, T, dt, sample_variants, bounded_sample_variants, sample_ages, sample_ages_lists, diagnosis_age, upper_freq, seeds, multiple_mutant_clone_colors, df_germline, plot_bounded_or_not, variant_depths_by_age):
    new_mut_traj= mutation_trajectories(sample_name, optimised_clones, T, dt) #gets the individual mutation trajectories (as apposed to clone trajectories)

    #Sort the optimised clones dictioanry by age of acquisition and number of mutations in clone
    new_mut_traj_ordered, sorted_optimised_clones, variants_ordered = sort_trajectories_mutation_number_and_est_time(optimised_clones, new_mut_traj)

    new_mut_traj_ordered_cell_fractions = convert_inferred_variant_trajectories_to_cell_fractions(new_mut_traj_ordered, sorted_optimised_clones)
    sample_variants_cell_fractions = convert_sample_variants_to_cell_fractions(sample_variants, sorted_optimised_clones)
    bounded_sample_variants_cell_fractions = convert_bounded_sample_variants_to_cell_fractions(sample_variants, bounded_sample_variants, sorted_optimised_clones)

    #Plotting
    plt.close('all')
    fig, axes = plt.subplots(4, 1, figsize = (19, 26))
    plt.subplots_adjust(hspace=0.4)

    ax1 = axes[0] #log scale
    ax3 = axes[1] #linear scale
    ax2 = axes[2] #log zoomed in
    ax4 = axes[3] #linear zoomed in

    ages = sample_ages_lists[sample_name] #create a list of ages at which a sample was taken

    #colouring according to number of mutations
    variants_of_multiple_mutant_clones = {} #e.g. {1: [DNMT3A R882H, TET2...], 2: [SRSF2 P95H..]}
    number_of_multiple_mutant_clones = {} #e.g. {1: 2, 2: 1, 3: 2} = 2 single-mutant clones, 1 double-mutant, 2 triple-mutant

    for variant, new_traj in new_mut_traj_ordered_cell_fractions.items():
#         print('variant = ', variant)
        gene = variant.split('_')[0]
        # print('mutation classes dictionary = ', mutation_classes)
        mutation_class = mutation_classes[gene]
        for clone in sorted_optimised_clones.keys():
            if variant in clone:
                number_mutations = clone.index(variant)+1
#         print('number of mutations in clone = ', number_mutations)

        if number_mutations in variants_of_multiple_mutant_clones.keys():
            if variant not in variants_of_multiple_mutant_clones[number_mutations]:
                variants_of_multiple_mutant_clones[number_mutations].append(variant)
        else:
            variants_of_multiple_mutant_clones[number_mutations]=[variant]

    for k, v in variants_of_multiple_mutant_clones.items():
        number_of_multiple_mutant_clones[k]=len(v)

#     print('number of multiple mutant clones = ', number_of_multiple_mutant_clones)

    colors_plotted = {}
    number_mutations_plotted = {}
    clone_colors = {}
    variant_colors = {}

    for variant, new_traj in new_mut_traj_ordered_cell_fractions.items():
        print('variant = ', variant)
        gene = variant.split('_')[0]
        mutation_class = mutation_classes[gene]
        for clone in sorted_optimised_clones.keys():
            if variant in clone:
                number_mutations = clone.index(variant)+1

        if number_mutations in number_mutations_plotted.keys():
            number_of_clones_same_number_mutations = number_of_multiple_mutant_clones[number_mutations]
#             print('number of clones with same number mutations = ', number_of_clones_same_number_mutations)
            multiple_mutant_color_list = multiple_mutant_clone_colors[number_mutations][number_of_clones_same_number_mutations] #colour dictionary for plotting e.g. 5 single mutant clones on 1 plot
#             print('multiple_mutant_color_list = ', multiple_mutant_color_list)
            mutation_color = multiple_mutant_color_list[number_mutations_plotted[number_mutations]]
#             print('mutation_color = ', mutation_color)
            number_mutations_plotted[number_mutations]+=1
        else:
            number_of_clones_same_number_mutations = number_of_multiple_mutant_clones[number_mutations]
#             print('number of clones with same number mutations = ', number_of_clones_same_number_mutations)
            multiple_mutant_color_list = multiple_mutant_clone_colors[number_mutations][number_of_clones_same_number_mutations]
#             print('multiple_mutant_color_list = ', multiple_mutant_color_list)
            mutation_color = multiple_mutant_color_list[0]
            print('mutation_color = ', mutation_color)
            number_mutations_plotted[number_mutations]=1

        for ax in [ax1, ax2]:
            if mutation_class in ['missing driver', 'missing driver 2']:
                ax.plot(np.linspace(0, 100, 1000), new_traj,linewidth=3.75, color=mutation_color, linestyle = '--', zorder = 10) #plot line
            else:
                ax.plot(np.linspace(0, 100, 1000), new_traj,linewidth=3.75, color=mutation_color, zorder = 10) #plot line

        for ax in [ax3, ax4]:
            if mutation_class in ['missing driver', 'missing driver 2']:
                ax.plot(np.linspace(0, 100, 1000), new_traj, linewidth=3.75, linestyle = '--', color=mutation_color, zorder = 10)
            else:
                ax.plot(np.linspace(0, 100, 1000), new_traj, linewidth=3.75, color=mutation_color, zorder = 10)

        colors_plotted[variant]=mutation_color

        if variant in sample_variants_cell_fractions:
            timepoints_detected = []
            variant_traj=sample_variants_cell_fractions[variant]
            variant_label = variant.replace('_', ' ')
            for kk, vv in variant_traj.items():
                if vv != 'not processed':
                    timepoints_detected.append(kk)
                    for ax in [ax1, ax2]:
                        ax.scatter(kk, float(vv), color=mutation_color, s=150, edgecolors='k', zorder=100, label=variant_label) #plot datapoints
                    for ax in [ax3, ax4]:
                        ax.scatter(kk, float(vv), color=mutation_color, s=150, edgecolors='k', zorder=100, label=variant_label)
                    variant_colors[variant]=mutation_color
            if plot_bounded_or_not == 'plot_bounded':
                if variant in bounded_sample_variants_cell_fractions.keys():
                    bounded_traj = bounded_sample_variants_cell_fractions[variant]
                    for kk, vv in bounded_traj.items():
                        if kk not in timepoints_detected: #i.e. only plot the inferred VAF (bounded) datapoints (as dashed markers)
                            for ax in [ax1, ax2]:
                                ax.scatter(kk, float(vv), color='None', s=150, edgecolors=mutation_color, zorder=100, lw=1.5, linestyle='-') #plot datapoints
                            for ax in [ax3, ax4]:
                                ax.scatter(kk, float(vv), color='None', s=150, edgecolors=mutation_color, zorder=100, lw=1.5, linestyle='--')

    for time in ages:
        for ax in axes.flatten():
            ax.plot([time, time], [0.0000005, 1.0], color = grey2, lw = 2, linestyle = ':', zorder = 0)

    #plot diagnosis age
    for ax in [ax1, ax2]:
        ax.text(diagnosis_age, 1.7, 'AML \n diagnosis', ha = 'center', fontsize = 19)
        ax.plot([diagnosis_age, diagnosis_age], [0.0000005, 1.0], color = 'black', lw = 2, linestyle = '--', zorder = 150)

    for ax in [ax3, ax4]:
        ax.text(diagnosis_age, 1.03, 'AML \n diagnosis', ha = 'center', fontsize = 19)
        ax.plot([diagnosis_age, diagnosis_age], [0.0000005, 1.0], color = 'black', lw = 2, linestyle = '--', zorder = 150)

    #plot acquisition ages
    for k, v in optimised_clones.items():
        fitness = v['fitness']
        est_time = v['establishment_time']
        final_variant = k[-1]
        clone_color = colors_plotted[final_variant]
        clone_colors[k] = clone_color
        for ax in axes.flatten():
            ax1.annotate('', xy=(est_time, 0.00001), ha="center", va="top", xytext=(0, -25), textcoords='offset points',
                         arrowprops=dict(arrowstyle="simple, head_length=1, head_width=1.5, tail_width=0.5",
                                         facecolor=clone_color,
                                         edgecolor = grey4))

    #write fitness effects
    y_pos = 0.9
    text_string = ''
    for k, v in sorted_optimised_clones.items():
        fitness = v['fitness']
        gene_list = []
        for i in k:
            gene_list.append(i.split('_')[0])
        if len(gene_list) == 1:
            clone_size = ''
            clone = gene_list[0]+', '
        if len(gene_list)>1:
            clone = ''
            for gene in gene_list:
                clone+=gene+', '
            if len(gene_list) == 2:
                clone_size = 'double'
            if len(gene_list) == 3:
                clone_size = 'triple'
            if len(gene_list) == 4:
                clone_size = 'quadruple'
            if len(gene_list) == 5:
                clone_size = 'quintuple'
            if len(gene_list) == 6:
                clone_size = 'sextuple'
        text_string+=clone[:-2]+' '+clone_size+' mutant clone fitness = ~'+str(int(round(fitness*100, 0)))+' %'+'\n'
        y_pos-=0.1

    for ax in [ax1, ax3]:
        if y_pos >0.4:
            ax.text(0.016, ((0.9-y_pos)/2)+y_pos, text_string[:-1], transform=ax.transAxes, fontsize = 16, color = grey5, zorder = 10,
                    bbox=dict(facecolor='white', edgecolor=grey5, boxstyle='round, pad=0.5', alpha = 0.5), linespacing = 1.5)
        else:
            ax.text(0.016, ((0.7-y_pos)/2)+y_pos, text_string[:-1], transform=ax.transAxes, fontsize = 16, color = grey5, zorder = 10,
                    bbox=dict(facecolor='white', edgecolor=grey5, boxstyle='round, pad=0.5', alpha = 0.5), linespacing = 1.5)

    ###### format plot #######
    for ax in [ax1, ax2]:
        ax.set_yscale('log')

    #Set y tick labels
    y_major_ticks = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    y_major_tick_labels = ["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"]
    for ax in [ax1, ax2]:
        ax.set_yticks(y_major_ticks)
        ax.set_yticklabels(y_major_tick_labels)

    y_major_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    y_major_tick_labels = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
    for ax in [ax3, ax4]:
        ax.set_yticks(y_major_ticks)
        ax.set_yticklabels(y_major_tick_labels)

    #Set y-limits
    ax1.set_ylim(1*10**-5, 1.2)
    ax2.set_ylim(3*10**-4, 1.2)

    for ax in [ax3, ax4]:
        ax.set_ylim(0, 1.05)

    #Set x-limits
    for ax in [ax1, ax3]:
        ax.set_xlim(0, 86)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    for ax in [ax2, ax4]:
        ax.set_xlim(ages[0]-1, diagnosis_age+0.5)
        ax.xaxis.set_major_locator(MultipleLocator(1))

    for ax in axes.flatten():
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize = 16, loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xlabel('age', fontsize = 21)
        ax.set_ylabel('fraction of cells (%)', fontsize = 21)
        ax.set_title('')
        ax.grid(axis = 'y', which = 'both', zorder = 0, linestyle = ':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_tick_params(width=1.5, color = grey3, length = 6)
        ax.xaxis.set_tick_params(width=1.5, color = grey3, length = 6)
        ax.tick_params(axis='both', which='major', labelsize=21)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(1.5)
            ax.spines[axis].set_color(grey3)
        ax.text(0.016, 1.07, sample_name, transform=ax.transAxes, fontsize = 16, color = 'black', zorder = 10,
                bbox=dict(facecolor='white', edgecolor=grey4, boxstyle='round, pad=0.5'), linespacing = 1.5)

        germline_dict, germline_list, significant_germline, germline_df_summary = germline_variants(sample_name, df_germline)
        germline_text = 'germline: '
        for i in germline_list:
            germline_text+=i+', '
        germline_text = germline_text[:-2]
        ax.text(0.1, 1.11, germline_text, transform=ax.transAxes, fontsize = 12, color = 'black', zorder = 10, linespacing = 1.5)

        significant_germline_text = '?pathogenic germline: '
        for i in significant_germline:
            significant_germline_text+=i+', '
        if len(significant_germline)>1:
            significant_germline_text = significant_germline_text[:-2]
        ax.text(0.1, 1.05, significant_germline_text, transform=ax.transAxes, fontsize = 12, color = 'black', zorder = 10, linespacing = 1.5)

    plt.tight_layout()
    # plt.show()
    if plot_bounded_or_not == 'plot_bounded':
        plt.savefig('Data_files/'+sample_name+'_trajectories_'+str(seeds)+'_seeds_bounded_plotted_cell_fractions_v'+str(version)+'.pdf')
    else:
        plt.savefig('Data_files/'+sample_name+'_trajectories_'+str(seeds)+'_seeds_cell_fractions_v'+str(version)+'.pdf')

    return clone_colors, variant_colors, germline_df_summary, variants_ordered

def plot_trajectories_cell_fractions_small(sample_name, optimised_clones, T, dt, sample_variants, bounded_sample_variants, sample_ages, sample_ages_lists, diagnosis_age, upper_freq, seeds, multiple_mutant_clone_colors, df_germline, variant_depths_by_age):
    # old_mut_traj=likelihood_clones(old_clones, T, var_dict)[2]
    new_mut_traj= mutation_trajectories(sample_name, optimised_clones, T, dt) #gets the individual mutation trajectories (as apposed to clone trajectories)

    #Sort the optimised clones dictioanry by age of acquisition and number of mutations in clone
    new_mut_traj_ordered, sorted_optimised_clones, variants_ordered = sort_trajectories_mutation_number_and_est_time(optimised_clones, new_mut_traj)

    new_mut_traj_ordered_cell_fractions = convert_inferred_variant_trajectories_to_cell_fractions(new_mut_traj_ordered, sorted_optimised_clones)
    sample_variants_cell_fractions = convert_sample_variants_to_cell_fractions(sample_variants, sorted_optimised_clones)
    bounded_sample_variants_cell_fractions = convert_bounded_sample_variants_to_cell_fractions(sample_variants, bounded_sample_variants, sorted_optimised_clones)

    #Plotting
    plt.close('all')
    fig, ax1 = plt.subplots(1, 1, figsize = (9, 6))

    ages = sample_ages_lists[sample_name] #create a list of ages at which a sample was taken

    #colouring according to number of mutations
    variants_of_multiple_mutant_clones = {} #e.g. {1: [DNMT3A R882H, TET2...], 2: [SRSF2 P95H..]}
    number_of_multiple_mutant_clones = {} #e.g. {1: 2, 2: 1, 3: 2} = 2 single-mutant clones, 1 double-mutant, 2 triple-mutant

    for variant, new_traj in new_mut_traj_ordered_cell_fractions.items():
#         print('variant = ', variant)
        gene = variant.split('_')[0]
        mutation_class = mutation_classes[gene]
        for clone in sorted_optimised_clones.keys():
            if variant in clone:
                number_mutations = clone.index(variant)+1
#         print('number of mutations in clone = ', number_mutations)

        if number_mutations in variants_of_multiple_mutant_clones.keys():
            if variant not in variants_of_multiple_mutant_clones[number_mutations]:
                variants_of_multiple_mutant_clones[number_mutations].append(variant)
        else:
            variants_of_multiple_mutant_clones[number_mutations]=[variant]

    for k, v in variants_of_multiple_mutant_clones.items():
        number_of_multiple_mutant_clones[k]=len(v)

#     print('number of multiple mutant clones = ', number_of_multiple_mutant_clones)

    colors_plotted = {}
    number_mutations_plotted = {}

    for variant, new_traj in new_mut_traj_ordered_cell_fractions.items():
#         print('variant = ', variant)
        gene = variant.split('_')[0]
        mutation_class = mutation_classes[gene]
        for clone in sorted_optimised_clones.keys():
            if variant in clone:
                number_mutations = clone.index(variant)+1
#         print('number of mutations in clone = ', number_mutations)

        if number_mutations in number_mutations_plotted.keys():
            number_of_clones_same_number_mutations = number_of_multiple_mutant_clones[number_mutations]
#             print('number of clones with same number mutations = ', number_of_clones_same_number_mutations)
            multiple_mutant_color_list = multiple_mutant_clone_colors[number_mutations][number_of_clones_same_number_mutations] #colour dictionary for plotting e.g. 5 single mutant clones on 1 plot
#             print('multiple_mutant_color_list = ', multiple_mutant_color_list)
            mutation_color = multiple_mutant_color_list[number_mutations_plotted[number_mutations]]
#             print('mutation_color = ', mutation_color)
            number_mutations_plotted[number_mutations]+=1
        else:
            number_of_clones_same_number_mutations = number_of_multiple_mutant_clones[number_mutations]
#             print('number of clones with same number mutations = ', number_of_clones_same_number_mutations)
            multiple_mutant_color_list = multiple_mutant_clone_colors[number_mutations][number_of_clones_same_number_mutations]
#             print('multiple_mutant_color_list = ', multiple_mutant_color_list)
            mutation_color = multiple_mutant_color_list[0]
#             print('mutation_color = ', mutation_color)
            number_mutations_plotted[number_mutations]=1

        if mutation_class in ['missing driver', 'missing driver 2']:
            ax1.plot(np.linspace(0, 100, 1000), new_traj,linewidth=3.75, color=mutation_color, linestyle = '--', zorder = 10) #plot line
        else:
            ax1.plot(np.linspace(0, 100, 1000), new_traj,linewidth=3.75, color=mutation_color, zorder = 10) #plot line

        colors_plotted[variant]=mutation_color

        if variant in sample_variants_cell_fractions:
            timepoints_detected = []
            variant_traj=sample_variants_cell_fractions[variant]
            variant_label = variant.replace('_', ' ')
            for kk, vv in variant_traj.items():
                if vv != 'not processed':
                    timepoints_detected.append(kk)
                    ax1.scatter(kk, float(vv), color=mutation_color, s=150, edgecolors='k', zorder=100, label=variant_label) #plot datapoints
                    ax1.scatter(kk, float(vv), color=mutation_color, s=150, edgecolors='k', zorder=100, label=variant_label)
#             if variant in bounded_sample_variants.keys():
#                 bounded_traj = bounded_sample_variants[variant]
#                 for kk, vv in bounded_traj.items():
#                     if kk not in timepoints_detected: #i.e. only plot the inferred VAF (bounded) datapoints (as dashed markers)
#                         ax1.scatter(kk, float(vv), color='None', s=150, edgecolors=mutation_color, zorder=100, lw=1.5, linestyle='-') #plot datapoints
#                         ax1.scatter(kk, float(vv), color='None', s=150, edgecolors=mutation_color, zorder=100, lw=1.5, linestyle='--')

    for time in ages:
        ax1.plot([time, time], [0.0000005, 1.0], color = grey2, lw = 2, linestyle = ':', zorder = 0)

    #plot diagnosis age
#     ax1.text(diagnosis_age, 1.7, 'AML \n diagnosis', ha = 'center', fontsize = 19)
    ax1.plot([diagnosis_age, diagnosis_age], [0.0000005, 1.0], color = 'black', lw = 2, linestyle = '--', zorder = 150)

    #plot acquisition ages
    for k, v in sorted_optimised_clones.items():
        fitness = v['fitness']
        est_time = v['establishment_time']
        final_variant = k[-1]
        clone_color = colors_plotted[final_variant]
        lower_age = est_time-(1/fitness)
        upper_age = est_time+(1/fitness)
        bounds_est_time = abs(int(max(upper_age-est_time, est_time-lower_age)))
#         ax1.plot([lower_age, lower_age], [0, 1.0], color = grey2)
#         ax1.plot([upper_age, upper_age], [0, 1.0], color = grey2)
#         ax1.fill_between([lower_age, upper_age], 0, 1, color = clone_color, zorder = 0, alpha = 0.1)
        ax1.plot([est_time, est_time], [0.0000005, 1.0], color = grey3, lw = 2, linestyle = '--', zorder = 0)
    #     ax1.text(est_time, 1.7, str(k)+'\n~ acquisition age: \n~'+str(int(round(est_time, 0)))+' yrs (+/- '+str(bounds_est_time)+')', ha = 'center', fontsize =10, color = clone_color)

    ###### format plot #######
    ax1.set_yscale('log')

    #Set y tick labels
    y_major_ticks = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    y_major_tick_labels = ["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"]
    ax1.set_yticks(y_major_ticks)
    ax1.set_yticklabels(y_major_tick_labels)

    #Set y-limits
    ax1.set_ylim(3*10**-4, 1.2)

    #Set x-limits
    ax1.set_xlim(ages[0]-1, diagnosis_age+0.5)
    ax1.xaxis.set_major_locator(MultipleLocator(1))

#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     ax1.legend(by_label.values(), by_label.keys(), fontsize = 16, loc='upper left', bbox_to_anchor=(1, 1))
    ax1.set_xlabel('age', fontsize = 21)
    ax1.set_ylabel('fraction of cells (%)', fontsize = 21)
    ax1.set_title('')
    ax1.grid(axis = 'y', which = 'both', zorder = 0, linestyle = ':')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.set_tick_params(width=1.5, color = grey3, length = 6)
    ax1.xaxis.set_tick_params(width=1.5, color = grey3, length = 6)
    ax1.tick_params(axis='both', which='major', labelsize=21)
    for axis in ['bottom','left']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.spines[axis].set_color(grey3)
#     ax1.text(0.016, 1.07, sample_name, transform=ax1.transAxes, fontsize = 16, color = 'black', zorder = 10,
#             bbox=dict(facecolor='white', edgecolor=grey4, boxstyle='round, pad=0.5'), linespacing = 1.5)

    plt.tight_layout()

    return plt.savefig('Data_files/'+sample_name+'_trajectories_small_'+str(seeds)+'_seeds_cell_fractions_v'+str(version)+'.pdf')
#set the initial guess for clones depending on how many mutations in the clone:

def initial_fitness_establishment_time(sample_name, diagnosis_age):
    initial_fitness_establishment_guesses = {1: {'fitness': 0.1, 'establishment': diagnosis_age/5},
                                             2: {'fitness': 0.3, 'establishment': diagnosis_age/2},
                                             3: {'fitness': 0.9, 'establishment': diagnosis_age-10},
                                             4: {'fitness': 1.8, 'establishment': diagnosis_age-7},
                                             5: {'fitness': 3.0, 'establishment': diagnosis_age-5},
                                             6: {'fitness': 4.0, 'establishment': diagnosis_age-3}}

    return initial_fitness_establishment_guesses

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def estimate_trajectory_gradient_est_time(trajectory, clone, sample_name, version):

    ages = []
    logVAFs = []
    for age, VAF in trajectory.items():
        if 0 < VAF <0.4:
            ages.append(age)
            logVAFs.append(np.log(float(VAF)))

    print('clone = ', clone)
    print('ages = ', ages)
    print('logVAFs = ', logVAFs)

    if len(ages)<=1: #i.e. if not enough datapoints with VAF <0.4:
        s = 0
        t = 0

    else:
        if len(ages)>=4:
            early_ages, later_ages = split_list(ages)
            early_logVAFs, later_logVAFs = split_list(logVAFs)

            #find line of best fit using first half of datapoints
            s_early, c_early = np.polyfit(early_ages, early_logVAFs, 1)
            t_early = (np.log(1e-5) - c_early)/s_early

            #find line of best fit using second half of datapoints
            s_late, c_late = np.polyfit(later_ages, later_logVAFs, 1)
            t_late = (np.log(1e-5) - c_late)/s_late

            print('s early = ', s_early)
            print('s late = ', s_late)

            print('s_early/s_late = ',(s_late/s_early))

            #if gradient of later timepoints is <40% of early, use the earlier estimate
            if s_late/s_early < 0.4:
                s = s_early
                c = c_early
                t = t_early
            else: #find line of best fit using all datapoints
                s, c = np.polyfit(ages, logVAFs, 1)
                t = (np.log(1e-5) - c)/s

            print('s all = ', s)

        else:
            print('<4 datapoints, so use all data')
            s, c = np.polyfit(ages, logVAFs, 1)
            t = (np.log(1e-5) - c)/s

        #plot the line
        fig, ax1 = plt.subplots(1, 1, figsize = (5, 3.5))

        number_mutations_plotted = {}
        number_mutations = len(clone)
        if number_mutations in number_mutations_plotted.keys():
            mutation_color = mutation_number_colors[number_mutations][number_mutations_plotted[number_mutations]]
            number_mutations_plotted[number_mutations]+=1
        else:
            mutation_color = mutation_number_colors[number_mutations][0]
            number_mutations_plotted[number_mutations]=1

        ax1.plot(ages, logVAFs, color = mutation_color, zorder = 50, lw = 2)
        x = np.linspace(30, 80, 10)
        ax1.plot(x, s*x+c, color = grey3, linestyle = '--')
        ax1.set_xlim(0, 80)
        ax1.set_ylim(np.log(1e-5), np.log(0.5))

        #Set y tick labels
        y_major_ticks = [np.log(0.00001), np.log(0.0001), np.log(0.001), np.log(0.01), np.log(0.1), np.log(1.0)]
        y_major_tick_labels = ["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"]
        ax1.set_yticks(y_major_ticks)
        ax1.set_yticklabels(y_major_tick_labels)

        ax1.text(0.05, 0.90, 's (gradient) = '+str(round(s, 2)), transform=ax1.transAxes, fontsize = 12)
        ax1.text(0.05, 0.78, 'c (logy intercept) = '+str(round(c, 2)), transform=ax1.transAxes, fontsize = 12)
        ax1.text(0.05, 0.66, 't (logy = 1e-5 intercept) = '+str(round(t, 2)), transform=ax1.transAxes, fontsize = 12)

        clone_title_name = str(clone).replace('(', '').replace(')', '').replace("'", '').replace('_', ' ')

        ax1.set_xlabel('age', fontsize = 16)
        ax1.set_ylabel('VAF (or mCA cell fraction)', fontsize = 16)
        ax1.set_title(clone_title_name, y = 1.1)
        ax1.grid(axis = 'y', which = 'both', zorder = 0, linestyle = ':')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.yaxis.set_tick_params(width=1.5, color = grey3, length = 6)
        ax1.xaxis.set_tick_params(width=1.5, color = grey3, length = 6)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        for axis in ['bottom','left']:
            ax1.spines[axis].set_linewidth(1.5)
            ax1.spines[axis].set_color(grey3)

        clone_save_name = str(clone).replace('(', '').replace(')', '').replace("'", '').replace('_', ' ').replace("p.", '').replace("*", '').replace(" ", '_').replace(",", '')
        plt.tight_layout()
        plt.savefig('Data_files/'+sample_name+'_'+clone_save_name+'_gradient_estimation_v'+str(version)+'.pdf')

    return s, t

def create_initial_guess_clones_dictionary(sample_name, sample_variants, all_trajectories_with_bounded, list_of_clones, diagnosis_age, version):

    initial_clones = {}
    gradient_plots = []

    for clone in list_of_clones:
        number_of_mutations = len(clone)
        final_variant = clone[-1] #e.g. DNMT3A in (TET2, ASXL1, DNMT3A)
        if final_variant in all_trajectories_with_bounded.keys(): #i.e. is not e.g. 'X' (missing driver)
            non_bounded_final_variant_trajectory = sample_variants[final_variant] #not including the bounded VAF, e.g. {age: VAF, age: VAF}
            number_detected_timepoints = len(non_bounded_final_variant_trajectory)
            if number_detected_timepoints == 1: #only use the bounded if there is only a single timepoint in the non-bounded dictionary
                variant_trajectory = all_trajectories_with_bounded[final_variant] #including the bounded VAF
            else:
                variant_trajectory = non_bounded_final_variant_trajectory #not including the bounded VAF

            estimated_fitness, estimated_t = estimate_trajectory_gradient_est_time(variant_trajectory, clone, sample_name, version) #estimate fitness and t
            if estimated_fitness !=0:
                gradient_plots.append(clone)

            #if e.g. 4-mutant clone, but detected in only 1 timepoint, use the fitness estimate for the 3-mutant clone if it is higher
            if number_detected_timepoints == 1: #i.e. if variant only detected at one timepoint
                if len(clone)>1: #i.e. if not single mutant clone
                    penultimate_variant = clone[-2]
                    penultimate_clone = ''
                    for clone_ID in list_of_clones:
                        if clone_ID[-1]==penultimate_variant:
                            penultimate_clone = clone_ID

                    if penultimate_clone in initial_clones.keys():
                        estimated_penultimate_fitness = initial_clones[penultimate_clone]['fitness']
                        estimated_penultimate_t = initial_clones[penultimate_clone]['est_time']

                        if estimated_penultimate_fitness>estimated_fitness:
                            estimated_fitness = estimated_penultimate_fitness

                        if estimated_t < estimated_penultimate_t:
                            estimated_t = estimated_penultimate_t

                    else:
                        if penultimate_variant in all_trajectories_with_bounded.keys():
                            penultimate_variant_trajectory = all_trajectories_with_bounded[penultimate_variant]
                            estimated_penultimate_fitness, estimated_penultimate_t = estimate_trajectory_gradient_est_time(penultimate_variant_trajectory, clone, sample_name, version)

                            if estimated_penultimate_fitness>estimated_fitness:
                                if estimated_penultimate_fitness > 0:
                                    estimated_fitness = estimated_penultimate_fitness
                                else:
                                    estimated_fitness = guesses_based_on_mutation_number[number_of_mutations]['fitness']

                            if estimated_t < estimated_penultimate_t:
                                if estimated_penultimate_t < diagnosis_age:
                                    estimated_t = estimated_penultimate_t
                                else:
                                    estimated_t = guesses_based_on_mutation_number[number_of_mutations]['establishment']

            if estimated_t <0:
                estimated_t = 0
            if estimated_fitness <=0:
                guesses_based_on_mutation_number = initial_fitness_establishment_time(sample_name, diagnosis_age)
                estimated_fitness = guesses_based_on_mutation_number[number_of_mutations]['fitness']
                estimated_t = guesses_based_on_mutation_number[number_of_mutations]['establishment']

        else: #if missing driver
            if number_of_mutations == 1:
                guesses_based_on_mutation_number = initial_fitness_establishment_time(sample_name, diagnosis_age)
                estimated_fitness = guesses_based_on_mutation_number[number_of_mutations]['fitness']
                estimated_t = guesses_based_on_mutation_number[number_of_mutations]['establishment']

            else: #use the gradient of the final 2 timepoints of the predecessor clone if it is fitter than guess based on mutation_number
                penultimate_variant = clone[-2]
                non_bounded_final_variant_trajectory = sample_variants[penultimate_variant] #not including the bounded VAF, e.g. {age: VAF, age: VAF}
                variant_trajectory = {list(sample_variants[penultimate_variant].keys())[-2]: list(sample_variants[penultimate_variant].values())[-2],
                                           list(sample_variants[penultimate_variant].keys())[-1]: list(sample_variants[penultimate_variant].values())[-1]}

                estimated_fitness, estimated_t = estimate_trajectory_gradient_est_time(variant_trajectory, clone, sample_name, version) #estimate fitness and t

                guesses_based_on_mutation_number = initial_fitness_establishment_time(sample_name, diagnosis_age)
                estimated_fitness_based_on_mut_number = guesses_based_on_mutation_number[number_of_mutations]['fitness']
                estimated_t_based_on_mut_number = guesses_based_on_mutation_number[number_of_mutations]['establishment']

                if estimated_fitness_based_on_mut_number > estimated_fitness:
                    estimated_fitness = estimated_fitness_based_on_mut_number
                    estimated_t = estimated_t_based_on_mut_number

                if estimated_t <0:
                    estimated_t = 0

                if estimated_fitness <=0:
                    guesses_based_on_mutation_number = initial_fitness_establishment_time(sample_name, diagnosis_age)
                    estimated_fitness = guesses_based_on_mutation_number[number_of_mutations]['fitness']
                    estimated_t = guesses_based_on_mutation_number[number_of_mutations]['establishment']


        initial_clones[clone]={'fitness': estimated_fitness,
                              'est_time': estimated_t}

    return initial_clones, gradient_plots

def make_phylogeny_table_with_missing_drivers(sample_name, sample_variants, variants_ordered, sample_ages_lists, clonal_structure, clone_colors, variant_colors):
    #create a phylogeny table of the varaints, but excluding the missing drivers
    #record the colour used for each variant - if e.g. double-mutant clone when missing driver is included (e.g. clone = (X, DNMT3A), reassign it the colour of the single-mutant missing driver)

    clonal_structure_complete = [] #expand the clones which are only listed as e.g. triple-mutant drivers in to their predecessor single- and double-mutant clones

    for clone in clonal_structure:
        clonal_structure_complete.append(clone)
        clone_length = len(clone)
        for i in range(1, clone_length):
            subclone = clone[:-i]
            if subclone not in clonal_structure_complete:
                clonal_structure_complete.append(subclone)

#     print('clonal_structure_complete = ', clonal_structure_complete)

    #Assign colour to missing driver variants
    for k, v in clone_colors.items():
        if k[0] in ['X', 'Y', 'Z']:
            missing_driver = k[0]
            if len(k) == 1:
                variant_colors[missing_driver]=v
        if k[-1] in ['X', 'Y', 'Z']:
            missing_driver = k[-1]
            variant_colors[missing_driver] = v

    ### RECORD ANCESTORS AND DESCENDENTS OF EACH CLONE
    clone_direct_descendents = {'stem cells': []}
    clone_ancestors = {}
    clone_mutation_number = {}
    final_clones = {}

    for clone in clonal_structure_complete: #e.g. clonal_structure = [('TET2_p.S1593Afs*3', 'TET2_p.H1380D', 'SRSF2_p.P95H', 'JAK2_p.V617F'), ('GNAS_p.R844C',),....]
#         print('clone of interest = ', clone)
        number_mutations = len(clone) #e.g. 4
#         print('number mutations = ', number_mutations)
        if number_mutations in clone_mutation_number:
            clone_mutation_number[number_mutations]+=1
        else:
            clone_mutation_number[number_mutations]=1

        variant = clone[-1] #e.g. JAK2 V617F
#         print('variant defining clone = ', variant)
        if number_mutations == 1: #i.e. single mutant clone
            clone_direct_descendents['stem cells'].append(variant) #if single mutant clone, it is a direct descendant of the stem cells

        for clone_i in clonal_structure_complete: #iterate through the list of clones...
            if len(clone_i)>number_mutations: #i.e. possibly a descendant of the clone looking at
                if variant in clone_i: #if the final variant in the clone is in another clone, then the other clone is a descendent (not necessarily direct though)
    #                 print('descendent clone (clone i) = ', clone_i) #e.g. ('TET2_p.S1593Afs*3', 'TET2_p.H1380D', 'SRSF2_p.P95H', 'JAK2_p.V617F', 'chr9p_CNLOH')
                    final_variant = clone_i[-1] #e.g. 'chr9p_CNLOH' (final variant of the descendent clone)

                    if len(clone_i)==number_mutations+1: #i.e. direct descendent
                        if variant in clone_direct_descendents.keys():
                            clone_direct_descendents[variant].append(final_variant)
                        else:
                            clone_direct_descendents[variant]=[final_variant] #dictionary of the direct descendents of the variant of interest

                        #add the variant of interest as an ancestor of the descendent variant
                        clone_ancestors[final_variant]=variant
#                         print('clone ancestors = ', clone_ancestors)

        final_clones[variant]={'mutations_in_clone': number_mutations}

    ### ADD WILDTYPE (STEM CELL) CELL TO THE DICTIONARY
    final_clones['stem cells']={'mutations_in_clone': 0}

    #Create a phylogeny dictionary for each sample
    phylogeny_dictionary = {}

    for clone, info in final_clones.items():
#         print('clone = ', clone)
        #get parent information for each clone
        if info['mutations_in_clone']==0: #if stem cell
            clone_parent = ''
        else:
            if info['mutations_in_clone']==1: #if single mutant clone
                clone_parent = 'stem cells'
            else:
                clone_parent = clone_ancestors[clone]

        #get descendent (children) information for each clone
        descendents_list = []
        if clone in clone_direct_descendents.keys(): #look to see what the immediate descendent of the clone was
            descendents = clone_direct_descendents[clone] #immediate descendents
            for i in descendents: #add the immediate descendents to the descendents list
                descendents_list.append(i)
            for k, v in clone_direct_descendents.items(): #go back through the dictionary of immediate descendents
                if k in descendents_list: #see if any of the clones are clones that just added to descendents list
                    for i in v: #get those clones descendents too
                        descendents_list.append(i)

        #add further indirect descendents to the descendents list
        if len(descendents_list)>0:
            for descendent in descendents_list:
                for k, v in clone_direct_descendents.items():
                    if k in descendents_list:
                        for i in v:
                            if i not in descendents_list:
                                descendents_list.append(i)

        #get number of descendents in parent information for each clone (i.e. how many offspring did the parent have)
        if clone == 'stem cells':
            parent_descendents = 0
        else:
            if clone_parent == 'stem cells':
                parent_descendents = clone_mutation_number[1] #the number of single mutant clones in the phylogeny
            else:
                parent_descendents = len(clone_direct_descendents[clone_parent])
    #         print('parent descendents = ', parent_descendents)

        #get mutation_number_information_for_each_clone
        number_mutations = info['mutations_in_clone']
    #         print('number of mutations = ', number_mutations)

        variant_colors['stem cells'] = '#cccccc'

        phylogeny_dictionary[clone]={'parent': clone_parent,
                                    'descendents': descendents_list,
                                    'parent_descendents': parent_descendents,
                                    'mutations_in_clone': number_mutations,
                                    'variant color': variant_colors[clone]}

    for clone, phylogeny in phylogeny_dictionary.items():
        #get sibling information
        siblings = []
        parent = phylogeny['parent']
        if parent in clone_direct_descendents.keys():
            siblings_list = clone_direct_descendents[parent]
            for i in siblings_list:
                if i != clone:
                    siblings.append(i)

        phylogeny_dictionary[clone]['siblings']=siblings

    readable_phylogeny_dictionary = {}
    for clone, info in phylogeny_dictionary.items():
        clone_written = str(clone)
        readable_phylogeny_dictionary[clone_written]=info

    # Turn the phylogeny dictionary into a dataframe
    df_phylogeny = pd.DataFrame.from_dict(readable_phylogeny_dictionary, orient = 'index')
    df_phylogeny = df_phylogeny[['parent', 'siblings', 'descendents', 'parent_descendents', 'mutations_in_clone', 'variant color']]
    df_phylogeny = df_phylogeny.sort_values(by='mutations_in_clone', ascending = True)

    return df_phylogeny.to_csv('Data_files/'+sample_name+'_phylogenies_version_'+str(version)+'.csv')

def make_phylogeny_table_without_missing_drivers_for_muller(sample_name, sample_variants, variants_ordered, sample_ages_lists, clonal_structure, clone_colors, variant_colors):
    #create a phylogeny table of the varaints, but excluding the missing drivers
    #record the colour used for each variant - if e.g. double-mutant clone when missing driver is included (e.g. clone = (X, DNMT3A), reassign it the colour of the single-mutant missing driver)

    new_variant_colors = {}
    for k, v in variant_colors.items():
        new_variant_colors[k]=v

    #Calculating the cell fractions for each clone
    final_clone_cell_fractions = {}
    age_first_detected = {}

    #convert VAFs of each variant in to cell fractions (and leave cell fractions as cell fractions)
    variant_cell_fractions = {} #e.g. {'SF3B1_p.K700E': {73.73: 16.04, 75.0: 16.39....}, 'DNMT3A_p.Q606X': {73.73: 15.49, 75.0: 19.56....}}
    for k, v in sample_variants.items(): #e.g. {'SF3B1_p.K700E': {73.73: 0.080191502, 75.0: 0.081956378...}, 'DNMT3A_p.Q606X': {73.73: 0.077450585, 75.0: ....}}
        trajectory = {}
        n = 0
        for age, VAF in v.items():
            if 'chr' in k: #i.e. mCA
                trajectory[age]=round((VAF)*100, 2)
            else:
                trajectory[age]=round((VAF*2)*100, 2)
            if n ==0:
                age_first_detected[k]=age
            n+=1
        variant_cell_fractions[k]=trajectory

#     print('sample_variants = ', sample_variants)
#     print()
#     print('variant cell fractions = ', variant_cell_fractions)

    clones_without_missing_drivers = []
    clones_with_first_driver_missing = {}

    for clone in clonal_structure:
    #         print('clone = ', clone)
        final_variant = clone[-1]
        first_variant = clone[0]
        if final_variant in ['X', 'Y', 'Z']:
            print('exclude entire clone with final missing driver')
        else:
            if first_variant in ['X', 'Y', 'Z']:
#                 print('exclude entire clone with missing driver as first mutation')
                if len(clone)==2: #i.e. just clone is just e.g. ('X', 'DNMT3A'), retrieve the color of the single-mutant missing driver
                    for k, v in clone_colors.items():
                        if first_variant in k:
                            if len(k)==1:
                                missing_single_driver_color = v
                                new_single_driver = clone[-1]
                                new_variant_colors[clone[-1]] = missing_single_driver_color #re-assign the missing driver colour to the clone whose missing driver now removed
            clone = tuple([x for x in clone if x not in ['X', 'Y', 'Z']]) #exclude missing driver from any clones
            clones_without_missing_drivers.append(clone)
#     print('clones without missing drivers = ', clones_without_missing_drivers)

#     print('new variant colors = ', new_variant_colors)
#     print('old variant colors = ', variant_colors)

    clonal_structure_complete = [] #expand the clones which are only listed as e.g. triple-mutant drivers in to their predecessor single- and double-mutant clones

    for clone in clones_without_missing_drivers:
        clonal_structure_complete.append(clone)
        clone_length = len(clone)
        for i in range(1, clone_length):
            subclone = clone[:-i]
            if subclone not in clonal_structure_complete:
                clonal_structure_complete.append(subclone)

    ### CALCULATE THE CELL FRACTION FOR EACH CLONE
    clone_direct_descendents = {'stem cells': []} #just direct descendents
    clone_ancestors = {} #direct ancestor
    clone_mutation_number = {}

    for clone in clonal_structure_complete: #e.g. clonal_structure = [('TET2_p.S1593Afs*3', 'TET2_p.H1380D', 'SRSF2_p.P95H', 'JAK2_p.V617F'), ('GNAS_p.R844C',),....]
        number_mutations = len(clone) #e.g. 4
        if number_mutations in clone_mutation_number:
            clone_mutation_number[number_mutations]+=1
        else:
            clone_mutation_number[number_mutations]=1

        variant = clone[-1] #e.g. JAK2 V617F
        if number_mutations == 1: #i.e. single mutant clone
            clone_direct_descendents['stem cells'].append(variant) #if single mutant clone, it is a direct descendant of the stem cells
        clone_CFs = variant_cell_fractions[variant].copy() #cell fractions for the final variant in the clone
    #     print('clone cell fractions for variant defining clone = ', clone_CFs)

        for clone_i in clonal_structure_complete: #iterate through the list of clones...
            if len(clone_i)>number_mutations: #i.e. possibly a descendant of the clone looking at
                if variant in clone_i: #if the final variant in the clone is in another clone, then the other clone is a descendent (not necessarily direct though)
                    # print('descendent clone (clone i) = ', clone_i) #e.g. ('TET2_p.S1593Afs*3', 'TET2_p.H1380D', 'SRSF2_p.P95H', 'JAK2_p.V617F', 'chr9p_CNLOH')
                    final_variant = clone_i[-1] #e.g. 'chr9p_CNLOH' (final variant of the descendent clone)

                    if len(clone_i)==number_mutations+1: #i.e. direct descendent
                        if variant in clone_direct_descendents.keys():
                            clone_direct_descendents[variant].append(final_variant)
                        else:
                            clone_direct_descendents[variant]=[final_variant] #dictionary of the direct descendents of the variant of interest

                        #add the variant of interest as an ancestor of the descendent variant
                        clone_ancestors[final_variant]=variant

    #                     print('descendent clone defining variant = ', final_variant)
                        clone_i_CFs = variant_cell_fractions[final_variant] #e.g. cell fractions for chr9p_CNLOH
    #                     print('descendent clone cell fractions = ', clone_i_CFs)
                        for age, cell_fraction in clone_CFs.items(): #e.g. cell fractions for the JAK2 V617F variant
                            if age in clone_i_CFs.keys(): #i.e. if detected at timepoint in descendent (chr9p_CNLOH)
                                descendent_CF = clone_i_CFs[age]
                                if final_variant in ['chr9p_CNLOH', 'chr4q_CNLOH']:
                                    if final_variant == 'chr9p_CNLOH': #i.e. if the descendent is 9p CNLOH in a JAK2 clone
                                        if variant == 'JAK2_p.V617F':
                                            JAK2_clone_VAF = cell_fraction/2
                                            JAK2_alone_VAF = JAK2_clone_VAF-descendent_CF #i.e. JAK2 VAF - 9p cell fraction
                                            parent_minus_descendent_CF = JAK2_alone_VAF*2
                                    if final_variant == 'chr4q_CNLOH':
                                        if 'TET2' in variant:
                                            TET2_clone_VAF = cell_fraction/2
                                            TET2_alone_VAF = TET2_clone_VAF-descendent_CF
                                            parent_minus_descendent_CF = TET2_alone_VAF*2
                                else:
                                    parent_minus_descendent_CF = cell_fraction-descendent_CF
    #                                     print('clone minus descendents cell fraction at age '+str(age)+' = '+str(parent_minus_descendent_CF))
                                clone_CFs[age]= parent_minus_descendent_CF
    #                     print('clone CFs = ', clone_CFs)
            # else:
            #     print('clone_i not a descendent of clone of interest')

        final_clone_cell_fractions[variant]={'mutations_in_clone': number_mutations, 'trajectory': clone_CFs}

    ### ADD WILDTYPE (STEM CELL) CELL FRACTIONS TO THE DICTIONARY
    samples_ages_sequences = sample_ages_lists[sample_name]

    final_clone_cell_fractions_with_WT = {}

    total_cell_fraction_each_age = {}
    for age in samples_ages_sequences:
    #     print(age)
        total_cell_fraction_at_age = 0
        for variant, trajectory in final_clone_cell_fractions.items():
            if age in trajectory['trajectory'].keys():
                total_cell_fraction_at_age+=trajectory['trajectory'][age]
                if variant in final_clone_cell_fractions_with_WT.keys():
                    final_clone_cell_fractions_with_WT[variant]['trajectory'][age]=trajectory['trajectory'][age]
                else:
                    final_clone_cell_fractions_with_WT[variant]={'mutations_in_clone': final_clone_cell_fractions[variant]['mutations_in_clone'],
                                                                 'trajectory': {age: trajectory['trajectory'][age]},
                                                                'age first detected': age_first_detected[variant]}

    #         print('total cell fraction at age = '+str(age)+'= '+str(total_cell_fraction_at_age))
        WT_cell_fraction = round((100-total_cell_fraction_at_age), 2)
        if 'stem cells' in final_clone_cell_fractions_with_WT.keys():
            final_clone_cell_fractions_with_WT['stem cells']['trajectory'][age] = WT_cell_fraction
        else:
            final_clone_cell_fractions_with_WT['stem cells']={'mutations_in_clone': 0, 'trajectory': {age: WT_cell_fraction},
                                                             'age first detected': 0}

    #     print(final_clone_cell_fractions_with_WT)

    #Create a phylogeny dictionary for each sample
    phylogeny_dictionary = {}

    for clone, info in final_clone_cell_fractions_with_WT.items():
#         print('clone = ', clone)
        #get parent information for each clone
        if info['mutations_in_clone']==0: #if stem cell
            clone_parent = ''
        else:
            if info['mutations_in_clone']==1: #if single mutant clone
                clone_parent = 'stem cells'
            else:
                clone_parent = clone_ancestors[clone]
    #         print('parent = ', clone_parent)

        #get descendent (children) information for each clone
        descendents_list = []
        if clone in clone_direct_descendents.keys(): #look to see what the immediate descendent of the clone was
            descendents = clone_direct_descendents[clone] #immediate descendents
            for i in descendents: #add the immediate descendents to the descendents list
                descendents_list.append(i)
            for k, v in clone_direct_descendents.items(): #go back through the dictionary of immediate descendents
                if k in descendents_list: #see if the direct descenent of this clone has any descendents
                    for i in v: #get those clones descendents too
                        descendents_list.append(i)

        #add further indirect descendents to the descendents list
        if len(descendents_list)>0:
            for descendent in descendents_list:
                for k, v in clone_direct_descendents.items():
                    if k in descendents_list:
                        for i in v:
                            if i not in descendents_list:
                                descendents_list.append(i)

        #get number of descendents in parent information for each clone (i.e. how many offspring did the parent have)
        if clone == 'stem cells':
            parent_descendents = 0
        else:
            if clone_parent == 'stem cells':
                parent_descendents = clone_mutation_number[1] #the number of single mutant clones in the phylogeny
            else:
                parent_descendents = len(clone_direct_descendents[clone_parent])
    #         print('parent descendents = ', parent_descendents)

        #get mutation_number_information_for_each_clone
        number_mutations = info['mutations_in_clone']
    #         print('number of mutations = ', number_mutations)

        cell_fractions = []
        for age in samples_ages_sequences:
            if age in info['trajectory']:
                cell_fraction = info['trajectory'][age]
                if cell_fraction <0:
                    cell_fraction = 0
                cell_fractions.append(cell_fraction)
            else:
                cell_fractions.append(0)

        max_cell_fraction = max(cell_fractions)

        new_variant_colors['stem cells'] = '#cccccc'

        phylogeny_dictionary[clone]={'parent': clone_parent,
                                    'descendents': descendents_list,
                                    'parent_descendents': parent_descendents,
                                    'mutations_in_clone': number_mutations,
                                    'cell_fractions': cell_fractions,
                                    'age first detected': info['age first detected'],
                                    'max cell fraction': max_cell_fraction,
                                    'variant color': new_variant_colors[clone]}

    for clone, phylogeny in phylogeny_dictionary.items():
        #get sibling information
        siblings = []
        parent = phylogeny['parent']
        if parent in clone_direct_descendents.keys():
            siblings_list = clone_direct_descendents[parent]
            for i in siblings_list:
                if i != clone:
                    siblings.append(i)

        phylogeny_dictionary[clone]['siblings']=siblings

    readable_phylogeny_dictionary = {}
    for clone, info in phylogeny_dictionary.items():
        clone_written = str(clone)
        readable_phylogeny_dictionary[clone_written]=info

    # Turn the phylogeny dictionary into a dataframe
    df_phylogeny = pd.DataFrame.from_dict(readable_phylogeny_dictionary, orient = 'index')
    df_phylogeny = df_phylogeny[['parent', 'siblings', 'descendents', 'parent_descendents', 'mutations_in_clone',
                                 'cell_fractions', 'age first detected', 'max cell fraction', 'variant color']]
    df_phylogeny = df_phylogeny.sort_values(by=['mutations_in_clone', 'age first detected', 'max cell fraction'], ascending = [True, True, False])

    return df_phylogeny.to_csv('Data_files/'+sample_name+'_phylogenies_without_missing_drivers_for_muller_v'+str(version)+'.csv')

def main():
    # Parameters to be input.
    parser = ArgumentParser()
    parser.add_argument("--sample_name", action="store", dest="sample_name", help="name of the person whose trajecories you want to analyse", required=True)
    parser.add_argument("--seeds", action="store", type=int, dest="seeds", help="number of times to run for the optimiser", required=True)
    o = parser.parse_args()

    sample_name = o.sample_name
    seeds = o.seeds

    T=1000 #total timesteps
    dt=0.1 #time interval i.e. 0.1 = 10% of 1 year (1000 timesteps = 100 years)
    # effective_depth=1800

    print()
    print()
    print('STARTING ANALYSIS FOR SAMPLE '+sample_name+' WITH '+str(seeds)+' SEEDS')

    #Create dictionaries of sample information
    sample_ages, ages_dict_names, sample_ages_lists, diagnosis_age = dictionary_sample_details(sample_name)

    #Read in the dataframe of variant trajectories
    df = pd.read_csv("Data_files/UKCTOCS_watson_non-germline_variants_calls_SNVs_indels_mCAs_April_2024.csv")
    sample_variants, variant_details = sample_trajectories(sample_name, df)
    bounded_sample_variants, all_trajectories_with_bounded = create_bounded_variant_trajectories(sample_name, sample_variants, sample_ages_lists, ages_dict_names, variant_details)

    #Read in the dataframe of germline variants
    df_germline = pd.read_csv("Data_files/UKCTOCS_watson_non-germline_variants_calls_SNVs_indels_mCAs_April_2024.csv")

    #Read in the clonal structure dataframe
    clonal_structure, manual_initial_guesses, coarse_optimisation_attempts, upper_freq = inferred_clonal_structure(sample_name)
    print('upper freq = ', upper_freq)
    print()
    print('clonal_structure = ', clonal_structure)
    print()
    initial_guess_clones, gradient_plots = create_initial_guess_clones_dictionary(sample_name, sample_variants, all_trajectories_with_bounded, clonal_structure, diagnosis_age, version)
    print('initial guess clones = ', initial_guess_clones)

    for k, v in initial_guess_clones.items():
        if k in manual_initial_guesses.keys():
            initial_guess_clones[k]={'fitness': manual_initial_guesses[k]['fitness'], 'est_time': manual_initial_guesses[k]['est_time']}

    print('amended initial guess clones = ', initial_guess_clones)

    variant_depths_by_age = get_effective_depths_for_variant_each_timepoint(sample_name, ages_dict_names, clonal_structure, variant_details)

    # Run the optimiser
    #for choosing a new fitness from random normal distribution centered around old establishment time, with standard deviation of previous fitness estimate multiplied by....
    coarse_tune_fitness_factor = 0.1
    moderate_tune_fitness_factor = 0.01
    fine_tune_fitness_factor = 0.001

    fitness_adjustment_list = [coarse_tune_fitness_factor, moderate_tune_fitness_factor, fine_tune_fitness_factor]

    #for choosing a new establishment time from random normal distribution centered around old establishment time, with standard deviation of....
    coarse_age = 1
    moderate_age = 0.2
    fine_age = 0.04

    est_time_adjustment_list = [coarse_age, moderate_age, fine_age]

    #number of optimisation steps
    moderate_steps = 2000
    fine_steps = 1000

    if coarse_optimisation_attempts == 0: #i.e. if not manually set in the clonal composition/ optimisation file
        coarse_steps = 5000
    else:
        coarse_steps = coarse_optimisation_attempts

    print('coarse steps = ', coarse_steps)

    steps_list = [coarse_steps, moderate_steps, fine_steps]

    optimised_seeds, optimised_clones, fitness_est_time = clone_optimiser_seeds(initial_guess_clones, T, dt,
                                                                                sample_variants, all_trajectories_with_bounded,
                                                                                bounded_sample_variants, sample_ages_lists, upper_freq,
                                                                                sample_name, fitness_adjustment_list, est_time_adjustment_list,
                                                                                steps_list, seeds, diagnosis_age, variant_depths_by_age)

    # #Plot the trajectories - in terms of VAF/cell fraction
    # clone_colors, variant_colors, save_germline_table, variants_ordered = plot_trajectories(sample_name, optimised_clones, T, dt, sample_variants, bounded_sample_variants, sample_ages, sample_ages_lists, diagnosis_age, upper_freq, seeds, multiple_mutant_clone_colors, df_germline, 'plot_bounded', variant_depths_by_age)
    # clone_colors, variant_colors, save_germline_table, variants_ordered = plot_trajectories(sample_name, optimised_clones, T, dt, sample_variants, bounded_sample_variants, sample_ages, sample_ages_lists, diagnosis_age, upper_freq, seeds, multiple_mutant_clone_colors, df_germline, 'not_plot_bounded', variant_depths_by_age)
    # plot_trajectories_small(sample_name, optimised_clones, T, dt, sample_variants, bounded_sample_variants, sample_ages, sample_ages_lists, diagnosis_age, upper_freq, seeds, multiple_mutant_clone_colors, df_germline, variant_depths_by_age)

    #Plot the trajectories - in terms of cell fraction
    clone_colors, variant_colors, save_germline_table, variants_ordered = plot_trajectories_cell_fractions(sample_name, optimised_clones, T, dt, sample_variants, bounded_sample_variants, sample_ages, sample_ages_lists, diagnosis_age, upper_freq, seeds, multiple_mutant_clone_colors, df_germline, 'plot_bounded', variant_depths_by_age)
    clone_colors, variant_colors, save_germline_table, variants_ordered = plot_trajectories_cell_fractions(sample_name, optimised_clones, T, dt, sample_variants, bounded_sample_variants, sample_ages, sample_ages_lists, diagnosis_age, upper_freq, seeds, multiple_mutant_clone_colors, df_germline, 'not_plot_bounded', variant_depths_by_age)
    plot_trajectories_cell_fractions_small(sample_name, optimised_clones, T, dt, sample_variants, bounded_sample_variants, sample_ages, sample_ages_lists, diagnosis_age, upper_freq, seeds, multiple_mutant_clone_colors, df_germline, variant_depths_by_age)

    #Plot the seed likelihoods
    final_seed_likelihoods, seed_with_best_likelihood, best_likelihood = plot_seed_results(optimised_seeds, sample_name, seeds, clone_colors, diagnosis_age)

    #Save an image of the germline DataFrame
    dfi.export(save_germline_table,'Data_files/'+sample_name+'_germline_dataframe_v'+str(version)+'.png')
    #Save a PDF image of the germline DataFrame
    pdf = FPDF()
    pdf.add_page()
    pdf.image('Data_files/'+sample_name+'_germline_dataframe_v'+str(version)+'.png', w = 150)
    pdf.output('Data_files/'+sample_name+'_germline_dataframe_v'+str(version)+'.pdf', "F")

    #Save a combined PDF file of all the images generated for this sample
    gradients = []
    if len(gradient_plots)>0:
        for clone in clonal_structure:
            if clone in gradient_plots:
                clone_save_name = str(clone).replace('(', '').replace(')', '').replace("'", '').replace('_', ' ').replace("p.", '').replace("*", '').replace(" ", '_').replace(",", '')
                try:
                    gradients.append('Data_files/'+sample_name+'_'+clone_save_name+'_gradient_estimation_v'+str(version)+'.pdf')
                except:
                    print('no gradient plot found for clone '+clone_save_name)

    pdfs = ['Data_files/'+sample_name+'_optimiser_results_'+str(seeds)+'_seeds_v'+str(version)+'',
           'Data_files/'+sample_name+'_seed_results_'+str(seeds)+'_seeds_v'+str(version)+'.pdf',
           'Data_files/'+sample_name+'_trajectories_'+str(seeds)+'_seeds_bounded_plotted_cell_fractions_v'+str(version)+'.pdf',
          'Data_files/'+sample_name+'_trajectories_'+str(seeds)+'_seeds_cell_fractions_v'+str(version)+'.pdf',
          'Data_files/'+sample_name+'_trajectories_small_'+str(seeds)+'_seeds_cell_fractions_v'+str(version)+'.pdf',
           'Data_files/'+sample_name+'_germline_dataframe_v'+str(version)+'.pdf']

    merger = PdfMerger()

    if len(gradients)>0:
        for gradient_pdf in gradients:
            merger.append(gradient_pdf)

    for pdf in pdfs:
        merger.append(pdf)

    merger.write('Data_files/'+sample_name+'_combined_PDF_'+str(seeds)+'_seeds_v'+str(version)+'.pdf')
    merger.close()

    #Write output files
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    fitnesses = {}
    tau_times = {}
    establishment_times = {}
    for k, v in fitness_est_time.items():
        fitnesses[k]=v['fitness']
        tau_times[k]=v['est_time']
        establishment_times[k]=v['establishment_time']

    optimiser_results_file = open('Data_files/Optimiser_results.txt', 'a') #'a' means append to the file, rather than overwrite it
    optimiser_results_file.write(dt_string+'\t'+sample_name+'\t'+str(seeds)+'\t'+str(final_seed_likelihoods)+'\t'+str(seed_with_best_likelihood)+'\t'+str(best_likelihood)+'\t'+str(fitnesses)+'\t'+str(tau_times)+'\t'+str(establishment_times)+'\t'+str(variant_colors)+'\t'+str(clone_colors)+'\t'+version+'\n')
    optimiser_results_file.close()

    #Write a csv file of the phylogeny, in order to later make Muller plot
    make_phylogeny_table_with_missing_drivers(sample_name, sample_variants, variants_ordered, sample_ages_lists, clonal_structure, clone_colors, variant_colors)
    make_phylogeny_table_without_missing_drivers_for_muller(sample_name, sample_variants, variants_ordered, sample_ages_lists, clonal_structure, clone_colors, variant_colors)

    print('Best likelihood = '+str(best_likelihood))
    print('Inferred fitnesses = '+str(fitnesses))
    print('Inferred establishment times = '+str(establishment_times))

    print('Optimiser for '+sample_name+' with '+str(seeds)+' seeds completed in %s minutes' % round((time.time() - start_time)/60, 3))
    return

if __name__ == "__main__":
	main()
