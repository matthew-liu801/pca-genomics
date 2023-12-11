import numpy as np 
import pysam
from typing import List
import msprime

# Functions to filter for quality

def filter_maf(records: List[pysam.VariantRecord], threshold: float = 0.01) -> List[pysam.VariantRecord]:
    """ 
    Filters out SNVs with a minor allele frequency below a threshold.

    Inputs:
    - records (List[pysam.VariantRecord]): a list of genetic variants
    - threshold (float): the desired rejection threshold

    Returns:
    - filtered_records (List[pysam.VariantRecord]): a list of genetic variants with MAF >= threshold
    """
    filtered_records = []
    for record in records:
        ref_count = sum(sample['GT'][0] for sample in record.samples.values())
        alt_count = sum(sample['GT'][1] for sample in record.samples.values())
        total = 2 * len(record.samples)
        maf = min(ref_count, alt_count) / total
        if maf >= threshold:
            filtered_records.append(record)
    return filtered_records

def filter_missingness(records: List[pysam.VariantRecord], threshold: float = 0.02):
    """ 
    Filters out SNVs with missing data above a certain threshold.

    Inputs:
    - records (List[pysam.VariantRecord]): a list of genetic variants
    - threshold (float): the desired rejection threshold

    Returns:
    - filtered_records (List[pysam.VariantRecord]): a list of variants with % missing data <= threshold
    """
    filtered_records = []
    for record in records:
        missing_count = sum(1 for sample in record.samples.values() if None in sample['GT'])
        if missing_count / len(record.samples) <= threshold:
            filtered_records.append(record)
    return filtered_records

def filter_quality(records, quality_threshold=30):
    """ 
    Filters out SNVs whose data quality is below a certain threshold.

    Inputs:
    - records (List[pysam.VariantRecord]): a list of genetic variants
    - threshold (float): the desired rejection threshold

    Returns:
    - filtered_records (List[pysam.VariantRecord]): a list of variants with data quality >= threshold
    """
    filtered_records = []
    for record in records:
        if record.filter.keys() == ['PASS'] and record.qual >= quality_threshold:
            filtered_records.append(record)
    return filtered_records

def simulate_diploid_genotypes(sample_size, sequence_length, mutation_rate, recombination_rate):
    """
    Simulate diploid genotype data using msprime.
    
    Parameters:
    - sample_size: The number of sampled diploid individuals.
    - sequence_length: The length of the sequence to simulate.
    - mutation_rate: The mutation rate per base pair per generation.
    - recombination_rate: The recombination rate per base pair per generation.
    
    Returns:
    - Genotype matrix of shape (number of variants, sample_size)
    """
    tree_sequence = msprime.simulate(
        sample_size=2 * sample_size,  # Account for both copies of the chromosome
        length=sequence_length,
        mutation_rate=mutation_rate,
        recombination_rate=recombination_rate
    )
    
    # Extract genotype data
    haploid_genotypes = tree_sequence.genotype_matrix()
    
    # Convert haploid data to diploid data
    diploid_genotypes = np.zeros((haploid_genotypes.shape[0], sample_size), dtype=int)
    for i in range(sample_size):
        diploid_genotypes[:, i] = haploid_genotypes[:, 2*i] + haploid_genotypes[:, 2*i + 1]
    
    return diploid_genotypes.T

def count_genotypes(G: np.array):
    """ 
    Counts the percentages of genotypes corresponding to 0, 1, 2 (homozygous dominant, heterozygous, homozygous recessive).

    Input:
    - G (2D numpy array): matrix of genotypes

    Returns:
    - (zeros, ones, twos): tuple of floats corresponding to percentages of each genotype
    """
    (m, n) = G.shape

    zeros = np.count_nonzero(G == 0)/(m * n)
    ones = np.count_nonzero(G == 1)/(m * n)
    twos = np.count_nonzero(G == 2)/(m * n)

    return (zeros, ones, twos)

def compute_prop_variance(lambda_i: float, S: np.array):
    """ 
    Computes the proportion of total variance explained by the ith principal component.

    Inputs:
    - S (2D numpy array): covariance matrix 
    - lambda_i (float): ith eigenvalue of S

    Returns:
    - proportion of total variance explained by principal component i
    """
    return lambda_i/np.trace(S)

def compute_pc_from_svd(i: int, A: np.array, VT: np.array):
    """ 
    Computes the ith principal component of A given its right singular vectors.

    Inputs:
    - i (int): the index of the desired PC
    - A (2D numpy array): the matrix to compute the PC for
    - VT (2D numpy array): the matrix containing the right singular vectors of A
    """
    return A @ VT[i]