import numpy as np 
import msprime
from scipy import linalg, stats
import utils
import matplotlib.pyplot as plt 
import seaborn as sns 
import pysam
import pandas as pd
from typing import Callable

np.random.seed(42)

def rsvd(X, dstar, power_iters=2, delta=10):
	""" Perform rsvd algorithm on input matrix.
		Method must be supplied dstar.
		Returns truncated svd (U,S,V).
	Parameters
	----------
	X : int matrix
    	Matrix of n x m integers, where m <= n. If n < m,
    	matrix will be transposed to enforce m <= n.
   	dstar : int
   		The latent (underlying) matrix rank that will be
   		used to truncate the larger dimension (m).
   	power_iters : int
   		default: 2
   		Number of power iterations used (random matrix multiplications)
   	delta : int
   		default: 10
   		oversampling parameter (to improve numerical stability)
    Returns
	-------
	int matrix
    	Matrix of left singular vectors.
    int matrix
    	Matrix of singular values.
    int matrix
    	Matrix of right singular vectors.
    """
	transpose = False 
	if X.shape[0] < X.shape[1]:
		X = X.T 
		transpose = True 

	if power_iters < 1:
		power_iters = 1

	# follows manuscript notation as closely as possible
	P = np.random.randn(X.shape[0],dstar+delta)	
	for i in range(power_iters):
		P = np.dot(X.T,P)
		P = np.dot(X,P)
	Q,R = np.linalg.qr(P)
	B = np.dot(Q.T,X)
	U,S,V = linalg.svd(B)
	U = np.dot(Q, U)

	# Remove extra dimensionality incurred by delta
	U = U[:, 0:dstar]
	S = S[0:dstar]

	return (V.T, S, U.T) if transpose else (U, S, V)


def stabilityMeasure(X, d_max, B=5, power_iters=2):
	""" Calculate stability of 
	Parameters
	----------
	X : int matrix
		input matrix to determine rank of
	d_max : int
		upper bound rank to estimate
	B : int
		default: 5
		number of projections to correlate
	power_iters : int
		default: 2
   		Number of power iterations used (random matrix multiplications)
	Returns
	-------
	int
		Latent (lower-dimensional) matrix rank
	"""
	singular_basis = np.zeros((B,X.shape[0],d_max))
	# calculate singular basis under multiple projections
	for i in range(B):
		U = rsvd(X,d_max)[0]
		singular_basis[i,:,:] = U[:,0:d_max]

	# calculate score for each singular vector
	stability_vec = np.zeros((d_max))
	for k in range(d_max):
		stability = 0
		for i in range(0,B-1):
			for j in range(i+1,B):
				corr = stats.spearmanr(singular_basis[i,:,k],singular_basis[j,:,k])[0]
				stability = stability + abs(corr)
		N = B*(B-1)/2
		stability = stability/N
		stability_vec[k] = stability

	# wilcoxon rank-sum test p-values
	p_vals = np.zeros(d_max-2)
	for k in range(2,d_max):
		p_vals[k-2] = stats.ranksums(stability_vec[0:k-1],stability_vec[k-1:d_max])[1]

	dstar = np.argmin(p_vals)
	
	return dstar

def pca_arsvd(X, k, dstar):
    transpose = False 
    # Check if transposing is necessary
    if X.shape[0] < X.shape[1]:
        X = X.T 
        transpose = True 
        VT, _, _ = rsvd(X, dstar)
    else:
        _, _, V = rsvd(X, dstar)

    
    # Compute top k pcs
    pcs = []

    if transpose:
        for i in range(min(dstar, k)):
            pc = VT[:, i]
            pcs.append(pc)
    else:
        for i in range(min(dstar, k)):
            pc = V[i, :]
            pcs.append(pc)
    
    return np.array(pcs).T

def center_X(X: np.array):
    """ 
    Returns a centered version of X so that each column has mean 0.

    Input:
    - X (2D numpy array): dataset

    Returns:
    - Xtilde, centered version of X
    """
    return X - np.mean(X, axis = 0)

def normalize_X(X: np.array):
    """ 
    Normalizes X so each column has variance 1.

    Input: 
    - X (2D numpy array): dataset (assumed to be centered)

    Returns: 
    - Xtilde, normalized version of X
    """
    sd = np.std(X, axis = 0)
    return X/sd
    

def fastpca(G, k: int, I: int = 10):
    """ 
    Our implementation of the FastPCA algorithm in Python. 

    Inputs:
    - G (2D numpy array): the genotype matrix
    - k (int): the desired number of principal components
    - I (int): the number of iterations

    Returns:
    - top k principal components
    """
    n, p = G.shape[0], G.shape[1]
    # Normalize G
    Y = center_X(G)
    Y = normalize_X(Y)

    # Generate random matrix P0
    l = 2 * k # as suggested by the Galinsky paper
    P_i = np.random.randn(p, l) #P_0 in the paper; initializing step

    H_matrices = []

    # Update rule: H_i = Y*P_i, P_{i+1} = 1/n * Y^T * H_i
    for _ in range(I):
        H_i = Y @ P_i
        P_i = 1/n * Y.T @ H_i 
        H_matrices.append(H_i)

    H = np.concatenate(H_matrices, axis = 1)

    # Compute SVD of H
    U_H, _, _ = np.linalg.svd(H)

    T = U_H.T @ Y

    _, _, VT_T = np.linalg.svd(T) 

    return VT_T[:k, :].T    


def benchmark_1000_genomes(f: Callable, k: int, data: np.array, *args):
    """ 
    Benchmarks performance of an algorithm on the given datasets.

    Inputs:
    - f (Callable): the PCA algorithm
        - should take in an argument for data and an argument for number of principal components
    - k (int): the desired number of principal components
    - data (2D numpy array): the dataset

    Returns:
    - accuracy metric
    """
    pcs = f(data, k, *args)
    return utils.compute_prop_variance(pcs, data)


if __name__ == "__main__":
    # Load data
    G = np.load("processed_data/real/ALL.chr21.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.npz")['arr_0']
    G = np.array(G, dtype = np.float16)

    # Values for k
    k_vals = [2, 3, 5, 10]
    num_repetitions = 2

    # Benchmarking 
    arsvd_benchmark = []
    fastpca_benchmark = []

    for i in range(num_repetitions):
        print(f"Starting repetition {i + 1}")
        arsvd_benchmark_totals = []
        fastpca_benchmark_totals = []
        for k in k_vals:
            arsvd_result = benchmark_1000_genomes(pca_arsvd, k, G, 10)
            arsvd_benchmark_totals.append(arsvd_result)

            fastpca_result = benchmark_1000_genomes(fastpca, k, G)
            fastpca_benchmark_totals.append(fastpca_result)
        arsvd_benchmark.append(arsvd_benchmark_totals)
        fastpca_benchmark.append(fastpca_benchmark_totals)
    
    # Save benchmark values
    np.save("../output/1000_genomes_arsvd.npy", arsvd_benchmark)    
    np.save("../output/1000_genomes_fastpca.npy", fastpca_benchmark)    