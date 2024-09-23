#ifndef __TENSOR_PERMUTE_H__
#define __TENSOR_PERMUTE_H__

#include <stddef.h>
#include <stdint.h>
#include <limits.h>

int permute(const void* src, void* dst, uint64_t dtypeSize, uint64_t* src_dims,
    uint64_t src_ndim, uint64_t* permute_idx, uint64_t* dst_dims, int nThreads = 1);
/**
 * @brief This file provides an optimized tensor permutation method, under the function Permute()
 * It contains 4 optimization steps.
 * Step 1. Sqeeze the trivial dimensions of width = 1. It takes O(n) time and space complexity.
 * EX.   perm_idx[2, 3, 5, 4, 1, 7, 0,  6]  ==>  [1, 2, 3, 0, 5, 4]
 *       src_nums[8, 9, 1, 6, 4, 6, 1, 10]  ==>  [8, 9, 6, 4, 6, 10]
 *
 * Step 2. Compress the consecutive permuted dimensions. It takes O(n) time and space complexity.
 * EX.   perm_idx[2, 3,  4,  5,    0, 1,   6, 7 ]  ==>  [1,           0,    2 ]
 *       src_nums[8, 9,  12, 6,    4, 6,   9, 10]  ==>  [8*9,  12*6*4*6,  9*10]
 * 
 * Step 3. In the case where the last dim is unmuted, treat the last small dim, such that, dtypeSize*dim<=8, as a single data type 
 *         and then remove the last dim.
 * EX.   perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]
 *       trim_dims[20, 8, 16, 12, 7]   ==> trim_dims[20, 8, 16, 12]
 *       dtypeSize = 1                 ==> dtypeSize = 7
 *
 * Step 4. Apply the proposed generalized batch transpose over the case perm_idx[ndim-1] != ndim-1;
 *         Apply memcpy() to move the entire last dim of data over the case perm_idx[ndim-1] == ndim-1.
 *         It is worth noting that both operations involve nearly no multimplication and completely no division to compute permuted indexes
 *            as opposed to the straightforword method.
 *
 * By default, it utilizes single thread. Multi-thread is activated by setting the parameter nThreads >1.
 * By partitioning evenly along the first dim, i.e., [0], the proposed multi-thread operations literally linearly reduces the running time,
 * until it saturates the memory bandwidth.
 */

 /* permute_validation() uses the randomly generated tensors and permutations to test against the equality
 *  permute(tensor, perm_idx) -> permuted_tensor
 *  permute(permuted_tensor, perm_inv) = tensor, where perm_inv is the inverse array of perm_idx.
 *  The first set of tensor dimension and perm_idx is hand designed to test a particular corner case
 *  whereas the remaining test cases are randomly generated to expand coverage. 
 */
void permute_validation();

// measure_permute() measures the efficiency of the proposed permute()
void measure_permute();

#endif