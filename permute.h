#ifndef __TENSOR_PERMUTE_H__
#define __TENSOR_PERMUTE_H__

#include <stddef.h>
#include <stdint.h>
#include <limits.h>

/**
 * @brief This file provides an optimized tensor permutation method, under the function Permute()
 * It contains 4 optimization steps.
 * Step 1. Squeeze the trivial dimensions of width = 1. It takes O(n) time and space complexity.
 * EX.   perm_idx[2, 3, 5, 4, 1, 7, 0,  6]  ==>  [2, 4, 3, 1, 6, 0,  5]
 *       src_nums[8, 9, 1, 6, 4, 6, 9, 10]  ==>  [8, 9, 6, 4, 6, 9, 10]
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
 * EX.   perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]
 *       trim_dims[20, 8, 16, 12, 16]   ==> trim_dims[20, 8, 16, 12]
 *       dtypeSize = 1                 ==> dtypeSize = 16
 *
 * Step 4. Case 1. The last dim is permuted, i.e., perm_idx[ndim-1] != ndim-1.
 *   Fundamentally, such permutation can be viewed as a generalized transpose. We thus propose an innovative generalized batch transpose technique,
 *   which effectively takes advantage of the cache line by creating column-wise consecutive write addresses.
 *   For special dtypeSize in {3, 5, 6, 7} (which is created from the merging of the small last dim), the data movement in an overlapped manner. 
 *   For special dtypeSize in {9, ..., 16} (which is created from the merging of the small last dim), the data movement utilizes memcpy().
 *   The details are provided in
 *   void Generalized_Transpose(const void *src, void *dst, uint64_t srcOffset, uint64_t dstOffset, const G_Trans_Param gtrans)
 *
 *   Case 2. The last dim is unpermuted, i.e., perm_idx[ndim-1] == ndim-1. 
 *   In this case, we deploy memcpy() to move the entire last dim of data (recall the last dim size is coerced to be greater than 8B). 
 *   The details are given in
 *   void Permute_TypeB_Kernel(const void* src, void* dst, uint64_t dtypeSize, const uint64_t src_ndim, uint64_t* src_dims, uint64_t* src_wt, uint64_t* dst_wt);
 *
 * By default, it utilizes a single thread. Multi-thread is activated by setting the parameter nThreads >1.
 * By partitioning evenly along the first dim, i.e., [0], the proposed multi-thread operations nearly linearly reduce the running time,
 * until it saturates the memory bandwidth.
 */
int permute(const void* src, void* dst, uint64_t dtypeSize, uint64_t* src_dims,
    uint64_t src_ndim, uint64_t* permute_idx, uint64_t* dst_dims, int nThreads = 1);


 /* permute_validation() uses the randomly generated tensors and permutations to test against the equality
 *  permute(tensor, perm_idx) -> permuted_tensor
 *  permute(permuted_tensor, perm_inv) = tensor, where perm_inv is the inverse array of perm_idx.
 *  The first set of tensor dimensions and perm_idx is hand-designed to test a particular corner case
 *  whereas the remaining test cases are randomly generated to expand coverage. 
 */
void permute_validation();

// measure_permute() measures the efficiency of the proposed permute()
void measure_permute();

#endif
