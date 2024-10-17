#ifndef __TENSOR_PERMUTE_DTYPE_CONV_H__
#define __TENSOR_PERMUTE_DTYPE_CONV_H__

#include <stddef.h>
#include <stdint.h>
#include <limits.h>


/**
 * @brief This file effectively combines the proposed optimized tensor permutation with dtype converion, 
 * eliminating twice data movement and intermeidate buffer, under the function permute_dtypeConv(). 
 * The essential connection lies in 
 * #define DTYPE_CONV(dstPtr, srcPtr) {                                    \
    if constexpr (sizeof(T_S) <= sizeof(T_D))                           \
        SMtoLG_PTR(dstPtr, srcPtr);                                     \
    if constexpr (is_same<T_S, int32_t>::value && is_same<T_D, int8_t>::value)  \
        INT32toINT8_PTR(dstPtr, srcPtr);                                        \
    if constexpr (is_same<T_S, int32_t>::value && is_same<T_D, int16_t>::value) \
        INT32toINT16_PTR(dstPtr, srcPtr);                                       \
    if constexpr (is_same<T_S, int64_t>::value && is_same<T_D, int32_t>::value) \
        INT64toINT32_PTR(dstPtr, srcPtr); ;                                     \
    if constexpr (is_same<T_S, double>::value && is_same<T_D, float>::value)    \
        FP64toFP32_PTR(dstPtr, srcPtr);                                         \
  }
* The above macro function effectively eliminates the punishing cost related to the function call by using macros 
* and branching by using constexpr.   
 */
template<typename T_S, typename T_D> int permute_dtypeConv(const T_S* src, T_D* dst, uint64_t* src_dims,
    uint64_t src_ndim, uint64_t* permute_idx, uint64_t* dst_dims, int nThreads = 1);

 /* permute_dtypeConv_validation() uses the randomly generated tensors and permutations to test against the equality
 *  permute_dtypeConv(src_tensor, dst_tensor, perm_idx) -> dst_tensor
 *  permute_dtypeCOnv(dst_tensor, src_tensor, perm_inv) = src_tensor, where perm_inv is the inverse array of perm_idx.
 *  The first set of tensor dimensions and perm_idx is hand-designed to test a particular corner case
 *  whereas the remaining test cases are randomly generated to expand coverage. 
 */
void permute_dtypeConv_validation();

// measure_permute_dtypeCOnv() measures the efficiency of the proposed permute()
void measure_permute_dtypeConv();

#endif