/* LICENSE
The author makes NO WARRANTY or representation, either express or implied,
with respect to this software, its quality, accuracy, merchantability, or
fitness for a particular purpose.  This software is provided "AS IS", and you,
its user, assume the entire risk as to its quality and accuracy.

This software is copyright (C) 2024-2034, Cody (Yingquan) Wu.
All Rights Reserved except as specified below.

Permission is hereby granted to use, copy, modify, and distribute this
software (or portions thereof) for any purpose, without fee, subject to these
conditions:
(1) If any part of the source code for this software is distributed, then this
README file must be included, with this copyright and no-warranty notice
unaltered; and any additions, deletions, or changes to the original files
must be clearly indicated in the accompanying documentation.
(2) If only executable code is distributed, then the accompanying
documentation must state that "this software is based in part on the work of
Cody Wu".
(3) Permission for use of this software is granted only if the user accepts
full responsibility for any undesirable consequences; the authors accept
NO LIABILITY for damages of any kind.

We specifically permit and encourage the use of this software as the basis of
commercial products, provided that all warranty or liability claims are
assumed by the product vendor.
*/

#include "permute_dtypeConv.h"
#define MAX_DIM 32

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <algorithm> // For std::shuffle
#include <random>    // For std::random_device and std::mt19937
#include <thread>    // for std::thread
#include <vector>
#include <type_traits>  // For std::is_same
#include <cfloat>
//#include <half.hpp>

using namespace std;

#define HALF_MAX 65504.0f

template<typename T_S, typename T_D> bool dtype_conv()
{
    if (sizeof(T_S) <= sizeof(T_D)) return true;
    if (is_same<T_S, int32_t>::value && is_same<T_D, int8_t>::value) return true;
    if (is_same<T_S, int32_t>::value && is_same<T_D, int16_t>::value) return true;
    if (is_same<T_S, int64_t>::value && is_same<T_D, int32_t>::value) return true;
    //if (is_same<T_S, float>::value && is_same<T_D, half>::value) return true;
    if (is_same<T_S, double>::value && is_same<T_D, float>::value) return true;
    return false;
}


#define FP32toFP16_PTR(dstPtr, srcPtr)                                  \
{                                                                       \
    if (*srcPtr>=-HALF_MAX && *srcPtr <= HALF_MAX) {                    \
         *dstPtr++ = (T_D)(*srcPtr);                                    \
    } else if (*srcPtr < -HALF_MAX) {                                   \
        *dstPtr++ = -HALF_MAX;                                          \
    } else {                                                            \
        *dstPtr++ = HALF_MAX;                                           \
    }                                                                   \
    srcPtr++;                                                           \
}

#define FP64toFP32_PTR(dstPtr, srcPtr)                                  \
{                                                                       \
    if (*srcPtr>=-FLT_MAX && *srcPtr <= FLT_MAX) {                      \
        *dstPtr++ = (T_D)(*srcPtr);                                     \
    } else if (*srcPtr < -FLT_MAX) {                                    \
        *dstPtr++ = -INFINITY;                                          \
    } else {                                                            \
        *dstPtr++ = INFINITY;                                           \
    }                                                                   \
    srcPtr++;                                                           \
}

#define INT64toINT32_PTR(dstPtr, srcPtr)                                \
{                                                                       \
    if ( *srcPtr>= INT32_MIN && *srcPtr <= INT32_MAX) {                 \
        *dstPtr++ = (T_D)(*srcPtr);                                     \
    } else if (*srcPtr < INT32_MIN) {                                   \
        *dstPtr++ = INT32_MIN;                                          \
    } else {                                                            \
        *dstPtr++ = INT32_MAX;                                          \
    }                                                                   \
    srcPtr++;                                                           \
} 

#define INT32toINT16_PTR(dstPtr, srcPtr)                                \
{                                                                       \
    if ( *srcPtr>= INT16_MIN && *srcPtr <= INT16_MAX) {                 \
        *dstPtr++ = (T_D)(*srcPtr);                                     \
    } else if (*srcPtr < INT16_MIN) {                                   \
        *dstPtr++ = INT16_MIN;                                          \
    } else {                                                            \
        *dstPtr++ = INT16_MAX;                                          \
    }                                                                   \
    srcPtr++;                                                           \
} 

#define INT32toINT8_PTR(dstPtr, srcPtr)                                 \
{                                                                       \
    if ( *srcPtr>= INT8_MIN && *srcPtr <= INT8_MAX) {                   \
        *dstPtr++ = (T_D)(*srcPtr);                                     \
    } else if (*srcPtr < INT8_MIN) {                                    \
        *dstPtr++ = INT8_MIN;                                           \
    } else {                                                            \
        *dstPtr++ = INT8_MAX;                                           \
    }                                                                   \
    srcPtr++;                                                           \
} 

#define SMtoLG_PTR(dstPtr, srcPtr) {                                    \
    *dstPtr++ = *srcPtr++;                                              \
}

// it utilizes macro function to eliminate function call and "constexpr" to eliminate branching during compiling 
#define DTYPE_CONV(dstPtr, srcPtr) {                                    \
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

typedef struct {
    int64_t lastDim;       // last dim, this is used when the last (small) dim is treated as a single data unit
    int64_t rows, cols;    // numbers of rows and columns
    int64_t srcWt, dstWt;  // weights of source and destination indexes.    
} G_Trans_Param;

/* Batch Generalized Transpose, wherein 2D transpose is a special case with srcWt=cols, dstWt=rows
* BATCH rows are transposed in batch so as to optimize caching
*/
template <typename T_S, typename T_D, int BATCH> void G_Transpose(const T_S* src, T_D* dst, G_Trans_Param gtrans)
{
    int64_t i, n, s_i;
    T_D* dstPtr;
    const T_S* srcPtr[BATCH];
    for (i = s_i = 0; i < gtrans.rows - BATCH + 1; i += BATCH, s_i += BATCH * gtrans.srcWt) {
        dstPtr = dst + i;               // beginning of column no. i
        srcPtr[0] = src + s_i;          // beginning of row no. i 
        for (n = 1; n < BATCH; n++)         // beginning of rows no. i+1, i+2, i+3, ..., i+31 
            srcPtr[n] = srcPtr[0] + n * gtrans.srcWt;
        for (n = gtrans.cols; n > 0; n--) {  //Transpose the batch of rows in parallel
            for (int j = 0; j < BATCH; j++)
                DTYPE_CONV(dstPtr, srcPtr[j]);
            dstPtr += gtrans.dstWt - BATCH;         // jumping to next column index
        }
    }
    const int remBatch = gtrans.rows % BATCH;
    if (remBatch) {           // Transpose the residual batch of rows 
        dstPtr = dst + i;       // beginning of column no. i
        srcPtr[0] = src + s_i;  // beginning of row no. i 
        for (n = 1; n < remBatch; n++)
            srcPtr[n] = srcPtr[0] + n * gtrans.srcWt;
        for (n = gtrans.cols; n > 0; n--) {
            for (int j = 0; j < remBatch; j++)
                DTYPE_CONV(dstPtr, srcPtr[j]);
            dstPtr += gtrans.dstWt - remBatch;
        }
    }
}
/*
* Batch generalized transpose for the special case the compressed last dim, i.e., gtrans.lastDim >1.
* BATCH rows are transposed in batch so as to optimize caching
*/
template <typename T_S, typename T_D, int BATCH> void G1_Transpose(const T_S* src, T_D* dst, G_Trans_Param gtrans)
{
    const uint64_t lastDim = gtrans.lastDim;
    if (lastDim > 16) {
        throw runtime_error("last dim oversize");
    }
    const int  remBatch = gtrans.rows % BATCH;
    int64_t i, j, n, s_i;
    const T_S* srcPtr[BATCH];
    T_D* dstPtr;

    for (i = s_i = 0; i < gtrans.rows - BATCH + 1; i += BATCH, s_i += BATCH * gtrans.srcWt) {
        dstPtr = dst + i * lastDim;               // beginning of column no. i
        srcPtr[0] = src + s_i * lastDim;          // beginning of row no. i 
        for (n = 1; n < BATCH; n++)         // beginning of rows no. i+1, i+2, ..., i+7 
            srcPtr[n] = srcPtr[0] + n * gtrans.srcWt * gtrans.lastDim;
        for (n = gtrans.cols; n > 0; n--) {  //Transpose the batch of rows in parallel
            for (j = 0; j < BATCH; j++) { 
                for(int k=lastDim; k>0; k--)   // dtype converion over the last dim
                    DTYPE_CONV(dstPtr, srcPtr[j]);
            }
            dstPtr += (gtrans.dstWt - BATCH) * lastDim;         // jumping to next column index
        }
    }
    // Transpose the residual batch of rows 
    dstPtr = dst + i * lastDim;               // beginning of column no. i
    srcPtr[0] = src + s_i * lastDim;          // beginning of row no. i 
    for (n = 1; n < remBatch; n++)
        srcPtr[n] = srcPtr[0] + n * gtrans.srcWt * lastDim;
    for (n = gtrans.cols; n > 0; n--) {
        for (j = 0; j < remBatch; j++) {     // remBatch is locally defined as within batch so that compiler learns to unroll this for-loop
            for (int k = lastDim; k > 0; k--)       // dtype converion over the last dim
                DTYPE_CONV(dstPtr, srcPtr[j]);
        }
        dstPtr += (gtrans.dstWt - remBatch)* lastDim;         // jumping to next column index;
    }
    
}


/* Generalized batch transpose to optimize 64 - byte cache - line utilization.
 * Due to random starting address, 32-byte consecutive write addresses results in 33/64 chance of utilizing single cache line, while the remaining 31/64 chance in 2 cache lines.
 * Simulations indicates that the consecutive write addresses of 32-byte is the sweet spot. 
 * To this end, we define BATCH = 31/sizeof(T_D) +1;
 */
template <typename T_S, typename T_D>void Generalized_Transpose(const T_S *src, T_D *dst, uint64_t srcOffset, uint64_t dstOffset, const G_Trans_Param gtrans)
{
    const T_S* srcPtr = src + srcOffset * gtrans.lastDim;
    T_D* dstPtr = dst + dstOffset * gtrans.lastDim;
    if (gtrans.lastDim == 1) {   
        switch ( sizeof(T_D) ) {
        case 1:
            G_Transpose<T_S, T_D, 32>(srcPtr, dstPtr, gtrans);  break;
        case 2:
            G_Transpose<T_S, T_D, 16>(srcPtr, dstPtr, gtrans);  break;
        case 4:
            G_Transpose<T_S, T_D, 8>(srcPtr, dstPtr, gtrans);   break;
        default:
            G_Transpose<T_S, T_D, 4>(srcPtr, dstPtr, gtrans);   break;
        }
    } else {    // last small dim is treated as a single unit
        switch (sizeof(T_D) * gtrans.lastDim) {
        case 2:
            G1_Transpose<T_S, T_D, 16>(srcPtr, dstPtr, gtrans);  break;
        case 4:
            G1_Transpose<T_S, T_D, 8>(srcPtr, dstPtr, gtrans);   break;
        case 8:
            G1_Transpose<T_S, T_D, 4>(srcPtr, dstPtr, gtrans);   break;
        case 3:
            G1_Transpose<T_S, T_D, 11>(srcPtr, dstPtr, gtrans);  break;
        case 5:
            G1_Transpose<T_S, T_D, 7>(srcPtr, dstPtr, gtrans);   break;
        case 6:
            G1_Transpose<T_S, T_D, 6>(srcPtr, dstPtr, gtrans);   break;
        case 7:
            G1_Transpose<T_S, T_D, 5>(srcPtr, dstPtr, gtrans);   break;
        default:  // case {9, ..., 16}
            G1_Transpose<T_S, T_D, 3>(srcPtr, dstPtr, gtrans);   break;
        }
    }
}

template <typename T_S, typename T_D>void Permute_TypeA_Kernel(const T_S* src, T_D* dst, const uint64_t src_ndim,
        uint64_t * src_dims_new, uint64_t* src_wt_new, uint64_t* dst_wt_new, const G_Trans_Param gtrans)
{
    int64_t i, k;
    uint64_t  src_idx[MAX_DIM] = { 0 };
    uint64_t srcOffset, dstOffset;
    switch (src_ndim) {
    case 2:
        Generalized_Transpose<T_S, T_D>(src, dst, 0, 0, gtrans);
        break;
    case 3:
        for (srcOffset = dstOffset = 0; srcOffset < src_dims_new[0] * src_wt_new[0];
            srcOffset += src_wt_new[0], dstOffset += dst_wt_new[0]) {
            Generalized_Transpose<T_S, T_D>(src, dst, srcOffset, dstOffset, gtrans);
        }
        break;
    default:
        srcOffset = dstOffset = 0;
        while (1) {
            Generalized_Transpose<T_S, T_D>(src, dst, srcOffset, dstOffset, gtrans);

            // iteratively generate next src_idx[]
            // it is equivalent to src_dim-2 layers of nested for-loops.
            k = src_ndim - 3;
            if (src_idx[k] < src_dims_new[k] - 1) {     // pro-dominant case
                src_idx[k]++;
                srcOffset += src_wt_new[k];
                dstOffset += dst_wt_new[k];
            }
            else {
                k--;        // note src_idx[k] == src_dims_new[k] - 1
                while (k >= 0 && src_idx[k] >= src_dims_new[k] - 1) k--;
                if (k == -1) break;  // sweeping though full scope
                src_idx[k]++;
                for (i = src_ndim - 2; i > k; i--)
                    src_idx[i] = 0;

                srcOffset = dstOffset = 0;
                for (i = 0; i <= k; i++) {   // observing src_idx[i] = 0, for i>k
                    srcOffset += src_idx[i] * src_wt_new[i];
                    dstOffset += src_idx[i] * dst_wt_new[i];
                }
            }
        }
    }
}
/* Permute for Type-A: the last dimension is permuted, i.e., perm_idx[ndim - 1] != ndim - 1. 
* EX : perm_idx[3, 1, 2, 0],  perm_idx[3, 0, 1, 2],  perm_idx[4, 5, 0, 1, 2, 3]
* Its core operation is the generalized parallel transpose.
*/
template <typename T_S, typename T_D>int Permute_TypeA(const T_S* src, T_D* dst, const uint64_t src_ndim,
    uint64_t* src_dims, const uint64_t lastDim, uint64_t* perm_idx) {
    uint64_t src_wt[MAX_DIM], dst_wt[MAX_DIM];
    uint64_t src_order[MAX_DIM];
    int i;

    //Compute the weights of each dimension for source and destination tensors
    src_wt[src_ndim - 1] = 1;
    dst_wt[perm_idx[src_ndim - 1]] = 1;
    for (i = (int)src_ndim - 1; i > 0; i--) {
        src_wt[i - 1] = src_wt[i] * src_dims[i];
        dst_wt[perm_idx[i - 1]] = dst_wt[perm_idx[i]] * src_dims[perm_idx[i]];
    }

    for (i = 0; i < src_ndim; i++)
        src_order[i] = i;

    /* swap two indexes src_ndim - 2 and perm_idx[src_ndim - 1] */
    src_order[src_ndim - 2] = src_order[perm_idx[src_ndim - 1]];
    for(i= (int)perm_idx[src_ndim - 1]; i< (int)src_ndim - 2; i++)
        src_order[i]++;

    G_Trans_Param gtrans;
    gtrans.lastDim = lastDim;
    gtrans.rows = src_dims[src_order[src_ndim - 2]];
    gtrans.cols = src_dims[src_ndim - 1];
    gtrans.srcWt = src_wt[src_order[src_ndim - 2]];
    gtrans.dstWt = dst_wt[src_ndim-1];
 
    //A key advantage of the following implementation is that each permuted dimension is treated independently. 
    //Moreover, the index computation does not involve multiplications/divisions.
    uint64_t src_dims_new[MAX_DIM] = { 0 };
    uint64_t src_wt_new[MAX_DIM] = { 0 }, dst_wt_new[MAX_DIM] = { 0 };

    for (i = 0; i < src_ndim - 2; i++) {
        src_dims_new[i] = src_dims[src_order[i]];
        src_wt_new[i] = src_wt[src_order[i]];
        dst_wt_new[i] = dst_wt[src_order[i]];
    }
    
    Permute_TypeA_Kernel<T_S, T_D>(src, dst, src_ndim, src_dims_new, src_wt_new, dst_wt_new, gtrans);
    return 0;
}

/*~~~~~~~~~~~~~~~ Multi-thread implementation of Permute_TypeA() ~~~~~~~~~~~~~~~
* The first dimension, [0], is partitioned evenly for each thread. 
* Memory write is guaranteed to be non-overlapping, thus no need to involve mutex.
*/
template <typename T_S, typename T_D>int Permute_TypeA_MThread(const T_S* src, T_D* dst,  const uint64_t src_ndim,
    uint64_t* src_dims, const uint64_t lastDim, uint64_t* perm_idx, int nThreads) {
    uint64_t src_wt[MAX_DIM], dst_wt[MAX_DIM];
    uint64_t src_order[MAX_DIM];
    int i;

    //Compute the weights of each dimension for source and destination tensors
    src_wt[src_ndim - 1] = 1;
    dst_wt[perm_idx[src_ndim - 1]] = 1;
    for (i = (int)src_ndim - 1; i > 0; i--) {
        src_wt[i - 1] = src_wt[i] * src_dims[i];
        dst_wt[perm_idx[i - 1]] = dst_wt[perm_idx[i]] * src_dims[perm_idx[i]];
    }

    for (i = 0; i < src_ndim; i++)
        src_order[i] = i;

    /* swap two indexes src_ndim - 2 and perm_idx[src_ndim - 1] */
    src_order[src_ndim - 2] = src_order[perm_idx[src_ndim - 1]];
    for (i = (int)perm_idx[src_ndim - 1]; i < (int)src_ndim - 2; i++)
        src_order[i]++;

    G_Trans_Param gtrans;
    gtrans.lastDim = lastDim;
    gtrans.rows = src_dims[src_order[src_ndim - 2]];
    gtrans.cols = src_dims[src_ndim - 1];
    gtrans.srcWt = src_wt[src_order[src_ndim - 2]];
    gtrans.dstWt = dst_wt[src_ndim - 1];

    //A key advantage of the following implementation is that each permuted dimension is treated independently. 
    //Moreover, the index computation does not involve multiplications/divisions.
    uint64_t src_dims_new[MAX_DIM] = { 0 };
    uint64_t src_wt_new[MAX_DIM] = { 0 }, dst_wt_new[MAX_DIM] = { 0 };

    for (i = 0; i <= src_ndim - 2; i++) {
        src_dims_new[i] = src_dims[src_order[i]];
        src_wt_new[i] = src_wt[src_order[i]];
        dst_wt_new[i] = dst_wt[src_order[i]];
    }
    vector<thread> threads(nThreads);
    uint64_t src_dims_1[MAX_DIM], src_dims_2[MAX_DIM];
    memcpy(src_dims_1, src_dims_new, src_ndim * sizeof(uint64_t));
    memcpy(src_dims_2, src_dims_new, src_ndim * sizeof(uint64_t));

    const int rThreads = src_dims_new[0] % nThreads;
    const uint64_t dim0 = src_dims_new[0] / nThreads;
    src_dims_1[0] = dim0 + 1;
    src_dims_2[0] = dim0;
    
    const T_S* srcPtr = src;
    T_D* dstPtr = dst;
    for (int i = 0; i < rThreads; i++) {
        if (src_ndim == 2) {
            gtrans.rows = dim0 + 1;
        }
        threads[i] = thread(Permute_TypeA_Kernel<T_S, T_D>, srcPtr, dstPtr, src_ndim, src_dims_1, src_wt_new, dst_wt_new, gtrans);
        srcPtr += src_dims_1[0] * src_wt_new[0] * lastDim;
        dstPtr += src_dims_1[0] * dst_wt_new[0] * lastDim;
    }
    if (dim0 > 0) {
        for (int i = rThreads; i < nThreads; i++) {
            if (src_ndim == 2) {
                gtrans.rows = dim0;
            }
            threads[i] = thread(Permute_TypeA_Kernel<T_S, T_D>, srcPtr, dstPtr, src_ndim, src_dims_2, src_wt_new, dst_wt_new, gtrans);
            srcPtr += src_dims_2[0] * src_wt_new[0] * lastDim;
            dstPtr += src_dims_2[0] * dst_wt_new[0] * lastDim;
        }
    }
    if (dim0 == 0) nThreads = rThreads;     //effective number of threads
    for (int i = 0; i < nThreads; i++)
        threads[i].join();

    return 0;
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TYPE B Permute  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The last dimension is not permuted. consecutive dtype_conve is applied on the entire last dim of data.
*/
template <typename T_S, typename T_D> void Permute_TypeB_Kernel(const T_S* src, T_D* dst, const uint64_t src_ndim,
    uint64_t* src_dims, uint64_t* src_wt, uint64_t* dst_wt)
{
    const T_S* srcPtr;
    T_D* dstPtr;
    uint64_t srcOffset = 0, dstOffset = 0;
    uint64_t src_idx[MAX_DIM] = { 0 };
    while (1) {
        srcPtr = src + srcOffset;
        dstPtr = dst + dstOffset;
        for (int64_t i = src_dims[src_ndim - 1]; i > 0; i--)
            DTYPE_CONV(dstPtr, srcPtr);


        // iteratively generate next src_idx[]
        // it is equivalent to src_dim-1 layers of nested for-loops.
        int k = (int)(src_ndim)-2;
        if ( src_idx[k] < src_dims[k] - 1 ) {       // pro-dominant case
            src_idx[k]++;
            srcOffset += src_wt[k];
            dstOffset += dst_wt[k];
        }
        else {
            k--;    // as src_idx[k] == src_dims[k] - 1
            while (k >= 0 && src_idx[k] >= src_dims[k] - 1) k--;
            if (k == -1) break;  // sweeping though full scope
            src_idx[k]++;
            for (int i = (int)src_ndim - 2; i > k; i--)
                src_idx[i] = 0;

            srcOffset = dstOffset = 0;
            for (int i = 0; i <= k; i++) {       // observing src_idx[i] = 0, for i>k
                srcOffset += src_idx[i] * src_wt[i];
                dstOffset += src_idx[i] * dst_wt[i];
            }
        }
    }
}
/*Permute for Type - B: the last dimension is unpermuted, i.e., perm_idx[ndim - 1] == ndim - 1. 
*EX : perm_idx[2, 1, 0, 3], perm_idx[1, 0, 2], perm_idx[2, 3, 0, 1, 4]
*/
template <typename T_S, typename T_D> int Permute_TypeB(const T_S* src, T_D* dst, const uint64_t src_ndim,
    uint64_t* src_dims, uint64_t* perm_idx) {
    uint64_t src_wt[MAX_DIM], dst_wt[MAX_DIM];

    src_wt[src_ndim - 1] = 1;
    dst_wt[src_ndim - 1] = 1;
    for (int i = (int)src_ndim - 1; i > 0; i--) {
        src_wt[i - 1] = src_wt[i] * src_dims[i];
        dst_wt[perm_idx[i - 1]] = dst_wt[perm_idx[i]] * src_dims[perm_idx[i]];
    }
   
    Permute_TypeB_Kernel<T_S, T_D>(src, dst, src_ndim, src_dims, src_wt, dst_wt);
    return 0;
}

/* ~~~~~~~~~~~~~~~ Multi-thread implementation of Permute_TypeB() ~~~~~~~~~~~~~~~
* The first dimension, [0], is partitioned evenly for each thread. 
* Memory write is guaranteed to be non-overlapping, thus no need to involve mutex.
*/

template <typename T_S, typename T_D> int Permute_TypeB_MThread(const T_S* src, T_D* dst, const uint64_t src_ndim,
    uint64_t* src_dims, uint64_t* perm_idx, int nThreads) {
    uint64_t src_wt[MAX_DIM], dst_wt[MAX_DIM];

    src_wt[src_ndim - 1] = 1;
    dst_wt[src_ndim - 1] = 1;
    for (int i = (int)src_ndim - 1; i > 0; i--) {
        src_wt[i - 1] = src_wt[i] * src_dims[i];
        dst_wt[perm_idx[i - 1]] = dst_wt[perm_idx[i]] * src_dims[perm_idx[i]];
    }

    vector<thread> threads(nThreads);
    uint64_t src_dims_1[MAX_DIM], src_dims_2[MAX_DIM];
 
    memcpy(src_dims_1, src_dims, src_ndim * sizeof(uint64_t));
    memcpy(src_dims_2, src_dims, src_ndim * sizeof(uint64_t));
    const int rThreads = src_dims[0] % nThreads;
    const uint64_t dim0 = src_dims[0] / nThreads;
    src_dims_1[0] = dim0 + 1;
    src_dims_2[0] = dim0;

    const T_S* srcPtr = src;
    T_D* dstPtr = dst;
    for (int i = 0; i < rThreads; i++) {        
        threads[i] = thread(Permute_TypeB_Kernel<T_S, T_D>, srcPtr, dstPtr, src_ndim, src_dims_1, src_wt, dst_wt);
        srcPtr += src_dims_1[0] * src_wt[0];
        dstPtr += src_dims_1[0] * dst_wt[0];
    }
    if (dim0 > 0) {
        for (int i = rThreads; i < nThreads; i++) {
            threads[i] = thread(Permute_TypeB_Kernel<T_S, T_D>, srcPtr, dstPtr, src_ndim, src_dims_2, src_wt, dst_wt);
            srcPtr += src_dims_2[0] * src_wt[0];
            dstPtr += src_dims_2[0] * dst_wt[0];
        }
    }

    if (dim0 == 0) nThreads = rThreads;     //effective number of  threads
    for (int i = 0; i < nThreads; i++)
        threads[i].join();

    return 0;
}

template<typename T_S, typename T_D> int permute_dtypeConv(const T_S* src, T_D* dst, uint64_t* src_dims,
    uint64_t src_ndim, uint64_t* permute_idx, uint64_t* dst_dims, int nThreads) {
    uint64_t squz_dims[MAX_DIM], tmp_dims[MAX_DIM], trim_dims[MAX_DIM];
    uint64_t perm_idx[MAX_DIM];
    uint64_t perm_off[MAX_DIM] = { 0 };
    int64_t perm_inv[MAX_DIM];
    uint64_t squz_ndim, trim_ndim;
    

    uint64_t i, n;
    /*~~~~~~~~~~~~~~~~ squeeze dims with size 1 ~~~~~~~~~~~~~~~~
    * EX. 1.   perm_idx[2, 3, 5, 4, 1, 7, 0,  6]  ==>  [2, 4, 3, 1, 6, 0,  5]
    *          src_nums[8, 9, 1, 6, 4, 6, 9, 10]  ==>  [8, 9, 6, 4, 6, 9, 10]
    * 
    * EX. 2.   perm_idx[2, 3, 5, 4, 1, 7, 0,  6]  ==>  [1, 2, 4, 3, 0, 5]
    *          src_nums[1, 8, 9, 6, 4, 6, 1, 10]  ==>  [8, 9, 6, 4, 6, 10]
    * 
    * The following algorithm has O(n) time and space complexity
    */
    for (i = 0; i < src_ndim; i++) {/* compute inverse of permute_idx */
        perm_idx[i] = permute_idx[i];
        perm_inv[permute_idx[i]] =i;
        dst_dims[i] = src_dims[permute_idx[i]];
    }

    squz_ndim = 0;
    for (i = 0; i < src_ndim; i++) {
        if (src_dims[i] > 1) {
            squz_dims[squz_ndim++] = src_dims[i];
        }
        else {  /* squeeze the trivial dimension */
            perm_off[i] = 1;
            perm_idx[ perm_inv[i] ] = src_ndim;
        }
    }
    if (squz_ndim < 2) {
        cerr<<"the input matrix is an array, no need to permute\n";
        return 0;
    }

    /* compute cumulative offset */
    for (i = 1; i < src_ndim; i++) {
        perm_off[i] += perm_off[i - 1];
    }

    n = 0;
    for (i = 0; i < src_ndim; i++) {
        if (perm_idx[i] < src_ndim)
            perm_idx[n++] = perm_idx[i] - perm_off[perm_idx[i]];
    }

    /*~~~~~~~~~~~~~~~~  Merge consecutive permuted dims ~~~~~~~~~~~~~~~~ 
    * EX. 1.   perm_idx[2, 3,  4,  5,  6, 7,     0,  1]  ==>  [1,              0  ]
    *          src_nums[8, 9,  12, 6,  4, 6,     9, 10]  ==>  [8*9,  12*6*4*6*9*10] 
    * 
    * EX. 2.   perm_idx[2, 3,   5,  4,   6, 7,   0,  1]  ==>  [1,     3,    2,   4,    0  ]
    *          src_nums[8, 9,   12, 6,   4, 6,   9, 10]  ==>  [8*9,  12*6,  4,   6,   9*10] 
    * 
    * EX. 3.   perm_idx[2, 3,  4,  5,    0, 1,   6, 7 ]  ==>  [1,           0,    2 ]
    *          src_nums[8, 9,  12, 6,    4, 6,   9, 10]  ==>  [8*9,  12*6*4*6,  9*10] 
    * 
    * EX. 4.   perm_idx[5, 4,  7,  6,     0, 1,  2, 3 ]  ==>  [2,          1,  4,  3,  0 ]
    *          src_nums[8, 9,  12, 6,     4, 6,  9, 10]  ==>  [8*9*12*6,   4,  6,  9,  10] 
    * 
    * EX. 5.   perm_idx[5, 4,  2,  6,     1, 0,  3,  7]  ==>  leave as 
    *          src_nums[8, 9,  12, 6,     4, 6,  9, 10]
    * 
    *  The following algorithm has O(n) time and space complexity
    */ 
    memset(perm_off, 0, src_ndim * sizeof(uint64_t));

    uint64_t pm, idx;
    tmp_dims[perm_idx[0]] = squz_dims[perm_idx[0]];
    for (i = 1; i < squz_ndim; ) {
        if (perm_idx[i] != perm_idx[i - 1] + 1) {
            tmp_dims[perm_idx[i]] = squz_dims[perm_idx[i]];
            i++;
        }
        else {  /* reduce consecutive dimensions */
            pm = perm_idx[i - 1];
            idx = i - 1;
            while (i < squz_ndim && perm_idx[i]+idx == pm+i ) {
                tmp_dims[pm] *= squz_dims[perm_idx[i]];
                perm_off[ perm_idx[i] ] = 1;
                perm_idx[i] = src_ndim;
                i++;
            }
        }
    }

    /* compute cumulative offset */
    for (i = 1; i < squz_ndim; i++) {
        perm_off[i] += perm_off[i - 1];
    }

    trim_ndim = 0;
    for (i = 0; i < squz_ndim; i++) {
        if (perm_idx[i] < src_ndim) {
            n = perm_idx[i] - perm_off[perm_idx[i]];
            trim_dims[n] = tmp_dims[perm_idx[i]];
            perm_idx[trim_ndim++] = n;
        }
    }
    
    /*In case the last dim is unmuted, check it fits into a single data type.
    * EX. perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]
    *     trim_dims[20, 8, 16, 12, 8]   ==> trim_dims[20, 8, 16, 12]
    *     dtypeSize = 1                 ==> dtypeSize = 8
    * 
    * EX. perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]
    *     trim_dims[20, 8, 16, 12, 6]   ==> trim_dims[20, 8, 16, 12]
    *     dtypeSize = 1                 ==> dtypeSize = 6
    */
    uint64_t  lastDim = 1;
    if (perm_idx[trim_ndim - 1] == trim_ndim - 1) {
        uint64_t cpSize = trim_dims[trim_ndim - 1] * sizeof(T_D);  // the size of last dim
        if (cpSize <= 16) {   
            lastDim = trim_dims[trim_ndim - 1];     // treat the last dim as a single data
            trim_ndim--;            // get rid of the last dim
        }
    }

    if (!dtype_conv<T_S, T_D>()) {
        cerr << "Unsupported Dtype Convertion\n";
        return 1;
    }
    if (1 == trim_ndim) {
        const T_S* srcPtr = src;
        T_D* dstPtr = dst;
        for (i = trim_dims[0]; i > 0; i--) {
            DTYPE_CONV(dstPtr, srcPtr);
        }
        return 0;
    }

    if (nThreads == 1) {
        if (perm_idx[trim_ndim - 1] != trim_ndim - 1)
            return Permute_TypeA<T_S, T_D>(src, dst, trim_ndim, trim_dims, lastDim, perm_idx);
        else
            return Permute_TypeB<T_S, T_D>(src, dst,  trim_ndim, trim_dims, perm_idx);
    }
    else {
        if (perm_idx[trim_ndim - 1] != trim_ndim - 1)
            return Permute_TypeA_MThread<T_S, T_D>(src, dst, trim_ndim, trim_dims, lastDim, perm_idx, nThreads);
        else
            return Permute_TypeB_MThread<T_S, T_D>(src, dst, trim_ndim, trim_dims, perm_idx, nThreads);
    }
}

void permute_dtypeConv_validation()
{
    typedef int32_t stype;
    typedef int16_t dtype;
    const uint64_t src_ndim = 8;
    //The initial perm_idx and src_dims are guaranteed to be executed 
    uint64_t perm_idx[8] = { 3, 6, 5, 4, 0, 2, 1, 7 };
    uint64_t src_dims[8] = { 4, 6, 2, 3, 5, 4, 7, 3 };
    uint64_t perm_inv[MAX_DIM], dst_dims[MAX_DIM];
    size_t i, nElmt;
    int nThreads;

    // Initialize a random number generator with the seed
    mt19937 g(0);

    const int count = 100;      // number of random tests
    for (int cnt = count; cnt > 0; cnt--) {
        // compute the overall number of elements, and the inverse permute index array
        nElmt = 1;
        for (i = 0; i < src_ndim; i++) {
            nElmt *= src_dims[i];
            perm_inv[perm_idx[i]] = i;
        }

        stype* srcMtx = new stype[nElmt];
        dtype* dstMtx = new dtype[nElmt];
        for (i = 0; i < nElmt; i++)
            srcMtx[i] = (dtype)i;

        // verify the equality permute(inverse_permute) = original
        nThreads = rand() % 8 + 1;
        permute_dtypeConv(srcMtx, dstMtx, src_dims, src_ndim, perm_idx, dst_dims, nThreads);
        nThreads = rand() % 8 + 1;
        permute_dtypeConv(dstMtx, srcMtx, dst_dims, src_ndim, perm_inv, src_dims, nThreads);

        for (i = 0; i < nElmt; i++) {
            if (srcMtx[i] != (dtype)i) {
                cerr << "\nPermute[" << perm_idx[0] << ", " << perm_idx[1] << ", " << perm_idx[2] << ", " << perm_idx[3] << ", "
                    << perm_idx[4] << ", " << perm_idx[5] << ", " << perm_idx[6] << ", " << perm_idx[7] << "] over Tensor["
                    << src_dims[0] << ", " << src_dims[1] << ", " << src_dims[2] << ", " << src_dims[3] << ", "
                    << src_dims[4] << ", " << src_dims[5] << ", " << src_dims[6] << ", " << src_dims[7] << "] is incorrect";
                break;
            }
        }
        delete[] srcMtx;
        delete[] dstMtx;

        // Shuffle the permute index array 
        shuffle(perm_idx, perm_idx + src_ndim, g);

        for (i = 0; i < src_ndim; i++) {
            src_dims[i] = rand() % 8 + 1;  // each dimension width in [1, 10]
        }
    }

    cout << "Permute has been validated by " << count << " random tests\n";
}
#define NSEC_IN_SEC (1000000000L)
#define CLOCK_GETTIME(timestamp) clock_gettime(CLOCK_MONOTONIC, &(timestamp));
//#define CLOCK_GETTIME(timestamp)  timespec_get(&timestamp, TIME_UTC)
#define TIMESPEC_TO_NSEC(ts) ((ts).tv_sec * NSEC_IN_SEC + (ts).tv_nsec)

void measure_permute_dtypeConv() 
{
    typedef int32_t stype;
    typedef int16_t dtype;
    size_t numIters = 100;
    struct timespec start_time, end_time;
    uint64_t dtypeSize = sizeof(dtype);
    uint64_t perm_idx[] = { 3, 0, 1, 2, 4, 5, 6, 7 };
    uint64_t src_dims[] = { 10, 12, 14, 16, 24, 16, 18, 12 };
    uint64_t src_ndim = 8;
    uint64_t perm_inv[MAX_DIM], dst_dims[MAX_DIM];
    size_t i, nElmt;
    nElmt = 1;
    for (i = 0; i < src_ndim; i++) {
        nElmt *= src_dims[i];
        perm_inv[perm_idx[i]] = i;
    }

    cout<<"\nsrc_ndim=" << src_ndim <<", nElmt =" << nElmt <<", dtypeSize ="<< dtypeSize <<", Source dimensions\n";
    for (i = 0; i < src_ndim; i++)
        cout << src_dims[i] << " ";
    cout << "\nPermuted dimensions\n";
    for (i = 0; i < src_ndim; i++)
        cout << perm_idx[i] << " ";
    cout << endl;

    stype* srcMtx = new stype[nElmt];
    dtype* dstMtx = new dtype[nElmt];
    for (i = 0; i < nElmt; i++)
        srcMtx[i] = (dtype)i;

    for (int k = 0; k < 4; k++) {
        auto nThrds = 1 << k;
        CLOCK_GETTIME(start_time);
        for (i = numIters; i > 0; i--) {
            permute_dtypeConv(srcMtx, dstMtx, src_dims, src_ndim, perm_idx, dst_dims, nThrds);
        }
        CLOCK_GETTIME(end_time);
        double aveSecd = (double)(TIMESPEC_TO_NSEC(end_time) - TIMESPEC_TO_NSEC(start_time)) / NSEC_IN_SEC / numIters;
        printf("%zu iterations with ave %.6f s, thpt: %.3f GB/s, threads=%d\n", numIters, aveSecd, dtypeSize * nElmt / aveSecd / NSEC_IN_SEC, nThrds);
    }

    printf("\n%zu iterations with average %zu\n", numIters, (TIMESPEC_TO_NSEC(end_time) - TIMESPEC_TO_NSEC(start_time)) / numIters);
    delete[] srcMtx;
    delete[] dstMtx;
}