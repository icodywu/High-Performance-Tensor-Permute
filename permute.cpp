#include "permute.h"
#define MAX_DIM 32
#define DEBUG 0

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


typedef struct {
    int dtypeSize;     // bytes of data type
    int rows, cols;    // numbers of rows and columns
    int srcWt, dstWt;  // weights of source and destination indexes.    
} G_Trans_Param;

/* Batch Generalized Transpose, wherein 2D transpose is a special case with srcWt=cols, dstWt=rows
* BATCH rows are transposed in batch so as to optimize caching
*/
template <typename T, int BATCH> void G_Transpose(T* src, T* dst, G_Trans_Param gtrans)
{
    int i, n, s_i;
    T* dstPtr, * srcPtr[32];
    for (i = s_i = 0; i < gtrans.rows - BATCH + 1; i += BATCH, s_i += BATCH * gtrans.srcWt) {
        dstPtr = dst + i;               // beginning of column no. i
        srcPtr[0] = src + s_i;          // beginning of row no. i 
        for (n = 1; n < BATCH; n++)         // beginning of rows no. i+1, i+2, i+3, ..., i+31 
            srcPtr[n] = srcPtr[0] + n * gtrans.srcWt;
        for (n = gtrans.cols; n > 0; n--) {  //Transpose the batch of rows in parallel
            switch (BATCH) {        //Note that branching is removed at the compiling time
            case 4:
                *dstPtr++ = *srcPtr[0]++;   *dstPtr++ = *srcPtr[1]++;
                *dstPtr++ = *srcPtr[2]++;   *dstPtr++ = *srcPtr[3]++;
                break;
            case 8:
                *dstPtr++ = *srcPtr[0]++;   *dstPtr++ = *srcPtr[1]++;
                *dstPtr++ = *srcPtr[2]++;   *dstPtr++ = *srcPtr[3]++;
                *dstPtr++ = *srcPtr[4]++;   *dstPtr++ = *srcPtr[5]++;
                *dstPtr++ = *srcPtr[6]++;   *dstPtr++ = *srcPtr[7]++;
                break;
            case 16 :
                *dstPtr++ = *srcPtr[0]++;   *dstPtr++ = *srcPtr[1]++;
                *dstPtr++ = *srcPtr[2]++;   *dstPtr++ = *srcPtr[3]++;
                *dstPtr++ = *srcPtr[4]++;   *dstPtr++ = *srcPtr[5]++;
                *dstPtr++ = *srcPtr[6]++;   *dstPtr++ = *srcPtr[7]++;
                *dstPtr++ = *srcPtr[8]++;   *dstPtr++ = *srcPtr[9]++;
                *dstPtr++ = *srcPtr[10]++;   *dstPtr++ = *srcPtr[11]++;
                *dstPtr++ = *srcPtr[12]++;   *dstPtr++ = *srcPtr[13]++;
                *dstPtr++ = *srcPtr[14]++;   *dstPtr++ = *srcPtr[15]++;
                break;
            default:
                *dstPtr++ = *srcPtr[0]++;   *dstPtr++ = *srcPtr[1]++;
                *dstPtr++ = *srcPtr[2]++;   *dstPtr++ = *srcPtr[3]++;
                *dstPtr++ = *srcPtr[4]++;   *dstPtr++ = *srcPtr[5]++;
                *dstPtr++ = *srcPtr[6]++;   *dstPtr++ = *srcPtr[7]++;
                *dstPtr++ = *srcPtr[8]++;   *dstPtr++ = *srcPtr[9]++;
                *dstPtr++ = *srcPtr[10]++;   *dstPtr++ = *srcPtr[11]++;
                *dstPtr++ = *srcPtr[12]++;   *dstPtr++ = *srcPtr[13]++;
                *dstPtr++ = *srcPtr[14]++;   *dstPtr++ = *srcPtr[15]++;
                *dstPtr++ = *srcPtr[16]++;   *dstPtr++ = *srcPtr[17]++;
                *dstPtr++ = *srcPtr[18]++;   *dstPtr++ = *srcPtr[19]++;
                *dstPtr++ = *srcPtr[20]++;   *dstPtr++ = *srcPtr[21]++;
                *dstPtr++ = *srcPtr[22]++;   *dstPtr++ = *srcPtr[23]++;
                *dstPtr++ = *srcPtr[24]++;   *dstPtr++ = *srcPtr[25]++;
                *dstPtr++ = *srcPtr[26]++;   *dstPtr++ = *srcPtr[27]++;
                *dstPtr++ = *srcPtr[28]++;   *dstPtr++ = *srcPtr[29]++;
                *dstPtr++ = *srcPtr[30]++;   *dstPtr++ = *srcPtr[31]++;
            }
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
                *dstPtr++ = *srcPtr[j]++;
            dstPtr += gtrans.dstWt - remBatch;
        }
    }
}
/*
* Batch generalized transpose for the special dtypeSize = {3} and {5, 6, 7}, respectively.
* BATCH rows are transposed in batch so as to optimize caching
* It utilizes overlapped data movement within, while precise memcpy() at the boundary.
*/
template<typename T, int BATCH>void G1_Transpose(void* src, void* dst, G_Trans_Param gtrans)
{
    const int  remBatch = gtrans.rows % BATCH;
    const uint64_t dtypeSize = gtrans.dtypeSize;
    int i, j, n, s_i;
    uint8_t* dstPtr, * srcPtr[8];

    if (remBatch == 0) {
        for (i = s_i = 0; i < gtrans.rows - BATCH + 1; i += BATCH, s_i += BATCH * gtrans.srcWt) {
            dstPtr = static_cast<uint8_t*>(dst) + i * dtypeSize;               // beginning of column no. i
            srcPtr[0] = static_cast<uint8_t*>(src) + s_i * dtypeSize;          // beginning of row no. i 
            for (n = 1; n < BATCH; n++)         // beginning of rows no. i+1, i+2, ..., i+7 
                srcPtr[n] = srcPtr[0] + n * gtrans.srcWt * dtypeSize;
            for (n = gtrans.cols; n > 0; n--) {  //Transpose the batch of rows in parallel
                switch (BATCH) {        //Note that branching is removed at the compiling time
                case 4:
                    *((T*)dstPtr) = *((T*)srcPtr[0]);     dstPtr += dtypeSize; srcPtr[0] += dtypeSize;   // overlapped data movement
                    *((T*)dstPtr) = *((T*)srcPtr[1]);     dstPtr += dtypeSize; srcPtr[1] += dtypeSize;   // overlapped data movement   
                    *((T*)dstPtr) = *((T*)srcPtr[2]);     dstPtr += dtypeSize; srcPtr[2] += dtypeSize;   // overlapped data movement   
                    memcpy(dstPtr, srcPtr[3], dtypeSize);               srcPtr[3] += dtypeSize;          // avoid over-write on the last data 
                    break;
                default:
                    *(T*)dstPtr = *((T*)srcPtr[0]);     dstPtr += dtypeSize; srcPtr[0] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[1]);     dstPtr += dtypeSize; srcPtr[1] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[2]);     dstPtr += dtypeSize; srcPtr[2] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[3]);     dstPtr += dtypeSize; srcPtr[3] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[4]);     dstPtr += dtypeSize; srcPtr[4] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[5]);     dstPtr += dtypeSize; srcPtr[5] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[6]);     dstPtr += dtypeSize; srcPtr[6] += dtypeSize;
                    memcpy(dstPtr, srcPtr[7], dtypeSize);                    srcPtr[7] += dtypeSize;
                }
                dstPtr += (gtrans.dstWt - BATCH + 1) * dtypeSize;         // jumping to next column index
            }
        }
    }
    else {
        for (i = s_i = 0; i < gtrans.rows - BATCH + 1; i += BATCH, s_i += BATCH * gtrans.srcWt) {
            dstPtr = static_cast<uint8_t*>(dst) + i * dtypeSize;               // beginning of column no. i
            srcPtr[0] = static_cast<uint8_t*>(src) + s_i * dtypeSize;          // beginning of row no. i 
            for (n = 1; n < BATCH; n++)         // beginning of rows no. i+1, i+2, ..., i+7 
                srcPtr[n] = srcPtr[0] + n * gtrans.srcWt * dtypeSize;
            for (n = gtrans.cols; n > 0; n--) {  //Transpose the batch of rows in parallel
                switch (BATCH) {    
                case 4:
                    *(T*)dstPtr = *((T*)srcPtr[0]);     dstPtr += dtypeSize; srcPtr[0] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[1]);     dstPtr += dtypeSize; srcPtr[1] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[2]);     dstPtr += dtypeSize; srcPtr[2] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[3]);                          srcPtr[3] += dtypeSize;
                    break;
                default:
                    *(T*)dstPtr = *((T*)srcPtr[0]);     dstPtr += dtypeSize; srcPtr[0] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[1]);     dstPtr += dtypeSize; srcPtr[1] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[2]);     dstPtr += dtypeSize; srcPtr[2] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[3]);     dstPtr += dtypeSize; srcPtr[3] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[4]);     dstPtr += dtypeSize; srcPtr[4] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[5]);     dstPtr += dtypeSize; srcPtr[5] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[6]);     dstPtr += dtypeSize; srcPtr[6] += dtypeSize;
                    *(T*)dstPtr = *((T*)srcPtr[7]);                          srcPtr[7] += dtypeSize;
                }
                dstPtr += (gtrans.dstWt - BATCH + 1) * dtypeSize;         // jumping to next column index
            }
        }
        // Transpose the residual batch of rows 
        dstPtr = static_cast<uint8_t*>(dst) + i * dtypeSize;               // beginning of column no. i
        srcPtr[0] = static_cast<uint8_t*>(src) + s_i * dtypeSize;          // beginning of row no. i 
        for (n = 1; n < remBatch; n++)
            srcPtr[n] = srcPtr[0] + n * gtrans.srcWt * dtypeSize;
        for (n = gtrans.cols; n > 0; n--) {
            for (j = 0; j < remBatch - 1; j++) {     // remBatch is locally defined as within batch so that compiler learns to unroll this for-loop
                *(T*)dstPtr = *((T*)srcPtr[j]);
                dstPtr += dtypeSize;
                srcPtr[j] += dtypeSize;
            }
            memcpy(dstPtr, srcPtr[j], dtypeSize);           // avoid over-write on the last data        
            srcPtr[j] += dtypeSize;
            dstPtr += (gtrans.dstWt - remBatch + 1) * dtypeSize;
        }
    }
}

/* Generalized batch transpose to optimize 64-byte cache-line utilization.
 * Due to random starting address, 32-byte consecutive write addresses result in 33/64 chance of utilizing a single cache line, while the remaining 31/64 chance is in 2 cache lines.
 * Simulations indicate that the consecutive write address of 32-byte is the sweet spot. 
 */
void Generalized_Transpose(const void *src, void *dst, uint64_t srcOffset, uint64_t dstOffset, const G_Trans_Param gtrans)
{
    uint64_t* srcS8, * dstS8;
    uint32_t* srcS4, * dstS4;
    uint16_t* srcS2, * dstS2;
    uint8_t* srcS1, * dstS1;
    switch (gtrans.dtypeSize) {
    case 1:
        srcS1 = (uint8_t*)src + srcOffset;
        dstS1 = (uint8_t*)dst + dstOffset;
        G_Transpose<uint8_t, 32>(srcS1, dstS1, gtrans);
        break;
    case 2:
        srcS2 = (uint16_t*)src + srcOffset;
        dstS2 = (uint16_t*)dst + dstOffset;
        G_Transpose<uint16_t, 16>(srcS2, dstS2, gtrans);
        break;
    case 4:
        srcS4 = (uint32_t*)src + srcOffset;
        dstS4 = (uint32_t*)dst + dstOffset;
        G_Transpose<uint32_t, 8>(srcS4, dstS4, gtrans);
        break;
    case 8:
        srcS8 = (uint64_t*)src + srcOffset;
        dstS8 = (uint64_t*)dst + dstOffset;
        G_Transpose<uint64_t, 4>(srcS8, dstS8, gtrans);
    case 3: 
        srcS1 = (uint8_t*)src + srcOffset * gtrans.dtypeSize;
        dstS1 = (uint8_t*)dst + dstOffset * gtrans.dtypeSize;
        G1_Transpose<uint32_t, 8>(srcS1, dstS1, gtrans);
        break;
    default:  // cases: 5, 6, 7
        srcS1 = (uint8_t*)src + srcOffset * gtrans.dtypeSize;
        dstS1 = (uint8_t*)dst + dstOffset * gtrans.dtypeSize;
        G1_Transpose<uint64_t, 4>(srcS1, dstS1, gtrans);
        break;
    }
}
void Permute_TypeA_Kernel(const void* src, void* dst, const uint64_t src_ndim, 
        uint64_t * src_dims_new, uint64_t* src_wt_new, uint64_t* dst_wt_new, const G_Trans_Param gtrans)
{
    int i, k;
    uint64_t  src_idx[MAX_DIM] = { 0 };
    uint64_t srcOffset, dstOffset;
    switch (src_ndim) {
    case 2:
        Generalized_Transpose(src, dst, 0, 0, gtrans);
        break;
    case 3:
        for (srcOffset = dstOffset = 0; srcOffset < src_dims_new[0] * src_wt_new[0];
            srcOffset += src_wt_new[0], dstOffset += dst_wt_new[0]) {
            Generalized_Transpose(src, dst, srcOffset, dstOffset, gtrans);
        }
        break;
    default:
        srcOffset = dstOffset = 0;
        while (1) {
            Generalized_Transpose(src, dst, srcOffset, dstOffset, gtrans);

            // iteratively generate next src_idx[]
            // it is equivalent to src_dim-2 layers of nested for-loops.
            k = (int)src_ndim - 3;
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
                for (i = (int)src_ndim - 2; i > k; i--)
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
/*Permute for Type - A: the last dimension is permuted, i.e., perm_idx[ndim - 1] != ndim - 1. 
*EX : perm_idx[3, 1, 2, 0],  perm_idx[3, 0, 1, 2],  perm_idx[4, 5, 0, 1, 2, 3]
* Its main operation is the generalized parallel transpose.
*/
int Permute_TypeA(const void* src, void* dst, uint64_t dtypeSize, const uint64_t src_ndim,
    uint64_t* src_dims, uint64_t* perm_idx) {
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
    gtrans.dtypeSize = (int)dtypeSize;
    gtrans.rows = (int)src_dims[src_order[src_ndim - 2]];
    gtrans.cols = (int)src_dims[src_ndim - 1];
    gtrans.srcWt = (int)src_wt[src_order[src_ndim - 2]];
    gtrans.dstWt = (int)dst_wt[src_ndim-1];
 
    //A key advantage of the following implementation is that each permuted dimension is treated independently. 
    //Moreover, the index computation does not involve multiplications/divisions.
    uint64_t src_dims_new[MAX_DIM] = { 0 };
    uint64_t src_wt_new[MAX_DIM] = { 0 }, dst_wt_new[MAX_DIM] = { 0 };

    for (i = 0; i < src_ndim - 2; i++) {
        src_dims_new[i] = src_dims[src_order[i]];
        src_wt_new[i] = src_wt[src_order[i]];
        dst_wt_new[i] = dst_wt[src_order[i]];
    }
    
    Permute_TypeA_Kernel(src, dst, src_ndim, src_dims_new, src_wt_new, dst_wt_new, gtrans);
    return 0;
}

/*~~~~~~~~~~~~~~~ Multi-thread implementation of Permute_TypeA() ~~~~~~~~~~~~~~~
* The first dimension, [0], is partitioned evenly for each thread. 
* Memory write is guaranteed to be non-overlapping, thus no need to involve mutex.
*/
int Permute_TypeA_MThread(const void* src, void* dst, uint64_t dtypeSize, const uint64_t src_ndim,
    uint64_t* src_dims, uint64_t* perm_idx, int nThreads) {
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
    gtrans.dtypeSize = (int)dtypeSize;
    gtrans.rows = (int)src_dims[src_order[src_ndim - 2]];
    gtrans.cols = (int)src_dims[src_ndim - 1];
    gtrans.srcWt = (int)src_wt[src_order[src_ndim - 2]];
    gtrans.dstWt = (int)dst_wt[src_ndim - 1];

    //A key advantage of the following implementation is that each permuted dimension is treated independently. 
    //Moreover, the index computation does not involve multiplications/divisions.
    uint64_t src_dims_new[MAX_DIM] = { 0 };
    uint64_t src_wt_new[MAX_DIM] = { 0 }, dst_wt_new[MAX_DIM] = { 0 };

    for (i = 0; i < src_ndim - 2; i++) {
        src_dims_new[i] = src_dims[src_order[i]];
        src_wt_new[i] = src_wt[src_order[i]];
        dst_wt_new[i] = dst_wt[src_order[i]];
    }
    std::vector<std::thread> threads(nThreads);
    uint64_t src_dims_1[MAX_DIM], src_dims_2[MAX_DIM];
    uint8_t* srcPtr, * dstPtr;
    memcpy(src_dims_1, src_dims_new, src_ndim * sizeof(uint64_t));
    memcpy(src_dims_2, src_dims_new, src_ndim * sizeof(uint64_t));
    const int rThreads = src_dims_new[0] % nThreads;
    const uint64_t dim0 = src_dims_new[0] / nThreads;
    src_dims_1[0] = dim0 + 1;
    src_dims_2[0] = dim0;
    
    srcPtr = (uint8_t*)src;
    dstPtr = (uint8_t*)dst;
    for (int i = 0; i < rThreads; i++) {
        if (src_ndim == 2) {
            gtrans.rows = dim0 + 1;
        }
        threads[i] = std::thread(Permute_TypeA_Kernel, srcPtr, dstPtr, src_ndim, src_dims_1, src_wt_new, dst_wt_new, gtrans);
        srcPtr += src_dims_1[0] * src_wt_new[0] * dtypeSize;
        dstPtr += src_dims_1[0] * dst_wt_new[0] * dtypeSize;
        if (threads[i].joinable())
            threads[i].join();
    }
    if (dim0 > 0) {
        for (int i = rThreads; i < nThreads; i++) {
            if (src_ndim == 2) {
                gtrans.rows = dim0;
            }
            threads[i] = std::thread(Permute_TypeA_Kernel, srcPtr, dstPtr, src_ndim, src_dims_2, src_wt_new, dst_wt_new, gtrans);
            srcPtr += src_dims_2[0] * src_wt_new[0] * dtypeSize;
            dstPtr += src_dims_2[0] * dst_wt_new[0] * dtypeSize;
            if (threads[i].joinable())
                threads[i].join();
        }
    }

    Permute_TypeA_Kernel(src, dst, src_ndim, src_dims_new, src_wt_new, dst_wt_new, gtrans);
    return 0;
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TYPE B Permute  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The last dimension is not permuted. Memcpy is utilized for move the entire last dim of data.
*/
void Permute_TypeB_Kernel(const void* src, void* dst, uint64_t dtypeSize, const uint64_t src_ndim,
    uint64_t* src_dims, uint64_t* src_wt, uint64_t* dst_wt)
{
    uint8_t* srcPtr, * dstPtr;
    uint64_t srcOffset = 0, dstOffset = 0;
    uint64_t src_idx[MAX_DIM] = { 0 };
    const uint64_t copySize = src_dims[src_ndim - 1] * dtypeSize;
    while (1) {
        srcPtr = (uint8_t*)src + srcOffset * dtypeSize;
        dstPtr = (uint8_t*)dst + dstOffset * dtypeSize;
        memcpy(dstPtr, srcPtr, copySize);

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
*Its main operation is memcpy() over the last entire dimension.
*/
int Permute_TypeB(const void* src, void* dst, uint64_t dtypeSize, const uint64_t src_ndim,
    uint64_t* src_dims, uint64_t* perm_idx) {
    uint64_t src_wt[MAX_DIM], dst_wt[MAX_DIM];

    src_wt[src_ndim - 1] = 1;
    dst_wt[src_ndim - 1] = 1;
    for (int i = (int)src_ndim - 1; i > 0; i--) {
        src_wt[i - 1] = src_wt[i] * src_dims[i];
        dst_wt[perm_idx[i - 1]] = dst_wt[perm_idx[i]] * src_dims[perm_idx[i]];
    }
   
    Permute_TypeB_Kernel(src, dst, dtypeSize, src_ndim, src_dims, src_wt, dst_wt);
    return 0;
}

/* ~~~~~~~~~~~~~~~ Multi-thread implementation of Permute_TypeB() ~~~~~~~~~~~~~~~
* The first dimension, [0], is partitioned evenly for each thread. 
* Memory write is guaranteed to be non-overlapping, thus no need to involve mutex.
*/

int Permute_TypeB_MThread(const void* src, void* dst, uint64_t dtypeSize, const uint64_t src_ndim,
    uint64_t* src_dims, uint64_t* perm_idx, int nThreads) {
    uint64_t src_wt[MAX_DIM], dst_wt[MAX_DIM];

    src_wt[src_ndim - 1] = 1;
    dst_wt[src_ndim - 1] = 1;
    for (int i = (int)src_ndim - 1; i > 0; i--) {
        src_wt[i - 1] = src_wt[i] * src_dims[i];
        dst_wt[perm_idx[i - 1]] = dst_wt[perm_idx[i]] * src_dims[perm_idx[i]];
    }

    std::vector<std::thread> threads(nThreads);
    uint64_t src_dims_1[MAX_DIM], src_dims_2[MAX_DIM];
    uint8_t* srcPtr, *dstPtr;
    memcpy(src_dims_1, src_dims, src_ndim * sizeof(uint64_t));
    memcpy(src_dims_2, src_dims, src_ndim * sizeof(uint64_t));
    const int rThreads = src_dims[0] % nThreads;
    const uint64_t dim0 = src_dims[0] / nThreads;
    src_dims_1[0] = dim0 + 1;
    src_dims_2[0] = dim0;

    srcPtr = (uint8_t*)src;
    dstPtr = (uint8_t*)dst;
    for (int i = 0; i < rThreads; i++) {        
        threads[i] = std::thread(Permute_TypeB_Kernel, srcPtr, dstPtr, dtypeSize, src_ndim, src_dims_1, src_wt, dst_wt);
        srcPtr += src_dims_1[0] * src_wt[0] * dtypeSize;
        dstPtr += src_dims_1[0] * dst_wt[0] * dtypeSize;
        if (threads[i].joinable())
            threads[i].join();
    }
    if (dim0 > 0) {
        for (int i = rThreads; i < nThreads; i++) {
            threads[i] = std::thread(Permute_TypeB_Kernel, srcPtr, dstPtr, dtypeSize, src_ndim, src_dims_2, src_wt, dst_wt);
            srcPtr += src_dims_2[0] * src_wt[0] * dtypeSize;
            dstPtr += src_dims_2[0] * dst_wt[0] * dtypeSize;
            if (threads[i].joinable())
                threads[i].join();
        }
    }
    return 0;
}

int permute(const void* src, void* dst, uint64_t dtypeSize, uint64_t* src_dims,
    uint64_t src_ndim, uint64_t* permute_idx, uint64_t* dst_dims, int nThreads) {
    uint64_t squz_dims[MAX_DIM], tmp_dims[MAX_DIM], trim_dims[MAX_DIM];
    uint64_t perm_idx[MAX_DIM];
    uint64_t perm_off[MAX_DIM] = { 0 };
    int64_t perm_inv[MAX_DIM];
    uint64_t squz_ndim, trim_ndim;
    

    uint64_t i, n;
    /*~~~~~~~~~~~~~~~~ squeeze dims with size 1 ~~~~~~~~~~~~~~~~
    * EX. 1.   perm_idx[2, 3, 5, 4, 1, 7, 0,  6]  ==>  [2, 3, 4, 1, 6, 0,  5]
    *          src_nums[8, 9, 1, 6, 4, 6, 9, 10]  ==>  [8, 9, 6, 4, 6, 9, 10]
    * 
    * EX. 2.   perm_idx[2, 3, 5, 4, 1, 7, 0,  6]  ==>  [1, 2, 3, 0, 5, 4]
    *          src_nums[8, 9, 1, 6, 4, 6, 1, 10]  ==>  [8, 9, 6, 4, 6, 10]
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
        fprintf(stderr, "the input matrix is an array, no need to permute\n");
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
    if (DEBUG) {
        fprintf(stderr, "\nSqueezed dimensions\n");
        for (i = 0; i < squz_ndim; i++)
            fprintf(stderr, "%llu ", perm_idx[i]);
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

    if (DEBUG) {
        fprintf(stderr, "\nPermute after trimming consecutive dimensions\n");
        for (i = 0; i < trim_ndim; i++)
            fprintf(stderr, "%llu ", perm_idx[i]);
        fprintf(stderr, "\ntrimmed dimensions\n");
        for (i = 0; i < trim_ndim; i++)
            fprintf(stderr, "%llu ", trim_dims[i]);
    }
    
    /*In case the last dim is unpermuted, check if it fits into a single data type.
    * EX. perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]
    *     trim_dims[20, 8, 16, 12, 8]   ==> trim_dims[20, 8, 16, 12]
    *     dtypeSize = 1                 ==> dtypeSize = 8
    * 
    * EX. perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]
    *     trim_dims[20, 8, 16, 12, 6]   ==> trim_dims[20, 8, 16, 12]
    *     dtypeSize = 1                 ==> dtypeSize = 6
    */
    if (perm_idx[trim_ndim - 1] == trim_ndim - 1) {
        uint64_t cpSize = trim_dims[trim_ndim - 1] * dtypeSize;  // the size of last dim
        if (cpSize <= 8) {   
            dtypeSize = cpSize;     // treat the last dim as a single data
            trim_ndim--;            // get rid of the last dim
        }
    }
 
    if (1 == trim_ndim) {
        fprintf(stderr, "No need to permute\n");
        return 0;
    }
    if (nThreads == 1) {
        if (perm_idx[trim_ndim - 1] != trim_ndim - 1)
            return Permute_TypeA(src, dst, dtypeSize, trim_ndim, trim_dims, perm_idx);
        else
            return Permute_TypeB(src, dst, dtypeSize, trim_ndim, trim_dims, perm_idx);
    }
    else {
        if (perm_idx[trim_ndim - 1] != trim_ndim - 1)
            return Permute_TypeA_MThread(src, dst, dtypeSize, trim_ndim, trim_dims, perm_idx, nThreads);
        else
            return Permute_TypeB_MThread(src, dst, dtypeSize, trim_ndim, trim_dims, perm_idx, nThreads);
    }
}

void permute_validation()
{
    const uint64_t src_ndim = 8;
    uint64_t dtypeSize = 1;
    //The initial perm_idx and src_dims are guaranteed to be executed
    uint64_t perm_idx[8] = { 4, 5, 6, 3, 0, 1, 2, 7 };  
    uint64_t src_dims[8] = { 4, 5, 6, 8, 9, 10, 12, 7};
    uint64_t perm_inv[MAX_DIM], dst_dims[MAX_DIM];
    uint32_t i, nElmt;
    
    // Initialize a random number generator with the seed
    std::mt19937 g(0);

    const int count = 200;      // number of random tests
    for(int cnt=count; cnt>0; cnt--){
        //Compute the overall number of elements, and the inverse permute index array
        nElmt = 1;
        for (i = 0; i < src_ndim; i++) {
            nElmt *= (uint32_t)src_dims[i];
            perm_inv[perm_idx[i]] = i;
        }

        uint32_t* srcMtx = new uint32_t[nElmt];
        uint32_t* dstMtx = new uint32_t[nElmt];
        for (i = 0; i < nElmt; i++)
            srcMtx[i] = (uint8_t)i;

        // verify the equality permute(inverse_permute) = original
        permute(srcMtx, dstMtx, dtypeSize, src_dims, src_ndim, perm_idx, dst_dims, 4);
        permute(dstMtx, srcMtx, dtypeSize, dst_dims, src_ndim, perm_inv, src_dims, 4);

        for (i = 0; i < nElmt; i++)
            if (srcMtx[i] != (uint8_t)i) {
                fprintf(stderr, "\nPermute[%llu, %llu, %llu, %llu,   %llu, %llu, %llu, %llu] over Tensor[%llu, %llu, %llu, %llu,    %llu, %llu, %llu, %llu] is incorrect",
                    perm_idx[0], perm_idx[1], perm_idx[2], perm_idx[3], perm_idx[4], perm_idx[5], perm_idx[6], perm_idx[7],
                    src_dims[0], src_dims[1], src_dims[2], src_dims[3], src_dims[4], src_dims[5], src_dims[6], src_dims[7]);
                break;
            }
        delete[] srcMtx;
        delete[] dstMtx;

        // Shuffle the permute index array 
        std::shuffle(perm_idx, perm_idx + src_ndim, g); 

        for (i = 0; i < src_ndim; i++) {
            src_dims[i] = rand() % 8 + 1;  // each dimension width in [1, 10]
        }
    }

    std::cout << "Permute has been validated by " << count << " random tests" << std::endl;
}

#define NSEC_IN_SEC (1000000000L)
//#define CLOCK_GETTIME(timestamp) clock_gettime(CLOCK_MONOTONIC, &(timestamp));
#define CLOCK_GETTIME(timestamp)  timespec_get(&timestamp, TIME_UTC)
#define TIMESPEC_TO_NSEC(ts) ((ts).tv_sec * NSEC_IN_SEC + (ts).tv_nsec)

void measure_permute() {

    size_t numIters = 100;
    struct timespec start_time, end_time;
    uint32_t* srcMtx, * dstMtx;
    uint64_t dtypeSize = 4;
    uint64_t perm_idx[] = { 3, 2, 5, 6,   0, 1, 7, 4 };
    uint64_t src_dims[] = { 12, 8, 16, 20,   16, 18, 12, 24 };
    uint64_t src_ndim = 8;
    uint64_t perm_inv[MAX_DIM], dst_dims[MAX_DIM];
    size_t i, nElmt;
    nElmt = 1;
    for (i = 0; i < src_ndim; i++) {
        nElmt *= src_dims[i];
        perm_inv[perm_idx[i]] = i;
    }

    fprintf(stderr, "\nsrc_ndim=%llu, nElmt=%zu, Source dimensions\n", src_ndim, nElmt);
    for (i = 0; i < src_ndim; i++)
        fprintf(stderr, "%llu ", src_dims[i]);
    fprintf(stderr, "\nPermuted dimensions\n");
    for (i = 0; i < src_ndim; i++)
        fprintf(stderr, "%llu ", perm_idx[i]);

    srcMtx = (uint32_t*)malloc(nElmt * dtypeSize);
    dstMtx = (uint32_t*)malloc(nElmt * dtypeSize);
    for (i = 0; i < nElmt; i++)
        srcMtx[i] = (uint32_t)i;

    CLOCK_GETTIME(start_time);
    for (i = numIters; i >0; i--) {
        permute(srcMtx, dstMtx, dtypeSize, src_dims, src_ndim, perm_idx, dst_dims);
    }
    CLOCK_GETTIME(end_time);

    printf("\n%zu iterations with average %zu\n", numIters, (TIMESPEC_TO_NSEC(end_time) - TIMESPEC_TO_NSEC(start_time)) / numIters);
    free(srcMtx);
    free(dstMtx);
}
