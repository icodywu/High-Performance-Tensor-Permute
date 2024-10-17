## **Permute Algorithm Description**
This C++ code provides the most efficient implementation for the tensor permute function, which beats torch.permute() by 3-20X: \
int permute(const void* src, void* dst, uint64_t dtypeSize, uint64_t* src_dims,
    uint64_t src_ndim, uint64_t* permute_idx, uint64_t* dst_dims, int nThreads = 1);

**The code/algorithm incorporates 4 unique optimization steps.** 
1. Squeeze the trivial dimensions of size = 1. It takes O(n) time and space complexity. \
EX.   perm_idx[2, 3, 5,  4, 1, 7, 0,  6] ==>  [1,  2, 4, 3, 0, 5]   \
      src_nums[1, 9, 12, 6, 4, 6, 1, 10] ==>  [9, 12, 6, 4, 6, 10]  \
EX.   perm_idx[2, 3, 5, 4, 1, 7, 0,  6]  ==>  [2, 3,  5, 4, 1, 6, 0]   \
      src_nums[8, 9, 12, 6, 4, 6, 1, 10] ==>  [8, 9, 12, 6, 4, 6, 10]     
      
 
2. Compress the consecutive permuted dimensions. It takes O(n) time and space complexity. \
EX.   perm_idx[2, 3,  4,  5,    0, 1,   6, 7 ]  ==>  [1,           0,    2 ]  \
      src_nums[8, 9,  12, 6,    4, 6,   9, 10]  ==>  [8x9,  12x6x4x6,  9x10]       
EX.   perm_idx[2, 3,  4,  5,  6, 7,     0,  1]  ==>  [1,              0  ]    \
      src_nums[8, 9,  12, 6,  4, 6,     9, 10]  ==>  [8x9,  12x6x4x6x9x10]         
EX.   perm_idx[2, 3,   5,  4,   6, 7,   0,  1]  ==>  [1,     3,    2,   4,    0  ]   \
      src_nums[8, 9,   12, 6,   4, 6,   9, 10]  ==>  [8x9,  12x6,  4,   6,   9x10]        
  
3. In the case where the last dim is unmuted, treat the last small dim, such that, dtypeSize*dim<=16, as a single data type 
         and then remove the last dim. \
EX. perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]  \
    trim_dims[20, 8, 16, 12, 16]   ==> trim_dims[20, 8, 16, 12]  \
    dtypeSize = 1                 ==> dtypeSize = 16             \
This case is simply to treat the last dimension of 16 bytes to be the dim size of 1 wherein memcpy() is utilized to copy data. \    
EX.   perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]  \
      trim_dims[20, 8, 16, 12, 7]   ==> trim_dims[20, 8, 16, 12]  \
      dtypeSize = 1                 ==> dtypeSize = 7             \
This case is treated differently, wherein the last dim is also reduced to the size of 1 and the dtype of uint64_t is utilized to move data in an overlapped manner. 
To avoid the write conflict under multi-thread, the boundary data is moved precisely by using memcpy of 7 bytes. 
 
4. **Case 1.** The last dim is permuted, i.e., perm_idx[ndim-1] != ndim-1. \
Fundamentally, such permutation can be viewed as a generalized transpose. 
We thus propose an innovative generalized batch transpose technique, which effectively takes advantage of the cache line by creating column-wise consecutive write addresses.  
For special dtypeSize in {3, 5, 6, 7} (created from the merging of the small last dim), the data movement is achieved in an overlapped manner. \
For special dtypeSize in {9, ..., 16} (created from the merging of the small last dim), the data movement utilizes memcpy(). \
The details are provided in void Generalized_Transpose()

4. **Case 2.** The last dim is unpermuted, i.e., perm_idx[ndim-1] == ndim-1.
In this case, we deploy memcpy() to move the entire last dim of data (recall the last dim size is coerced to be greater than 8B). \
The details are given in void Permute_TypeB_Kernel(); \    
It is worth noting that both types of operations involve nearly no multiplication and completely no division in computing permuted indexes
as opposed to the straightforward method.
 
By default, it utilizes a single thread. Multi-thread is activated by setting the parameter nThreads >1.
By partitioning evenly along the first dim, i.e., [0], the proposed multi-thread operations linearly reduce the running time,
until it saturates the memory bandwidth.

## **Combining Permute and Dtype-Conv**
permute_dtypeConv() combines the above-optimized tensor permutation and the dtype conversion, effectively eliminating twice data movements as well as an intermediate buffer. 
The essential connection lies in the following macro function: 
```
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
```
The above macro function eliminates the punishing cost related to the function call by using macros and branching using constexpr.   

## **Validation**
Function permute_validation() uses the randomly generated tensors and permutations to test against the equality
permute(tensor, perm_idx) -> permuted_tensor
permute(permuted_tensor, perm_inv) = tensor, where perm_inv is the inverse array of perm_idx.
The first set of tensor dimensions and perm_idx is hand-designed to test a particular corner case
whereas the remaining test cases are randomly generated to expand coverage. 

## **Performance Measurements and Comparisons**
Function measure_permute() measures the efficiency of the proposed permute(). 

We measured the proposed permute() against the PyTorch torch.permute() (the version on 09/25/2024) under a single thread. \
**System setting**: \
Processor	Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz   2.59 GHz \
Installed RAM	32.0 GB \
System type	64-bit operating system, x64-based processor

**Tensor: dtype=uint16_t, dims={ 12, 32, 8, 16, 24, 16, 20, 3 }, size=2.1 GB** 
| Permute | This Thrpt     | This Thrpt     | This Thrpt     | This Thrpt     | Torch Thrpt   | Boost |
|----------|----------|----------|----------|----------|------------|--------|
| Order           | 1-Thrd (GB/s) | 2-Thrd (GB/s) | 4-Thrd (GB/s) | 8-Thrd (GB/s) | 1-Thrd (GB/s) | 1-Thrd(X) | 
| (3 4 0 1 2 5 6 7)   | 6.949    | 8.221 | 9.603  | 9.630 | 3.157  | 2.20 |
| (6 7 3 4 5 0 1 2)   | 1.649   | 2.597 | 3.284  | 3.550 | 0.149  | 11.07 |
| (2 0 1 6 7 3 4 5)   |  3.296   | 5.646  | 8.281  | 9.056 | 1.090 | 3.02 |
|  (4 5 6 7 1 2 3 0)  |  1.301   | 2.020  | 2.433 | 2.549 | 0.145  | 8.97 |
| (1 2 3 0 4 5 6 7)   |  8.474   | 9.597 | 10.137 | 10.323 | 3.603  | 2.35 | 
| (4 5 1 2 3 0 6 7)   | 3.237    | 4.427  | 5.765 | 5.852 | 0.743  | 4.36 |
| (6 7 4 5 1 2 3 0)   | 1.204    | 1.630 | 2.199 | 2.461 | 0.138  | 8.72 |
| (6 7 0 1 4 5 2 3)   | 2.801    |  4.666  | 6.841 | 7.525  | 0.263  | 10.65 |
| (6 0 1 4 5 2 3 7)   | 4.083    | 6.534 | 7.834 | 8.165 | 0.217  | 18.82 | 
| (6 5 4 0 1 2 3 7)   | 1.273    | 2.201  | 3.664 | 4.764 | 0.165  | 7.72  |
| (4 5 6 7 0 1 2 3)   | 1.779    | 3.043 | 4.397 | 4.497 | 0.144  | 12.35 |
| (6 5 4 3 2 1 0 7)   | 1.455    | 2.647 | 3.946 |  4.309 | 0.112  | 13.00 |



___

       
**Tensor: dtype=uint16_t, dims={ 12, 32, 28, 16, 24, 16, 20, 4 }, size=9.9 GB** 
| Permute | This Thrpt     | This Thrpt     | This Thrpt     | This Thrpt     | Torch Thrpt   | Boost |
|----------|----------|----------|----------|----------|------------|--------|
| Order           | 1-Thrd (GB/s) | 2-Thrd (GB/s) | 4-Thrd (GB/s) | 8-Thrd (GB/s) | 1-Thrd (GB/s) | 1-Thrd(X) | 
| (3 4 0 1 2 5 6 7)   | 7.272    | 8.874   |  9.159 |  9.260 | 0.241 | 30.17 |
| (6 7 3 4 5 0 1 2)   |  1.716   | 2.682   | 3.519 | 4.985 | 0.123  | 13.95 |
| (2 0 1 6 7 3 4 5)   |  2.846   |  5.196 | 8.213  |  9.032  | 0.208   | 13.68 |
|  (4 5 6 7 1 2 3 0)  |  0.941   | 1.442 | 1.608 | 1.718  | 0.088   | 10.69 |
| (1 2 3 0 4 5 6 7)   |  8.580   | 10.108 | 10.497  | 10.505 | 0.192   | 44.69 | 
| (4 5 1 2 3 0 6 7)   | 4.051    | 5.563 | 7.236 | 7.341 | 0.201   | 20.15 |
| (6 7 4 5 1 2 3 0)   | 1.048    | 1.429  | 1.600 | 1.850 | 0.092  | 11.39 |
| (6 7 0 1 4 5 2 3)   | 2.628    | 4.277 | 5.528 | 5.553 | 0.182  | 14.44 |
| (6 0 1 4 5 2 3 7)   | 2.650    | 4.824 | 5.393 | 5.791  | 0.167  | 15.87 | 
| (6 5 4 0 1 2 3 7)   | 1.154    | 1.711  | 2.207  | 2.443 | 0.138  | 8.36  |
| (4 5 6 7 0 1 2 3)   | 1.656    | 1.913 | 2.176 | 2.601 | 0.084  | 19.71 |
| (6 5 4 3 2 1 0 7)   | 1.437    | 2.509 | 3.662 | 4.772 | 0.069  | 20.83 |


**Key Observations** 
1. The proposed permute() achieves high performance regardless of permute order, while torch.permute() varies dramatically.
2. The proposed permute() maintains high performance even when memory is strained (the second case utilizes 20 GB out of 32 GB DRAM), whereas torch.permute() suffers from a strained memory.
3. The proposed permute() outperforms the benchmark torch.permute() by 3-20X in single thread in all test cases.
4. The proposed permute() (nearly) saturates performance with 4-thread in all test cases. For torch.permute(), under both setups of os.environ['OMP_NUM_THREADS'] = '4' and torch.set_num_threads(4), the resulting throughput is similar or even worse than that of single-thread. 



