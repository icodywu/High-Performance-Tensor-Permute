## **Algorithm Description**
This C++ code provides the most efficient implementation for the tensor permute function: \
int permute(const void* src, void* dst, uint64_t dtypeSize, uint64_t* src_dims,
    uint64_t src_ndim, uint64_t* permute_idx, uint64_t* dst_dims, int nThreads = 1);

The code contains 4 unique optimization steps. \
1. Squeeze the trivial dimensions of size = 1. It takes O(n) time and space complexity. \
EX.   perm_idx[2, 3, 5,  4, 1, 7, 0,  6] ==>  [1,  2, 4, 3, 0, 5]   \
      src_nums[1, 9, 12, 6, 4, 6, 1, 10] ==>  [9, 12, 6, 4, 6, 10]  \
EX.   perm_idx[2, 3, 5, 4, 1, 7, 0,  6]  ==>  [2, 3,  5, 4, 1, 6, 0]   \
      src_nums[8, 9, 12, 6, 4, 6, 1, 10] ==>  [8, 9, 12, 6, 4, 6, 10]     
      
 
2. Compress the consecutive permuted dimensions. It takes O(n) time and space complexity. \
EX.   perm_idx[2, 3,  4,  5,    0, 1,   6, 7 ]  ==>  [1,           0,    2 ]  \
      src_nums[8, 9,  12, 6,    4, 6,   9, 10]  ==>  [8x9,  12x6x4x6,  9x10]  \     
EX.   perm_idx[2, 3,  4,  5,  6, 7,     0,  1]  ==>  [1,              0  ]    \
      src_nums[8, 9,  12, 6,  4, 6,     9, 10]  ==>  [8x9,  12x6x4x6x9x10]    \     
EX.   perm_idx[2, 3,   5,  4,   6, 7,   0,  1]  ==>  [1,     3,    2,   4,    0  ]   \
      src_nums[8, 9,   12, 6,   4, 6,   9, 10]  ==>  [8x9,  12x6,  4,   6,   9x10]        
  
3. In the case where the last dim is unmuted, treat the last small dim, such that, dtypeSize*dim<=8, as a single data type 
         and then remove the last dim. \
EX. perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]  \
    trim_dims[20, 8, 16, 12, 8]   ==> trim_dims[20, 8, 16, 12]  \
    dtypeSize = 1                 ==> dtypeSize = 8             \
This case is simply to treat the last dimension of 8 bytes to be the dim size of 1 with dtype of uint64_t. \    
EX.   perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]  \
      trim_dims[20, 8, 16, 12, 7]   ==> trim_dims[20, 8, 16, 12]  \
      dtypeSize = 1                 ==> dtypeSize = 7             \
This case is treated differently, wherein the last dim is also reduced to the size of 1 and the dtype of uint64_t is utilized to move data in an overlapped manner. 
To avoid the write conflict under multi-thread, the boundary data is moved precisely by using memcpy of 7 bytes. 
 
4. **Case 1.** The last dim is permuted, i.e., perm_idx[ndim-1] != ndim-1. \
Fundamentally, such permutation can be viewed as a generalized transpose. 
We thus propose an innovative generalized batch transpose technique, which effectively takes advantage of the cache line by creating column-wise consecutive write addresses.  
For special dtypeSize in {3, 5, 6, 7} (which is created from the merging of the small last dim), the data movement in an overlapped manner. \
The details are provided in void Generalized_Transpose() \
**Case 2.** The last dim is unpermuted, i.e., perm_idx[ndim-1] == ndim-1.
In this case, we deploy memcpy() to move the entire last dim of data (recall the last dim size is coerced to be greater than 8B). \
The details are given in void Permute_TypeB_Kernel(); \    
It is worth noting that both types of operations involve nearly no multiplication and completely no division in computing permuted indexes
as opposed to the straightforward method.
 
By default, it utilizes a single thread. Multi-thread is activated by setting the parameter nThreads >1.
By partitioning evenly along the first dim, i.e., [0], the proposed multi-thread operations linearly reduce the running time,
until it saturates the memory bandwidth.
 
## **Algorithm Validation**
Function permute_validation() uses the randomly generated tensors and permutations to test against the equality
permute(tensor, perm_idx) -> permuted_tensor
permute(permuted_tensor, perm_inv) = tensor, where perm_inv is the inverse array of perm_idx.
The first set of tensor dimensions and perm_idx is hand-designed to test a particular corner case
whereas the remaining test cases are randomly generated to expand coverage. 

## **Performance Measurements and Comparisons**
Function measure_permute() measures the efficiency of the proposed permute(). \

We measured the proposed permute() against the PyTorch torch.permute() (the version on 09/25/2024) under a single thread. \
**System setting**: \
Processor	Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz   2.59 GHz \
Installed RAM	32.0 GB \
System type	64-bit operating system, x64-based processor\


**Tensor: dtype=uint16_t; dims={ 12, 32, 8, 16, 24, 16, 20, 3 }; size=2.1 GB** \
| Permte_idx | This Speed (GB/s) | Torch Speed (GB/s) | Speed Ratio |
|----------|----------|------------|--------|
| {2 0 1 6 7 3 4 5}   |  3.296   | 1.090 | 3.02 |
|  {4 5 6 7 1 2 3 0}  |  1.301   | 0.145  | 8.97 |
| {1 2 3 0 4 5 6 7}   |  8.474   | 3.603  | 2.35 | 
| {4 5 1 2 3 0 6 7}   | 3.237    | 0.743  | 4.36 |
| {6 7 4 5 1 2 3 0}   | 1.204    | 0.138  | 8.72 |
| {6 7 0 1 4 5 2 3}   | 2.801    | 0.263  | 10.65 |
| {6 0 1 4 5 2 3 7}   | 4.083    | 0.217  | 18.82 | 
| {6 5 4 0 1 2 3 7}   | 1.273    | 0.165  | 7.72  |
| {4 5 6 7 0 1 2 3}   | 1.779    | 0.144  | 12.35 |
| {6 5 4 3 2 1 0 7}   | 1.455    | 0.112  | 13.00 |

   

