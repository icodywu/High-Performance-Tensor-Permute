This C++ code provides a highly optimized tensor permute function:
int permute(const void* src, void* dst, uint64_t dtypeSize, uint64_t* src_dims,
    uint64_t src_ndim, uint64_t* permute_idx, uint64_t* dst_dims, int nThreads = 1);

It contains 4 optimization steps.
Step 1. Squeeze the trivial dimensions of size = 1. It takes O(n) time and space complexity.
EX.   perm_idx[2, 3, 5, 4, 1, 7, 0,  6]  ==>  [1, 2, 3, 0, 5, 4]
      src_nums[8, 9, 1, 6, 4, 6, 1, 10]  ==>  [8, 9, 6, 4, 6, 10]

EX.   perm_idx[2, 3, 5, 4, 1, 7, 0,  6]  ==>  [1, 2, 3, 0, 5, 4]
      src_nums[8, 9, 1, 6, 4, 6, 1, 10]  ==>  [8, 9, 6, 4, 6, 10]      
      
 
Step 2. Compress the consecutive permuted dimensions. It takes O(n) time and space complexity.
EX.   perm_idx[2, 3,  4,  5,    0, 1,   6, 7 ]  ==>  [1,           0,    2 ]
      src_nums[8, 9,  12, 6,    4, 6,   9, 10]  ==>  [8*9,  12*6*4*6,  9*10]
EX.   perm_idx[2, 3,  4,  5,  6, 7,     0,  1]  ==>  [1,              0  ]
      src_nums[8, 9,  12, 6,  4, 6,     9, 10]  ==>  [8*9,  12*6*4*6*9*10] 
     
EX.   perm_idx[2, 3,   5,  4,   6, 7,   0,  1]  ==>  [1,     3,    2,   4,    0  ]
      src_nums[8, 9,   12, 6,   4, 6,   9, 10]  ==>  [8*9,  12*6,  4,   6,   9*10]        
  
Step 3. In the case where the last dim is unmuted, treat the last small dim, such that, dtypeSize*dim<=8, as a single data type 
         and then remove the last dim.
EX. perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]
    trim_dims[20, 8, 16, 12, 8]   ==> trim_dims[20, 8, 16, 12]
    dtypeSize = 1                 ==> dtypeSize = 8
This case is simply to treat the last dimension of 8 bytes to be the dim size of 1 with dtype of uint64_t. 
    
EX.   perm_idx [2,  1,  3,  0, 4]   ==> perm_idx [ 2, 1,  3,  0]
      trim_dims[20, 8, 16, 12, 7]   ==> trim_dims[20, 8, 16, 12]
      dtypeSize = 1                 ==> dtypeSize = 7
This case is treated differently, wherein the last dim is also reduced to the size of 1 and the dtype of uint64_t is utilized to move data in an overlapped manner. 
To avoid the write conflict under multi-thread, the boundary data is moved precisely by using memcpy of 7 bytes. 
 
Step 4. Apply the proposed generalized batch transpose over the case perm_idx[ndim-1] != ndim-1;
        The proposed generalized batch transpose technique is a generalization of the batch transpose. 
        It takes advantage of cache line by creating column-wise consecutive write addresses (see .  
        For special dtypeSize in {3, 5, 6, 7}, the data movement in an overlapped manner. The details are provided in
        void Generalized_Transpose(const void *src, void *dst, uint64_t srcOffset, uint64_t dstOffset, const G_Trans_Param gtrans)

        Apply memcpy() to move the entire last dim of data over the case perm_idx[ndim-1] == ndim-1. The details are given in 
        void Permute_TypeB_Kernel(const void* src, void* dst, uint64_t dtypeSize, const uint64_t src_ndim,
                                  uint64_t* src_dims, uint64_t* src_wt, uint64_t* dst_wt);
    
        It is worth noting that both types of operations involve nearly no multimplication and completely no division to compute permuted indexes
             as opposed to the straightforword method.
 
By default, it utilizes single thread. Multi-thread is activated by setting the parameter nThreads >1.
By partitioning evenly along the first dim, i.e., [0], the proposed multi-thread operations literally linearly reduces the running time,
until it saturates the memory bandwidth.
 

Function permute_validation() uses the randomly generated tensors and permutations to test against the equality
permute(tensor, perm_idx) -> permuted_tensor
permute(permuted_tensor, perm_inv) = tensor, where perm_inv is the inverse array of perm_idx.
The first set of tensor dimension and perm_idx is hand designed to test a particular corner case
whereas the remaining test cases are randomly generated to expand coverage. 

 Function measure_permute() measures the efficiency of the proposed permute()
