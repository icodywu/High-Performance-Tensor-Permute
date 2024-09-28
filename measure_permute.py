import torch
import time
import os

# Set environment variables within Python
os.environ['OMP_NUM_THREADS'] = '1'

# Set the number of threads
#torch.set_num_threads(4)


tensor_8d =torch.randint(0, 65535, (12, 32, 8, 16, 24, 16, 20, 3 ), dtype=torch.uint16)
nElmt=1
for i in tensor_8d.size():
    nElmt *= i

# Start measuring time
start_time = time.perf_counter()
cnt = 25
for i in range(cnt):
    x_permuted = tensor_8d.permute(6, 5, 4, 3, 2, 1, 0, 7).contiguous() 

# End measuring time
end_time = time.perf_counter()

# Calculate the elapsed time
elapsed_time = (end_time - start_time)/cnt
thrpt = nElmt*2/elapsed_time/(10**9)
print(f"Time taken to execute the permute[{tensor_8d.size()}]: {elapsed_time:.6f} s, Throughput: {thrpt:.3f} GB/s")

