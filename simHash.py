import torch
from cupy_kernel import cupyKernel
import numpy as np
from pdb import set_trace

# from config import config
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kernel = '''
extern "C"
__global__ void fingerprint(const float* src, const int k, const int L, long* fp)
{
        // product (N x kL bits) -> Column-Major Order
        // const int L = gridDim.y;
        // const int k = blockDim.x;
        int offset = (k * L * blockIdx.x) + (k * blockIdx.y + threadIdx.x);
	long value = (threadIdx.x >= k || src[offset] <= 0) ? 0 : 1;
        value <<= threadIdx.x;
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
                value |= __shfl_down_sync(0xFFFFFFFF, value, offset, 32);
        }
        if(!threadIdx.x)
        {
                int fp_offset = L * blockIdx.x + blockIdx.y;
                fp[fp_offset] = value;
        }
}
'''

class SimHash:
    def __init__(self, d_, k_, L_, weights=None,seed_=8191):
        self.d = d_
        self.k = k_
        self.L = L_
        self.fp = cupyKernel(kernel, "fingerprint")


        if weights is None:
            self.rp = SimHash.generate(d_, k_, L_, seed_)
        else:
            self.rp = SimHash.generate_from_weight(weights)        

    def generate_from_weight(weights):
        #print("generated from triplet weight")
        return weights.to(device)
    
    def generate(d, k, L, seed):
        return torch.randn(d, k*L).to(device)
        
    # def generate_from_weight(weights):
    #     #print("generated from triplet weight")
    #     matrix = weights
    #     positive = torch.gt(matrix, 0).int()
    #     negative = (matrix < 0.0).int()
    #     result = (positive - negative).float()
    #     #return result.cpu()
    #     return result.to(device)
    
    # def generate(d, k, L, seed):
    #     print("random generate hash table weight")
    #     rand_gen = np.random.RandomState(seed)
    #     matrix = rand_gen.randn(d, k*L)
    #     positive = np.greater_equal(matrix, 0.0)
    #     negative = np.less(matrix, 0.0)
    #     result = positive.astype(np.float32) - negative.astype(np.float32)
    #     return torch.from_numpy(result).to(device)

    # def generate_from_list(srp_list):
    #     matrices = [item.rp for item in srp_list]
    #     return torch.cat(matrices, dim=1)


    # def hash(self, data, transpose=False):
    #     N, D = data.size()
    #     srp = torch.matmul(data, self.rp)
    #     #print("srp", srp)
    #     result = self.fingerprint(srp, N)
    #     #print("result", result)
    #     if transpose:
    #         result = torch.t(result) 
    #     return result

    def hash(self, data, transpose=False):
        N, D = data.size()
        srp = torch.matmul(data, self.rp)
        positive = torch.gt(srp, 0).int()
        negative = (srp < 0.0).int()
        srp = (positive - negative).float()
        result = self.fingerprint(srp, N)
        if transpose:
            result = torch.t(result) 
        return result

    def fingerprint(self, srp, N):
        result = torch.zeros(N, self.L).long().to(device)
        self.fp(grid=(N,self.L,1),
                block=(32,1,1),
                args=[srp.data_ptr(), self.k, self.L, result.data_ptr()],
                strm=torch.cuda.current_stream().cuda_stream)
        return result.int()