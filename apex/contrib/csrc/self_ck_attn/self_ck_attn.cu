#include <iostream>
#include <math.h>
#include <vector>
#include <cuda.h>
#include <cuda_fp16.h>
//#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "fused_attention.hpp"

namespace ck_attn {
namespace self {

std::vector<torch::Tensor> fwd_cuda(torch::Tensor const &query,
                                    torch::Tensor const &key,
                                    torch::Tensor const &value,
                                    torch::Tensor const &out,
                                    float dropout_prob, const int best_op_id) {

  const int sequences = query.size(0);
  const int heads = query.size(1);
  const int seq_len = query.size(2);
  const int head_dim = query.size(3);


  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  auto act_options = query.options().requires_grad(false);


  void *query_ptr = static_cast<void *>(query.data_ptr());
  void *key_ptr = static_cast<void *>(key.data_ptr());
  void *value_ptr = static_cast<void *>(value.data_ptr());
  void *out_ptr = static_cast<void *>(out.data_ptr());

  torch::Tensor attn_outputs = 
      //torch::empty({sequences, seq_len, heads, head_dim});
      torch::empty({sequences, seq_len, heads, head_dim}, act_options);
  void *attn_outputs_ptr = static_cast<void *>(attn_outputs.data_ptr());

  //fused_attention(sequences, heads, seq_len, seq_len, head_dim, head_dim, query_ptr, key_ptr, value_ptr, attn_outputs_ptr, best_op_id);
  fused_attention(sequences, heads, seq_len, seq_len, head_dim, head_dim, query_ptr, key_ptr, value_ptr, out_ptr, best_op_id);
  //fused_attention(sequences, heads, seq_len, seq_len, head_dim, head_dim, query.data_ptr<half>(), key.data_ptr<half>(), value.data_ptr<half>(), attn_outputs.data_ptr<half>(), best_op_id);


  return { attn_outputs };

}

} // end namespace self
} // end namespace multihead_attn
