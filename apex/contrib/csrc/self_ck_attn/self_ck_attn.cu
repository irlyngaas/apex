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

//Choose whether to pass pointer or deviceArray object
//#include "fused_attention_deviceArray.hpp"
#include "fused_attention.hpp"
#include "python/tools/kernel_explorer/device_array.h"

namespace ck_attn {
namespace self {

std::vector<torch::Tensor> fwd_cuda(onnxruntime::DeviceArray& query,
                                    onnxruntime::DeviceArray& key,
                                    onnxruntime::DeviceArray& value,
                                    onnxruntime::DeviceArray& out,
                                    int num_sequences,
                                    int seq_length,
                                    int embed_dim,
                                    int num_heads,
                                    int head_dim,
                                    float dropout_prob, const int best_op_id) {


  torch::Tensor attn_outputs = 
      //torch::empty({sequences, seq_len, heads, head_dim});
      //torch::empty({num_sequences, seq_length, num_heads, head_dim}, act_options);
      torch::empty({num_sequences, seq_length, num_heads, head_dim});
  void *attn_outputs_ptr = static_cast<void *>(attn_outputs.data_ptr());

  void *query_ptr = query.ptr();
  void *key_ptr   = key.ptr();
  void *value_ptr = value.ptr();
  void *out_ptr   = out.ptr();
  //fused_attention(num_sequences, num_heads, seq_length, seq_length, head_dim, head_dim, query, key, value, out, best_op_id);
  fused_attention(num_sequences, num_heads, seq_length, seq_length, head_dim, head_dim, query_ptr, key_ptr, value_ptr, out_ptr, best_op_id);


  return { attn_outputs };

}

} // end namespace self
} // end namespace multihead_attn
