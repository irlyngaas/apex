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

namespace ck_attn {
namespace self {

std::vector<torch::Tensor> fwd_cuda(int heads, torch::Tensor const &inputs,
                                    torch::Tensor const &input_weights,
                                    torch::Tensor const &output_weights,
                                    float dropout_prob) {
  const int embed_dim = inputs.size(2);
  const int sequences = inputs.size(1);
  const int q_seq_len = inputs.size(0);
  const int k_seq_len = q_seq_len;
  const int batches = sequences * q_seq_len;
  const int head_dim = embed_dim / heads;
  const int output_lin_dim = 3 * embed_dim;
  const int attn_batches = heads * sequences;
  const int lead_dim = attn_batches * 3 * head_dim;
  const int batch_stride = 3 * head_dim;
  const int dropout_elems = attn_batches * q_seq_len * k_seq_len;
  const float alpha = 1.0;
  const float beta = 0.0;
  const float scale = 1.0 / sqrt(static_cast<float>(head_dim));

  void *q_lin_results_ptr = static_cast<void *>(input_lin_results.data_ptr());
  void *k_lin_results_ptr = static_cast<void *>(
      static_cast<half *>(input_lin_results.data_ptr()) + head_dim);
  void *v_lin_results_ptr = static_cast<void *>(
      static_cast<half *>(input_lin_results.data_ptr()) + 2 * head_dim);

}

} // end namespace self
} // end namespace multihead_attn
