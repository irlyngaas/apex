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

std::vector<torch::Tensor> fwd_cuda(int heads, torch::Tensor const &inputs,
                                    torch::Tensor const &input_weights,
                                    torch::Tensor const &output_weights,
                                    float dropout_prob, const int best_op_id) {
                                    //, const int num_blocks, const int block_size_k, 
                                    //const int block_size_o) {

  //std::cout << inputs << std::endl;
  //std::cout << input_weights << std::endl;
  const int embed_dim = inputs.size(2);
  const int sequences = inputs.size(1);
  const int q_seq_len = inputs.size(0);
  const int k_seq_len = q_seq_len;
  const int batches = sequences * q_seq_len;
  const int head_dim = embed_dim / heads;
  const int output_lin_dim = 3 * embed_dim;
  //NOT USED: Intermediate Calc for Multihead
  //const int attn_batches = heads * sequences;
  //NOT USED: Used in Multihead for MatVec -> Softmax
  //const int lead_dim = attn_batches * 3 * head_dim;
  const int batch_stride = 3 * head_dim;
  const float alpha = 1.0;
  const float beta = 0.0;

  const int seq_dim = q_seq_len / heads;

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  auto act_options = inputs.options().requires_grad(false);
  torch::Tensor input_lin_results =
      torch::empty({q_seq_len, sequences, output_lin_dim}, act_options);
      
  torch::Tensor attn_outputs = torch::empty_like(inputs, act_options);
  torch::Tensor outputs = torch::empty_like(inputs, act_options);

  //void *q_lin_results_ptr = static_cast<void *>(static_cast<half *>(input_lin_results.data_ptr()));
  void *q_lin_results_ptr = static_cast<void *>(input_lin_results.data_ptr());

  //void *k_lin_results_ptr = static_cast<void *>(
  //    static_cast<half *>(input_lin_results.data_ptr()) + head_dim);
  void *k_lin_results_ptr = static_cast<void *>(
      //static_cast<half *>(input_lin_results.data_ptr()) + head_dim*heads*embed_dim*q_seq_len*sequences);
      static_cast<half *>(input_lin_results.data_ptr()) + head_dim);

  //void *v_lin_results_ptr = static_cast<void *>(
  //    static_cast<half *>(input_lin_results.data_ptr()) + 2 * head_dim);
  void *v_lin_results_ptr = static_cast<void *>(
      //static_cast<half *>(input_lin_results.data_ptr()) + 2 * head_dim*heads*embed_dim*q_seq_len*sequences);
      static_cast<half *>(input_lin_results.data_ptr()) + 2 * head_dim);

  void *attn_outputs_ptr = static_cast<void *>(attn_outputs.data_ptr());
  void *outputs_ptr = static_cast<void *>(outputs.data_ptr());

  rocblas_int flags = 0;

  TORCH_CUDABLAS_CHECK(rocblas_gemm_ex(handle,
                             CUBLAS_OP_T, 
                             CUBLAS_OP_N,
                             output_lin_dim, 
                             batches, 
                             embed_dim,
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(input_weights.data_ptr()),
                             rocblas_datatype_f16_r, 
                             embed_dim,
                             static_cast<const void*>(inputs.data_ptr()),
                             rocblas_datatype_f16_r, 
                             embed_dim, 
                             static_cast<const void*>(&beta),
                             q_lin_results_ptr,
                             rocblas_datatype_f16_r, 
                             output_lin_dim,
                             q_lin_results_ptr,
                             rocblas_datatype_f16_r, 
                             output_lin_dim,
                             rocblas_datatype_f32_r,
                             rocblas_gemm_algo_standard /*algo*/,
                             0 /*solution_index*/,
                             flags));
  //std::cout << input_lin_results << std::endl;

  fused_attention(sequences, heads, q_seq_len, embed_dim, head_dim, seq_dim, q_lin_results_ptr, k_lin_results_ptr, v_lin_results_ptr, attn_outputs_ptr, best_op_id);
  //fused_attention(sequences, heads, q_seq_len, embed_dim, head_dim, head_dim, q_lin_results_ptr, k_lin_results_ptr, v_lin_results_ptr, attn_outputs_ptr, best_op_id);
  //fused_attention(sequences, num_blocks, q_seq_len, embed_dim, block_size_k, block_size_o, q_lin_results_ptr, k_lin_results_ptr, v_lin_results_ptr, outputs_ptr, best_op_id);
  std::cout << attn_outputs << std::endl;
  
  TORCH_CUDABLAS_CHECK(rocblas_gemm_ex(handle,
                             CUBLAS_OP_T, 
                             CUBLAS_OP_N,
                             embed_dim, 
                             batches, 
                             embed_dim,
                             static_cast<const void*>(&alpha),
                             static_cast<const void*>(output_weights.data_ptr()),
                             rocblas_datatype_f16_r, 
                             embed_dim,
                             static_cast<const void*>(attn_outputs.data_ptr()),
                             rocblas_datatype_f16_r, 
                             embed_dim, 
                             static_cast<const void*>(&beta),
                             static_cast<void*>(outputs.data_ptr()),
                             rocblas_datatype_f16_r, 
                             embed_dim,
                             static_cast<void*>(outputs.data_ptr()),
                             rocblas_datatype_f16_r, 
                             embed_dim,
                             rocblas_datatype_f32_r,
                             rocblas_gemm_algo_standard /*algo*/,
                             0 /*solution_index*/,
                             flags));
  return { outputs };

}

} // end namespace self
} // end namespace multihead_attn
