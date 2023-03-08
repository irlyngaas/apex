#include <vector>

#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

namespace ck_attn {
namespace self {

std::vector<torch::Tensor> fwd_cuda(torch::Tensor const &query,
                                    torch::Tensor const &key,
                                    torch::Tensor const &value,
                                    torch::Tensor const &out,
                                    float dropout_prob, const int best_op_id);
                                    //, const int num_blocks, const int block_size_k, 
                                    //const int block_size_o);

//std::vector<torch::Tensor> bwd_cuda(
//    int heads, torch::Tensor const &output_grads,
//    torch::Tensor const &matmul2_results, torch::Tensor const &dropout_results,
//    torch::Tensor const &softmax_results,
//    torch::Tensor const &input_lin_results, torch::Tensor const &inputs,
//    torch::Tensor const &input_weights, torch::Tensor const &output_weights,
//    torch::Tensor const &dropout_mask, float dropout_prob);

std::vector<torch::Tensor>
fwd(torch::Tensor const &query, 
    torch::Tensor const &key, 
    torch::Tensor const &value, 
    torch::Tensor const &out,
    float dropout_prob, const int best_op_id) {
  AT_ASSERTM(query.dim() == 4, "expected 4D tensor");
  AT_ASSERTM(key.dim() == 4, "expected 4D tensor");
  AT_ASSERTM(value.dim() == 4, "expected 4D tensor");

  //AT_ASSERTM(query.type().scalarType() == at::ScalarType::Half,
  //           "Query: Only HALF is supported");
  //AT_ASSERTM(key.type().scalarType() == at::ScalarType::Half,
  //           "Key: Only HALF is supported");
  //AT_ASSERTM(value.type().scalarType() == at::ScalarType::Half,
  //           "Value: Only HALF is supported");
  //AT_ASSERTM(out.type().scalarType() == at::ScalarType::Half,
  //           "Out: Only HALF is supported");

  return fwd_cuda( 
      query,
      key,
      value,
      out,
      dropout_prob, best_op_id);
}
      
//std::vector<torch::Tensor>
//bwd(int heads, torch::Tensor const &output_grads,
//    torch::Tensor const &matmul2_results, torch::Tensor const &dropout_results,
//    torch::Tensor const &softmax_results,
//    torch::Tensor const &input_lin_results, torch::Tensor const &inputs,
//    torch::Tensor const &input_weights, torch::Tensor const &output_weights,
//    torch::Tensor const &dropout_mask, float dropout_prob) {
//  AT_ASSERTM(output_grads.dim() == 3, "expected 3D tensor");
//  AT_ASSERTM(matmul2_results.dim() == 3, "expected 3D tensor");
//  AT_ASSERTM(dropout_results.dim() == 3, "expected 3D tensor");
//  AT_ASSERTM(softmax_results.dim() == 3, "expected 3D tensor");
//  AT_ASSERTM(input_lin_results.dim() == 3, "expected 3D tensor");
//  AT_ASSERTM(inputs.dim() == 3, "expected 3D tensor");
//  AT_ASSERTM(input_weights.dim() == 2, "expected 2D tensor");
//  AT_ASSERTM(output_weights.dim() == 2, "expected 2D tensor");
//  AT_ASSERTM(dropout_mask.dim() == 3, "expected 3D tensor");
//
//  AT_ASSERTM(output_grads.type().scalarType() == at::ScalarType::Half,
//             "Only HALF is supported");
//  AT_ASSERTM(matmul2_results.type().scalarType() == at::ScalarType::Half,
//             "Only HALF is supported");
//  AT_ASSERTM(dropout_results.type().scalarType() == at::ScalarType::Half,
//             "Only HALF is supported");
//  AT_ASSERTM(softmax_results.type().scalarType() == at::ScalarType::Half,
//             "Only HALF is supported");
//  AT_ASSERTM(input_lin_results.type().scalarType() == at::ScalarType::Half,
//             "Only HALF is supported");
//  AT_ASSERTM(inputs.type().scalarType() == at::ScalarType::Half,
//             "Only HALF is supported");
//  AT_ASSERTM(input_weights.type().scalarType() == at::ScalarType::Half,
//             "Only HALF is supported");
//  AT_ASSERTM(output_weights.type().scalarType() == at::ScalarType::Half,
//             "Only HALF is supported");
//  AT_ASSERTM(dropout_mask.type().scalarType() == at::ScalarType::Byte,
//             "Only BYTE is supported");
//
//  return bwd_cuda(heads, output_grads, matmul2_results, dropout_results,
//                  softmax_results, input_lin_results, inputs, input_weights,
//                  output_weights, dropout_mask, dropout_prob);
//}

} // end namespace self
} // end namespace ck_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("self_attn_forward", &ck_attn::self::fwd,
        "Self Multihead Attention Forward.");
//  m.def("self_attn_backward", &ck_attn::self::bwd,
//        "Self Multihead Attention Backward.");
}
