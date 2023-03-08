import torch

#import ck_multihead_attn
import self_ck_attn

class CKSelfAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        out,
        num_sequences,
        seq_length,
        embed_dim,
        num_heads,
        head_dim,
        dropout_prob,
        best_op_id,
    ):
        dropout_prob_t = torch.tensor([dropout_prob])
        null_tensor = torch.tensor([])
        #mask_additive_t = torch.tensor([mask_additive])
        (
            #input_lin_results,
            #softmax_results,
            #dropout_results,
            #dropout_mask,
            #matmul2_results,
            outputs,
        ) = self_ck_attn.self_attn_forward(
            query,
            key,
            value,
            out,
            num_sequences,
            seq_length,
            embed_dim,
            num_heads,
            head_dim,
            dropout_prob,
            best_op_id,
        )
        #ctx.save_for_backward(
        #    use_biases_t,
        #    heads_t,
        #    matmul2_results,
        #    dropout_results,
        #    softmax_results,
        #    null_tensor,
        #    null_tensor,
        #    mask_additive_t,
        #    input_lin_results,
        #    inputs,
        #    input_weights,
        #    output_weights,
        #    dropout_mask,
        #    dropout_prob_t,
        #)
        return outputs.detach()

#    def backward(ctx, output_grads):
#        (
#            use_biases_t,
#            heads_t,
#            matmul2_results,
#            dropout_results,
#            softmax_results,
#            bmm1_results,
#            pad_mask,
#            mask_additive_t,
#            input_lin_results,
#            inputs,
#            input_weights,
#            output_weights,
#            dropout_mask,
#            dropout_prob_t,
#        ) = ctx.saved_tensors
#
#            input_bias_grads = None
#            output_bias_grads = None
#            input_grads, input_weight_grads, output_weight_grads = ck_multihead_attn.self_attn_backward(
#                heads_t[0],
#                output_grads,
#                matmul2_results,
#                dropout_results,
#                softmax_results,
#                input_lin_results,
#                inputs,
#                input_weights,
#                output_weights,
#                dropout_mask,
#                dropout_prob_t[0],
#            )
#            # fast_self_multihead_attn.backward(                          \
#        return (
#            None,
#            None,
#            None,
#            input_grads,
#            input_weight_grads,
#            output_weight_grads,
#            input_bias_grads,
#            output_bias_grads,
#            None,
#            None,
#            None,
#        )



self_attn_func = CKSelfAttnFunc.apply
