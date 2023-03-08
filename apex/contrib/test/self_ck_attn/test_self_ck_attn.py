import torch

import unittest
import numpy as np
import kernel_explorer as ke

from apex.contrib.self_ck_attn import SelfCKAttn

class SelfCKAttnTest(unittest.TestCase):
    def setUp(self, seed=1234):
        self.batch_size    = 64#10
        self.seq_length   = 512#1024
        self.hidden_dim   = 768#1024
        self.num_heads    = 12#16
        self.head_dim     = int(self.hidden_dim/self.num_heads)
        self.dropout_prob = 0.0#not used right now
        self.best_op_id   = 2
        self.scale        = 1/np.sqrt(self.head_dim)#.125
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.tst_inputs_q = np.random.normal(size=(self.batch_size, self.num_heads, self.seq_length, self.head_dim))
        self.tst_inputs_k = np.random.normal(size=(self.batch_size, self.num_heads, self.seq_length, self.head_dim))
        self.tst_inputs_v = np.random.normal(size=(self.batch_size, self.num_heads, self.seq_length, self.head_dim))
        self.out = np.zeros((self.batch_size, self.seq_length, self.num_heads, self.head_dim), dtype=np.float16)
        
        self.dev_inputs_q = ke.DeviceArray(self.tst_inputs_q.astype(np.float16))
        self.dev_inputs_k = ke.DeviceArray(self.tst_inputs_k.astype(np.float16))
        self.dev_inputs_v = ke.DeviceArray(self.tst_inputs_v.astype(np.float16))
        self.dev_out      = ke.DeviceArray(self.out)

        self.ref_layer = SelfCKAttn(self.batch_size,
                                    self.seq_length,
                                    self.hidden_dim,
                                    self.num_heads,
                                    self.head_dim,
                                    dropout=self.dropout_prob,
                                    best_op_id=self.best_op_id)
        #self.ref_layer.cuda().half()
        #self.ref_layer.reset_parameters()


    def test_self_ck_attn(self):
        #grads         = torch.randn_like(self.tst_inputs)

        attn = torch.softmax(torch.matmul(torch.Tensor(self.tst_inputs_q), torch.Tensor(self.tst_inputs_k).transpose(2,3)) * self.scale, dim =-1)
        ref_out = torch.permute(torch.matmul(attn,torch.Tensor(self.tst_inputs_v)), [0, 2, 1, 3]).numpy()

        ref_outputs,_ = self.ref_layer.forward(self.dev_inputs_q,
                                               self.dev_inputs_k,
                                               self.dev_inputs_v,
                                               self.dev_out)
                                               #key_padding_mask=None,
                                               #need_weights=False,
                                               #attn_mask=None,
                                               #is_training=True)
        #out is cpu pointer
        self.dev_out.UpdateHostNumpyArray()

        diff = self.out-ref_out
        print(diff)

        #print(ref_outputs.size())
        #print(self.ref_out.size())
        #print(tst_outputs.size())
        #print(ref_outputs[0][0][0][:])
        #print(self.ref_out[0][0][0][:])
        #print(tst_outputs[0][0][0][:])

        #self.ref_inputs.backward(grads)
        #self.tst_inputs.backward(grads)

        #Inputs Test
        #self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs,  atol=1e-5, rtol=1e-5))
        #self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs.view(self.sequences,self.seq_length,self.hidden_dim),  atol=1e-5, rtol=1e-5))
        #Forward Pass Test
        #self.assertTrue(torch.allclose(self.tst_out, tst_outputs, atol=1e-3, rtol=1e-3))
        self.assertTrue(np.allclose(self.tst_out, tst_outputs, atol=1e-3, rtol=1e-3))
        #Backward Pass Test
        #self.assertTrue(torch.allclose(self.ref_inputs.grad, self.tst_inputs.grad, atol=1e-3, rtol=1e-3))

    #def test_self_multihead_attn_time_mask(self) :
    #    grads         = torch.randn_like(self.tst_inputs)
    #    time_mask_byte= torch.triu(torch.ones(self.tst_inputs.size(0), self.tst_inputs.size(0), device=torch.device("cuda"), dtype=torch.uint8), 1)
    #    time_mask_bool= time_mask_byte.to(torch.bool)

    #    ref_outputs,_ = self.ref_layer.forward(self.ref_inputs,
    #                                           self.ref_inputs,
    #                                           self.ref_inputs,
    #                                           key_padding_mask=None,
    #                                           need_weights=False,
    #                                           attn_mask=time_mask_bool,
    #                                           is_training=True)

    #    tst_outputs,_ = self.tst_layer.forward(self.tst_inputs,
    #                                           self.tst_inputs,
    #                                           self.tst_inputs,
    #                                           key_padding_mask=None,
    #                                           need_weights=False,
    #                                           attn_mask=time_mask_byte,
    #                                           is_training=True)


    #    self.ref_inputs.backward(grads)
    #    self.tst_inputs.backward(grads)

    #    self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs,  atol=1e-5, rtol=1e-5))
    #    self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
    #    self.assertTrue(torch.allclose(self.ref_inputs.grad, self.tst_inputs.grad, atol=1e-3, rtol=1e-3))
    #
    #def test_self_multihead_attn_pad_mask(self) :
    #    grads         = torch.randn_like(self.tst_inputs)
    #    pad_mask_byte = torch.tril(torch.ones(self.tst_inputs.size(1), self.tst_inputs.size(0), device=torch.device("cuda"), dtype=torch.uint8), 1)
    #    pad_mask_bool = pad_mask_byte.to(torch.bool)

    #    ref_outputs,_ = self.ref_layer.forward(self.ref_inputs,
    #                                           self.ref_inputs,
    #                                           self.ref_inputs,
    #                                           key_padding_mask=pad_mask_bool,
    #                                           need_weights=False,
    #                                           attn_mask=None,
    #                                           is_training=True)

    #    tst_outputs,_ = self.tst_layer.forward(self.tst_inputs,
    #                                           self.tst_inputs,
    #                                           self.tst_inputs,
    #                                           key_padding_mask=pad_mask_byte,
    #                                           need_weights=False,
    #                                           attn_mask=None,
    #                                           is_training=True)


    #    self.ref_inputs.backward(grads)
    #    self.tst_inputs.backward(grads)

    #    self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs,  atol=1e-5, rtol=1e-5))
    #    self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
    #    self.assertTrue(torch.allclose(self.ref_inputs.grad, self.tst_inputs.grad, atol=1e-3, rtol=1e-3))

if __name__ == '__main__':
    unittest.main()
