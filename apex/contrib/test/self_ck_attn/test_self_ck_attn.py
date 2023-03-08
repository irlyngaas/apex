import torch

import unittest
import numpy as np

from apex.contrib.self_ck_attn import SelfCKAttn
import copy

class SelfCKAttnTest(unittest.TestCase):
    def setUp(self, seed=1234):
        #self.seq_length   = 80
        self.seq_length   = 1024
        self.sequences    = 10
        self.hidden_dim   = 1024
        self.heads        = 16
        self.head_dim     = int(self.hidden_dim/self.heads)
        self.dropout_prob = 0.0
        self.best_op_id   = 0
        self.scale        = 1/np.sqrt(self.head_dim)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.ref_inputs_q = torch.randn(self.sequences, self.heads, self.seq_length, self.head_dim, dtype=torch.float16, device=torch.device("cuda"))
        self.ref_inputs_k = torch.randn(self.sequences, self.heads, self.seq_length, self.head_dim, dtype=torch.float16, device=torch.device("cuda"))
        self.ref_inputs_v = torch.randn(self.sequences, self.heads, self.seq_length, self.head_dim, dtype=torch.float16, device=torch.device("cuda"))
        #self.ref_inputs_q = np.random.normal(size=(self.sequences, self.heads, self.seq_length, self.head_dim))
        #self.ref_inputs_k = np.random.normal(size=(self.sequences, self.heads, self.seq_length, self.head_dim))
        #self.ref_inputs_v = np.random.normal(size=(self.sequences, self.heads, self.seq_length, self.head_dim))

        
        ##self.ref_out = torch.zeros(self.sequences, self.heads,self.seq_length,self.head_dim).to(torch.float16).to("cuda:0")
        ##self.ref_out = torch.Tensor(self.tst_out).to(torch.float16).to("cuda:0")
        #self.ref_inputs_q = torch.Tensor(self.tst_inputs_q.astype(np.float16)).to(torch.float16)
        #self.ref_inputs_k = torch.Tensor(self.tst_inputs_k.astype(np.float16)).to(torch.float16)
        #self.ref_inputs_v = torch.Tensor(self.tst_inputs_v.astype(np.float16)).to(torch.float16)
        #self.ref_out = torch.zeros(self.sequences, self.heads,self.seq_length,self.head_dim).to(torch.float16)
        ##self.ref_out = torch.Tensor(self.tst_out).to(torch.float16).to("cuda:0")

        #Tried w/ and w/o contiguous
        self.tst_inputs_q = self.tst_inputs_q.contiguous()
        self.tst_inputs_k = self.tst_inputs_k.contiguous()
        self.tst_inputs_v = self.tst_inputs_v.contiguous()
        #self.tst_inputs_q = torch.Tensor(self.tst_inputs_q.astype(np.float16)).to(torch.float16)
        #self.tst_inputs_k = torch.Tensor(self.tst_inputs_k.astype(np.float16)).to(torch.float16)
        #self.tst_inputs_v = torch.Tensor(self.tst_inputs_v.astype(np.float16)).to(torch.float16)
        #Different variation of above
        #self.tst_inputs_q = torch.Tensor(self.tst_inputs_q.astype(np.float16)).to(torch.float16).to("cuda:0")
        #self.tst_inputs_k = torch.Tensor(self.tst_inputs_k.astype(np.float16)).to(torch.float16).to("cuda:0")
        #self.tst_inputs_v = torch.Tensor(self.tst_inputs_v.astype(np.float16)).to(torch.float16).to("cuda:0")


        #Tried w/ and w/o contiguous
        self.tst_out = torch.zeros(self.sequences,self.seq_length,self.heads,self.head_dim, dtype=torch.float16,device=torch.device("cuda")).contiguous()
        #Permute output
        #self.tst_out = torch.zeros(self.sequences, self.heads,self.seq_length,self.head_dim, dtype=torch.float16,device=torch.device("cuda")).contiguous()

        #self.ref_out = np.zeros((self.sequences, self.seq_length, self.heads, self.head_dim), dtype=np.float16)
        #Permute output
        #self.ref_out = np.zeros((self.sequences, self.heads, self.seq_length, self.head_dim), dtype=np.float16)
        #self.tst_out = torch.Tensor(self.ref_out).to(torch.float16).to("cuda:0")

        #Different variations
        #riq = torch.from_numpy(self.ref_out)
        #riqt = torch.Tensor(self.tst_out)

        self.tst_layer = SelfCKAttn(self.hidden_dim,
                                           self.heads,
                                           dropout=self.dropout_prob,
                                           best_op_id=self.best_op_id)
        #self.tst_layer.cuda().half()
        #self.tst_layer.reset_parameters()

    def test_self_ck_attn(self):
        attn = torch.softmax(torch.matmul(self.ref_inputs_q, self.ref_inputs_k.transpose(2,3)) * self.scale, dim =-1)
        ref_outputs = torch.permute(torch.matmul(attn,self.ref_inputs_v), [0, 2, 1, 3])
        #w/o permute
        #tst_outputs = torch.matmul(attn,self.ref_inputs_v)

        #attn = torch.softmax(torch.matmul(torch.Tensor(q), torch.Tensor(k).transpose(2, 3)) * scale, dim=-1)
        #ref_outputs = torch.permute(torch.matmul(attn, torch.Tensor(v)), [0, 2, 1, 3]).numpy()
        #print(ref_outputs.stride())

        tst_outputs,_ = self.tst_layer.forward(self.tst_inputs_q,
                                               self.tst_inputs_k,
                                               self.tst_inputs_v,
                                               self.tst_out)
                                               #key_padding_mask=None,
                                               #need_weights=False,
                                               #attn_mask=None,
                                               #is_training=True)
        #print(self.tst_out.stride(0))
        #print(self.tst_out.stride(1))
        #print(self.tst_out.stride(2))
        #print(self.tst_out.stride(3))
        #print(self.tst_out.cpu().numpy().strides)

        #Attempts to reshape, resize and permute output
        #reffer = self.tst_out.resize(self.sequences,self.heads,self.seq_length,self.head_dim)
        #reffer = torch.reshape(self.tst_out,[0,2,1,3])
        #reffer = reffer.resize_(self.sequences,self.heads,self.seq_length,self.head_dim)
        #reffer2 = torch.permute(reffer,[0,2,1,3])


        print(self.tst_out - ref_outputs)

        #self.ref_inputs.backward(grads)
        #self.tst_inputs.backward(grads)

        #Inputs Test
        #self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs,  atol=1e-5, rtol=1e-5))
        #self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs.view(self.sequences,self.seq_length,self.hidden_dim),  atol=1e-5, rtol=1e-5))

        #Forward Pass Test
        self.assertTrue(torch.allclose(self.tst_out, ref_outputs, atol=1e-3, rtol=1e-3))
        #self.assertTrue(np.allclose(self.tst_out, tst_outputs, atol=1e-3, rtol=1e-3))

if __name__ == '__main__':
    unittest.main()
