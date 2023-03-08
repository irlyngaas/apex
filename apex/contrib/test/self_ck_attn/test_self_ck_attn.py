import torch

import unittest
import numpy as np

from apex.contrib.self_ck_attn import SelfCKAttn

class SelfCKAttnTest(unittest.TestCase):
    def setUp(self, seed=1234):
        #self.seq_length   = 80
        self.sequences    = 64#10
        self.seq_length   = 512#1024
        self.hidden_dim   = 768#1024
        self.num_heads    = 12#16
        self.head_dim     = int(self.hidden_dim/self.num_heads)
        self.dropout_prob = 0.0#not used right now
        self.best_op_id   = 2
        self.scale        = 1/np.sqrt(self.head_dim)#.125


        self.sequences    = 64#10
        self.seq_length   = 512#1024
        self.hidden_dim   = 768#1024
        self.heads        = 12#16
        self.head_dim     = int(self.hidden_dim/self.heads)
        self.dropout_prob = 0.0
        self.best_op_id   = 2
        self.scale        = 1/np.sqrt(self.head_dim)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.tst_inputs_q = np.random.normal(size=(self.sequences, self.heads, self.seq_length, self.head_dim))
        self.tst_inputs_k = np.random.normal(size=(self.sequences, self.heads, self.seq_length, self.head_dim))
        self.tst_inputs_v = np.random.normal(size=(self.sequences, self.heads, self.seq_length, self.head_dim))
        self.tst_out = np.zeros((self.sequences, self.seq_length, self.heads, self.head_dim), dtype=np.float16)
        
        self.ref_inputs_q = torch.Tensor(self.tst_inputs_q.astype(np.float16)).to(torch.float16).to("cuda:0")
        self.ref_inputs_k = torch.Tensor(self.tst_inputs_k.astype(np.float16)).to(torch.float16).to("cuda:0")
        self.ref_inputs_v = torch.Tensor(self.tst_inputs_v.astype(np.float16)).to(torch.float16).to("cuda:0")
        self.ref_out = torch.Tensor(self.tst_out.astype(np.float16)).to(torch.float16).to("cuda:0")

        self.ref_layer = SelfCKAttn(self.hidden_dim,
                                           self.heads,
                                           dropout=self.dropout_prob,
                                           best_op_id=self.best_op_id)
        #self.ref_layer.cuda().half()
        #self.ref_layer.reset_parameters()

        ## Reset seed so parameters are identical
        #torch.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)
        #self.tst_inputs = torch.randn(self.seq_length, self.sequences, self.hidden_dim,
        #                              dtype=torch.float16, device=torch.device("cuda")).requires_grad_(False)

        #self.tst_inputs_q = self.ref_inputs_q.cpu()
        #self.tst_inputs_k = self.ref_inputs_k.cpu()
        #self.tst_inputs_v = self.ref_inputs_v.cpu()

        #`self.tst_layer = SelfMultiheadAttn(self.hidden_dim,
        #`                                   self.heads,
        #`                                   dropout=self.dropout_prob,
        #`                                   bias=False,
        #`                                   include_norm_add=False,
        #`                                   impl='fast')
                                           #impl='default')
        #self.tst_layer.cuda().half()
        #self.tst_layer.reset_parameters()


    def test_self_ck_attn(self):
        attn = torch.softmax(torch.matmul(self.ref_inputs_q, self.ref_inputs_k.transpose(2,3)) * self.scale, dim =-1)
        tst_outputs = torch.permute(torch.matmul(attn,self.ref_inputs_v), [0, 2, 1, 3])

        ref_outputs,_ = self.ref_layer.forward(self.ref_inputs_q,
                                               self.ref_inputs_k,
                                               self.ref_inputs_v,
                                               self.ref_out)
                                               #key_padding_mask=None,
                                               #need_weights=False,
                                               #attn_mask=None,
                                               #is_training=True)


        print(tst_outputs - self.ref_out)


        #Forward Pass Test
        #self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(self.ref_out, tst_outputs, atol=1e-3, rtol=1e-3))

if __name__ == '__main__':
    unittest.main()
