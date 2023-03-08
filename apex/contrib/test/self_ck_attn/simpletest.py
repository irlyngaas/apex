print("HERE", flush=True)
import torch
print("HERE", flush=True)
import numpy as np
import kernel_explorer as ke
print("HERE", flush=True)

from apex.contrib.self_ck_attn import SelfCKAttn

batch_size    = 64#10
seq_length   = 512#1024
hidden_dim   = 768#1024
num_heads    = 12#16
head_dim     = int(hidden_dim/num_heads)
dropout_prob = 0.0#not used right now
best_op_id   = 2
scale        = 1/np.sqrt(head_dim)#.125
print("HERE", flush=True)

tst_inputs_q = np.random.normal(size=(batch_size, num_heads, seq_length, head_dim))
tst_inputs_k = np.random.normal(size=(batch_size, num_heads, seq_length, head_dim))
tst_inputs_v = np.random.normal(size=(batch_size, num_heads, seq_length, head_dim))
out = np.zeros((batch_size, seq_length, num_heads, head_dim), dtype=np.float16)
out2 = np.zeros((batch_size, seq_length, num_heads, head_dim), dtype=np.float16)

attn = torch.softmax(torch.matmul(torch.Tensor(tst_inputs_q), torch.Tensor(tst_inputs_k).transpose(2,3)) * scale, dim =-1)
ref_out = torch.permute(torch.matmul(attn,torch.Tensor(tst_inputs_v)), [0, 2, 1, 3]).numpy()

print("HERE", flush=True)

q = tst_inputs_q.astype(np.float16)
k = tst_inputs_k.astype(np.float16)
v = tst_inputs_v.astype(np.float16)
dev_inputs_q = ke.DeviceArray(q)
dev_inputs_k = ke.DeviceArray(k)
dev_inputs_v = ke.DeviceArray(v)
dev_out      = ke.DeviceArray(out)

print("HERE", flush=True)
ref_layer = SelfCKAttn(batch_size,
                            seq_length,
                            hidden_dim,
                            num_heads,
                            head_dim,
                            dropout=dropout_prob,
                            best_op_id=best_op_id)

print("HERE", flush=True)
ref_outputs,_ = ref_layer.forward(dev_inputs_q,
                                  dev_inputs_k,
                                  dev_inputs_v,
                                  dev_out)
print("HERE", flush=True)
dev_out.UpdateHostNumpyArray()


dev_q = ke.DeviceArray(q)
dev_k = ke.DeviceArray(k)
dev_v = ke.DeviceArray(v)
dev_out2      = ke.DeviceArray(out2)
op = ke.BatchedGemmSoftmaxGemmPermute_half(dev_q, dev_k, dev_v, dev_out2, batch_size, seq_length, num_heads, head_dim, scale)
op.Run()
dev_out2.UpdateHostNumpyArray()

#print("HERE", flush=True)
#attn = torch.softmax(torch.matmul(torch.Tensor(tst_inputs_q), torch.Tensor(tst_inputs_k).transpose(2,3)) * scale, dim =-1)
#ref_out = torch.permute(torch.matmul(attn,torch.Tensor(tst_inputs_v)), [0, 2, 1, 3]).numpy()

print("HERE")
diff = out-ref_out
print(diff)
diff2 = out-out2
print(diff2)
diff3 = out2-ref_out
print(diff3)
