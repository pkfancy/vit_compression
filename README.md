# vit_compression

As we all know, the self-attention is computed by 
```python
# x: [N, D]
# w_q, w_k: [D, D1]
q = x @ w_q # [N, D1]
k = x @ w_k # [N, D1]
attn = q @ k.T # = x @ (w_q @ w_k.T) @ x.T = x @ w_qk @ x.T
```

To compress the vit model, that is, we can compute `w_qk = w_q @ w_k.T`, conduct SVD on it and discard the zero singular values and 


```python
```
