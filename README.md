# vit_compression

As we all know, the self-attention is computed by 
```python
# x: [N, D]
# w_q, w_k: [D, D1]
# amount of parameters: D * 3 * D1
q = x @ w_q # [N, D1]
k = x @ w_k # [N, D1]
attn = q @ k.T # = x @ (w_q @ w_k.T) @ x.T = x @ w_qk @ x.T
```

To compress the vit model, that is, we can compute `w_qk = w_q @ w_k.T`, conduct SVD on it and discard the zero singular values and 


```python
```

amount of parameters of a Transformer `N = L * (D * 3 * D1 + D * D1 + D1 * D2 + D2 * D) = L * (4 * D * D1 + D1 * D2 + D2 * D)`
`D1 = D, D2 = 4 * D`, therefore, `N = 12 * L * D ** 2`

plus the embedding and unembedding parameters, each `W * D`, while `W` is the vocabulary size, so `N = 12 * L * D ** 2 + 2 * W * D`
