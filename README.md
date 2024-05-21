# Transformers compression

As we all know, the self-attention is computed by 
```python
# x: [N, D]
# w_q, w_k: [D, D1]
# amount of parameters: D * 3 * D1
q = x @ w_q # [N, D1]
k = x @ w_k # [N, D1]
attn = q @ k.T # = x @ (w_q @ w_k.T) @ x.T = x @ w_qk @ x.T
```

To compress the vit model, that is, we can compute `w_qk = w_q @ w_k.T`, conduct SVD on it and discard the zero singular values

Also, we can compress the mapping of values `w_v` and linear projection right after the self-attention `w` into one `w_v @ w`


```python
```

amount of parameters of a Transformer `N = L * (D * 3 * D1 + D * D1 + D1 * D2 + D2 * D) = L * (4 * D * D1 + D1 * D2 + D2 * D)`
`D1 = D, D2 = 4 * D`, therefore, `N = 12 * L * D ** 2`

plus the embedding and unembedding parameters, each `W * D`, while `W` is the vocabulary size, so `N = 12 * L * D ** 2 + 2 * W * D`


## two-layer MLP compression

motivation: there are too many parameters (nearly 2 / 3) in the two-lay MLP

method:
1. Because there is l2 norm in layernorm, which is used beform the MLP and after addition, we can approximate the mapping with higher dimensional spherical harmonics or Fourier series, such that we can reduce the huge overhead entailed by the MLP
2. approximate the mapping with a linear transform such that $Av \approx \lambda u$, where $v$ is the input, $u$ is the output, $A$ is the linear mapping in the space (but not on the sphere) and $\lambda$ an unknown scale factor which can be determined analytically




## Finetune a Tranformer with LDA while without training

Linear probes, which are linear layers appended to the end of a pretrained model, are widely utilized in zero-shot learning. However, this can be solved analytically by LDA, without any training, while attaining an optimal objective. This should be also applicable to finetuning text-generation models, or matching a pair of image model and text model. In the fisrt scenarion, we can solve the linear mapping by least squares, while in the second scenario we can solve it by $Av \approx \lambda u$

