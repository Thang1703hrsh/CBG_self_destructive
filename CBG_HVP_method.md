## Method: Exact Curvature Gradients via Hessian–Vector Products

### Objective

CBG trains the model parameters $\theta$ on the objective

$$
\mathcal{L}(\theta) \;=\; \mathcal{L}_{\text{aln}}(\theta) \;+\; \alpha\,\mathcal{L}_{\text{bgn}}(\theta) \;+\; \beta \log \eta_k(\theta).
$$

Here $\mathcal{L}_{\text{aln}}$ is the alignment loss on the safe responses, and $\mathcal{L}_{\text{bgn}}$ is the KL divergence $\mathrm{KL}(p_{\theta_0}\,\|\,p_\theta)$ on benign data, taken with respect to the frozen initial model $\theta_0$. The quantity $\eta_k$ measures collateral-barrier geometry. It is computed from the Gram matrix $K \in \mathbb{R}^{E\times E}$ of the per-example measurement gradients over the admissible weights $U$ (all Linear layers of the transformer blocks), using $E = m + n$ examples: $m$ safety pairs and $n$ benign sequences. The first two terms are ordinary losses. The main difficulty is the gradient of $\log \eta_k$.

### Decomposition of $\nabla_\theta \log\eta_k$

Let $\tilde F(\theta)\in\mathbb{R}^E$ be the vector of weighted measurements: per-pair safety margins scaled by $1/\sqrt{m}$, and sampled-Fisher scores of the benign sequences scaled by $1/\sqrt{n}$. Let $c_e = \partial \tilde F_e/\partial\theta|_U$ be the per-example gradients. Write $C$ for their stack, and set $K = C\Lambda^{-1}C^\top$, where $\Lambda$ is the diagonal preconditioner on $U$.

The gradient of $\log\eta_k$ splits into two parts:

- a **preconditioner term** $d_\Lambda$, supported on $U$, which comes from the dependence of $\Lambda$ on $\theta$;
- a **curvature term**, which comes from the dependence of $C$ on $\theta$ and therefore involves second derivatives of $\tilde F$.

We use a shared eigen-sign scalarization of $\partial\log\eta_k/\partial K$. A Rademacher draw $s\in\{\pm1\}^E$ over the spectral decomposition of $K$ yields two vectors $a, b\in\mathbb{R}^E$, and from them the scalar measurement and the direction

$$
\Omega(\theta) \;=\; a^\top \tilde F(\theta),
\qquad
w \;=\; \Lambda^{-1} C^\top b .
$$

The direction $w$ has the shape of the model weights and lives on $U$. Both $a$ and $w$ are held fixed once computed. In expectation over $s$, the curvature term of $\beta\nabla_\theta\log\eta_k$ equals

$$
g_{\text{curv}} \;=\; 2\beta\, H_\Omega(\theta)\, w,
\qquad
H_\Omega = \nabla^2_\theta\,\Omega ,
$$

where $H_\Omega w$ is the Hessian of $\Omega$ in **all** trainable parameters, applied along $w$. The direction $w$ is nonzero only on $U$, but $g_{\text{curv}}$ also reaches parameters outside $U$ (embeddings, normalization layers, the output head) through the mixed second derivatives.

### Exact computation by double backward

Because $w$ is fixed, the Hessian–vector product is the gradient of a directional derivative:

$$
H_\Omega w \;=\; \nabla_\theta\,\big\langle \nabla_U \Omega(\theta),\, w \big\rangle .
$$

We compute it exactly with two reverse-mode passes (Pearlmutter, 1994):

1. **Forward and first backward.** Evaluate $\Omega(\theta)$ and compute $\gamma = \nabla_U\Omega(\theta)$, keeping the graph of this backward pass so it can itself be differentiated.
2. **Directional derivative.** Form the scalar $D(\theta) = \langle \gamma, w\rangle = \sum_{\ell\in U}\langle \gamma_\ell, w_\ell\rangle$, accumulated in float64.
3. **Second backward.** Back-propagate the combined scalar

   $$
   \mathcal{J}(\theta) \;=\; \mathcal{L}_{\text{aln}}(\theta) + \alpha\,\mathcal{L}_{\text{bgn}}(\theta) + 2\beta\, D(\theta)
   $$

   into the parameter gradients. This yields $\nabla(\mathcal{L}_{\text{aln}}+\alpha\mathcal{L}_{\text{bgn}}) + g_{\text{curv}}$ in a single pass.

Adding $\beta\, d_\Lambda$ gives the full CBG gradient, which is passed to the optimizer.

The exact method replaces the central finite difference $\beta\,[\nabla\Omega(\theta+\varepsilon w)-\nabla\Omega(\theta-\varepsilon w)]/\varepsilon$, whose limit as $\varepsilon\to 0$ is the same quantity $2\beta H_\Omega w$. Compared with that estimator, it has four advantages:

- **No radius.** The finite difference requires choosing $\varepsilon$ (through an RMS radius $r$) inside a plateau between truncation error $O(\varepsilon^2)$ and floating-point cancellation $O(u/\varepsilon)$. On Llama-2-7B this plateau is only about a factor of 3 wide ($r\in[6\times10^{-8},\,2\times10^{-7}]$) and may drift during training.
- **No weight perturbation.** The weights are never moved, so no copy of $\theta$ is needed for exact restoration, and the two endpoint passes need not run with deterministic kernels.
- **No catastrophic cancellation.** The two large terms $\pm\nabla\Omega/\varepsilon$ do not appear, so the pass does not have to run in true FP32.
- **The base loss is evaluated at $\theta$ itself** rather than averaged over $\theta\pm\varepsilon w$. As a result, the curvature contribution can be separated exactly from the base gradient.

### Micro-batching and normalization

The $E$ examples are split into micro-batches under a padded-token budget. Safety pairs and benign sequences are never mixed in one micro-batch, so each is padded only to its own longest sequence. All terms are normalized with whole-batch totals: $\sqrt{m}$ and $\sqrt{n}$ for $\tilde F$, and the full response-token counts for $\mathcal{L}_{\text{aln}}$ and $\mathcal{L}_{\text{bgn}}$. Each micro-batch contributes its exact share of $\Omega$, $D$ and $\mathcal{J}$. Because the Hessian–vector product is linear in $\Omega$, summing the per-micro-batch double-backward gradients gives exactly the full-batch $g_{\text{curv}}$. The HVP pass has its own token budget, since the retained first-backward graph roughly doubles to triples activation memory.

### Numerical precision

Second-order quantities are more sensitive to rounding than first-order ones. We therefore check the precision of the HVP pass against an FP32 reference on the target model, using the same batch, measurement samples and sign draw (Table X).

| HVP precision | Relative $\ell_2$ difference to FP32 HVP | Cosine | Time per pass |
|---|---|---|---|
| FP32 | — | — | 24.5 s |
| TF32 (FP32 storage, TF32 matmuls) | 6.5% | 0.998 | 6.8 s |
| BF16 autocast | 75% | 0.74 | 5.8 s |
| Central FD at $r=1.1\times10^{-7}$ (2 FP32 passes) | 1.7% | 0.9999 | 19.4 s |

*Table X. Llama-2-7B-Chat, one CBG batch ($m=8$, $n=32$, 256 tokens), 2×H200.*

BF16 autocast is not accurate enough: the bfloat16 rounding of activations is amplified in the second-order term. TF32 keeps activations in FP32 and differs from the FP32 HVP by an error that is small next to the variance of the single-draw sign estimator. The FP32 finite difference converges to the FP32 HVP as $r$ decreases (2.7%, 1.7% and 1.1% at $r\cdot\{\sqrt3, 1, 1/\sqrt3\}$), which independently confirms both implementations. We use TF32 for the HVP pass and accept a configuration when its estimate agrees with the FP32 HVP within 10% relative $\ell_2$ error and cosine $\ge 0.995$.

### Verification

On a small causal language model in float64, the HVP gradient matches the reference CBG step to within its finite-difference truncation error. Finite-difference estimates converge to the HVP at second order: the relative gap decreases from $4.3\times10^{-3}$ to $4.4\times10^{-9}$ as $r$ decreases from $10^{-2}$ to $10^{-5}$. The estimate is invariant, up to $10^{-10}$, to micro-batching, to placement of layers across devices, and to checkpointed log-softmax evaluation.

### Training procedure

A CBG step computes the Gram matrix $K$ from activations and output gradients captured in one center pass. It then solves the $E\times E$ spectral problem in float64, constructs $w$ and $\beta d_\Lambda$, and runs a single HVP pass. We apply CBG every $K_{\text{lazy}}=4$ optimizer steps. The geometry gradient $\beta(d_\Lambda + 2H_\Omega w)$ is cached in bfloat16 and added to each of the $K_{\text{lazy}}$ steps; every step also computes the base gradient of $\mathcal{L}_{\text{aln}}+\alpha\mathcal{L}_{\text{bgn}}$ in bfloat16. Geometry gradients whose norm exceeds three times the median of the last 16 are rescaled to that bound.

Training is full-parameter with AdamW (learning rate $2\times10^{-5}$, cosine schedule, 10% warmup, 500 steps), with the model split by layers across two GPUs. With identical seeds, the HVP and finite-difference implementations produce matching training curves: after 100 steps, $\Delta\log\eta_k$ on a held-out geometry set is $-0.0586$ for HVP and $-0.0569$ for finite differences. The HVP version reduces the time per CBG step from about 37 s to about 25 s and peak GPU memory from about 124 to about 112 GiB.

### References

- B. A. Pearlmutter. Fast exact multiplication by the Hessian. *Neural Computation*, 6(1):147–160, 1994.
