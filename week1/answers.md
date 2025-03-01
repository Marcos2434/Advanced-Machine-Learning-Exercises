### **How is the reparametrisation trick handled in the code?**
- The trick is used in `VAE.elbo(x)`, where `q = self.encoder(x)`.
- Sampling is done using `z = q.rsample()`, which allows gradients to propagate through the stochastic variable.

---

### **ELBO Implementation: Dimensions of Key Terms**
1. **`self.decoder(z).log_prob(x)`**
   - `self.decoder(z)` returns a `td.Independent(td.Bernoulli(logits=logits), 2)`, which outputs a **log probability per datapoint**.
   - Dimension: `(batch_size,)` after `torch.mean()` is applied.

2. **`td.kl_divergence(q, self.prior())`**
   - `q` is a `td.Independent(td.Normal(mean, std), 1)`, and `self.prior()` is the same.
   - `td.kl_divergence(q, self.prior())` computes KL divergence per latent variable.
   - Dimension: `(batch_size, latent_dim)`, but `torch.mean()` collapses it.

---

### **Purpose of `td.Independent`**
- Used in `GaussianPrior`, `GaussianEncoder`, and `BernoulliDecoder` to define **independent** probability distributions over multiple dimensions.
- Ensures that:
  1. The latent variables in `p(z)` and `q(z|x)` are independent.
  2. The pixels in the output of `p(x|z)` are modeled independently in a Bernoulli distribution.
- The last argument (`1` or `2`) specifies how many dimensions should be treated as independent.

---
