import jax
import jax.numpy as jnp

def _domain_length(x):
    dx = x[1] - x[0]
    return x.size * dx, dx

def add_noise(u, noise_std=0.0, seed=0, kind="gaussian"):
    if noise_std <= 0:
        return u
    key = jax.random.PRNGKey(seed)
    if kind == "gaussian":           # N(0, σ^2)
        eps = jax.random.normal(key, shape=u.shape)
        return u + noise_std * eps
    elif kind == "uniform01":        # Uniform in [-σ, +σ]  (your “0–1 sigma” idea)
        eps = jax.random.uniform(key, shape=u.shape, minval=-1.0, maxval=1.0)
        return u + noise_std * eps
    elif kind == "laplace":          # heavier tails
        eps = jax.random.laplace(key, shape=u.shape)  # mean 0, b=1
        return u + noise_std * eps
    else:
        return u  # fallback: no noise

def make_ic(x, kind="sin16", noise_std=0.0, seed=0, noise_kind="gaussian"):
    L, _dx = _domain_length(x)
    if kind == "sin16":
        u0 = jnp.sin(16 * jnp.pi * x / L)
    elif kind == "gaussian":
        u0 = 0.5 * jnp.exp(-100 * (x - L/2.0)**2)
    elif kind == "multi_sine":
        u0 = (jnp.sin(2*jnp.pi*x/L)
              + 0.5*jnp.sin(4*jnp.pi*x/L)
              + 0.25*jnp.sin(8*jnp.pi*x/L))
    elif kind == "cos_small":
        u0 = jnp.cos(3*jnp.pi*x/L + 0.3)/10.0
    else:
        u0 = jnp.zeros_like(x)

    return add_noise(u0, noise_std=noise_std, seed=seed, kind=noise_kind)
