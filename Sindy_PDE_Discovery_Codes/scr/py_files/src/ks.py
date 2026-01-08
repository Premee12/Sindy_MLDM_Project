import jax
import jax.numpy as jnp

class KuramotoSivashinsky:
    """
    1D KS: u_t = a u_xx + b u_xxxx + c * d/dx (u^2)
    Spectral space (rFFT), ETD1 time step, 2/3 dealiasing.
    """
    def __init__(self, L, N, dt, a=-1.0, b=-1.0, c=-0.5):
        self.L, self.N, self.dt = L, N, dt
        self.a, self.b, self.c = a, b, c
        self.dx = L / N

        # rFFT frequencies (cycles per unit length) and angular k
        freqs = jnp.fft.rfftfreq(N, d=self.dx)
        k = 2 * jnp.pi * freqs
        self._ik = 1j * k

        # Linear operator L(k) = a*(-k^2) + b*(k^4)
        Lk = self.a * (-k**2) + self.b * (k**4)
        self._exp = jnp.exp(dt * Lk)
        self._coef = jnp.where(Lk == 0.0, dt, (self._exp - 1.0) / Lk)

        # 2/3 dealiasing (cut highest third)
        self._alias = (freqs < (2.0/3.0) * jnp.max(freqs))

    def __call__(self, u):
        # Nonlinear term: c * d/dx(u^2)
        u2 = self.c * (u**2)
        u_hat = jnp.fft.rfft(u)
        u2_hat = jnp.fft.rfft(u2) * self._alias
        du2dx_hat = self._ik * u2_hat

        # ETD1 step in Fourier, then back to real
        u_next_hat = self._exp * u_hat + self._coef * du2dx_hat
        return jnp.fft.irfft(u_next_hat, n=self.N)

def run_sim(L=100.0, N=200, dt=0.1, steps=2000, u0=None,
            a=-1.0, b=-1.0, c=-0.5, save_stride=1):
    x = jnp.linspace(0.0, L, N, endpoint=False)
    if u0 is None:
        u0 = jnp.sin(16 * jnp.pi * x / L)

    stepper = jax.jit(KuramotoSivashinsky(L, N, dt, a=a, b=b, c=c))
    u = u0
    trj = [u]  # include t=0
    for i in range(steps):
        u = stepper(u)
        if (i + 1) % save_stride == 0:
            trj.append(u)
    trj = jnp.stack(trj)
    # time vector matches saved frames
    t = jnp.arange(trj.shape[0]) * dt * save_stride
    return trj, x, t
