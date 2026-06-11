import numpy as np
import matplotlib.pyplot as plt

def polar_plot(theta_funcs, n=100):
    r = np.linspace(0, 1, n)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    for theta in theta_funcs:
        t = theta(r)
        t_scaled = t/np.max(t) * .1 * np.pi
        ax.plot(t_scaled, r, lw=2)
        ax.set_rmax(1.0)
        ax.grid(True)
    plt.show()

# Example
funcs = [lambda r: -r**0 + 1, lambda r: -r**1 + 1,
         lambda r: -r**2 + 1, lambda r: -r**3 + 1,
         lambda r: -r**4 + 1, lambda r: -r**5 + 1,
         # lambda r: -r**6 + 1, lambda r: -r**7 + 1,
         # lambda r: -r**8 + 1, lambda r: -r**9+1,
         # lambda r: -r**10+1, lambda r: -r**11+1,
         # lambda r: -r**12+1, lambda r: -r**13+1,
         # lambda r: -r**14+1, lambda r: -r**15+1,
         # lambda r: -r**16+1, lambda r: -r**17+1,
         # lambda r: -r**18+1, lambda r: -r**19+1,
         ]
#
# funcs2 = [lambda r: -2.25*r+2.25, ]
polar_plot(funcs)
# polar_plot(lambda r: -2.25*r+2.25)


