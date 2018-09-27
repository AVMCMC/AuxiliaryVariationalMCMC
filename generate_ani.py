import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

DATA_PATH = 'logs/avs/ring/run18/run_results/samples.npy'
SAVE_PATH = 'logs/avs/ring/run18/run_figs/ani.mp4'


data = np.load(DATA_PATH)
fig, ax = plt.subplots(figsize=(5, 5))


sca, = plt.plot([], [], '-o', animated=True)

def init():
    ax.set_xlim((-6, 6))
    ax.set_ylim((-6, 6))
    sca.set_data([], [])
    return (sca,)

def animate(i):
    x = data[:i, 0, 0]
    y = data[:i, 0, 1]
    sca.set_data(x, y)
    return sca,

print('generating animation')
anim = animation.FuncAnimation(fig, func=animate, init_func=init,
                               frames=1000, interval=20, blit=True)

print('saving animation')
anim.save(SAVE_PATH)
print('done')

