import utils
import os
import numpy as np
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
file_path = os.path.join(src_dir, 'simulations', 'data1.json')

x = np.linspace(0, 1, 10)

a = 1.1

def y1(x):
    return np.sin(x)

def y2(x):
    return np.sin(a*x)

# fig = plt.figure()


# ax2 = fig.add_subplot(111)
# ax2.plot(x, y1(x), alpha=1.0, linewidth=1.8, color='C0', label="y1")
# ax2.plot(x, y2(x), alpha=1.0, linewidth=1.8, color='C2', label="y2")

# ax2.set_xlabel(xlabel=r"Space x", fontsize=7)  # xlabel
# ax2.set_ylabel(ylabel=r"y", fontsize=7)  # ylabel
# ax2.set_title(r"Compare for metrics", fontsize=7, weight='semibold')
# ax2.set_ylim(bottom=0.0)
# ax2.set_xlim(0, 1.01)
# ax2.legend()
# plt.yticks(fontsize=7)

# plt.grid()
# ax2.set_box_aspect(1)
# plt.savefig(f"{src_dir}/debug_metrics.png")
# plt.show()
# plt.clf()
true = y2(x)
print(utils.compute_metrics(y1(x), y2(x)))
true_nonzero = np.where(true != 0, true, 1e-1)
print(np.abs(y1(x) - true)/true)
