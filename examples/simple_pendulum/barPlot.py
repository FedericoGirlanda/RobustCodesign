from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use("WebAgg")
import pandas
import numpy as np


df = pandas.DataFrame(dict(graph=["RTC (V)", "RTC (Lw)", "Grid Search (V)","RTCD (V)", "RTCD (Lw)"],
                           n=[0.45, 0.53, 3.97, 2.27, 2.13], m=[1.29, 0.79, 1.53, 6.58, 1.07], l=[2.87, 1.49, 0.38, 2.9, 0.5])) 

# Figure Size
fig, ax = plt.subplots(figsize =(16, 9))

# Horizontal Bar Plot
ind = np.arange(len(df))
width = 0.2
ax.barh(ind, df.n, width,  label=r"$time\ (h)$")
ax.barh(ind + width, df.m, width, label=r"$volume\ increment\ ratio$")
ax.barh(ind + 2*width, df.l, width,  label=r"$\frac{volume\ increment\ ratio}{time\ (h)}$")

# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)

# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)

# Add x, y gridlines
ax.grid(visible = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)

# Show top values
ax.invert_yaxis()

# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.05,
            str(round((i.get_width()), 2)),
            fontsize = 10, fontweight ='bold',
            color ='grey')

ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
ax.legend(fontsize=15)
#ax.set_xlabel(r"$\frac{volume\ increment(\%)}{time}$")

# Show Plot
plt.show()