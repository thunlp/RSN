import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import numpy as np

x = [40, 48, 56, 64]
model = {}
model['SN-L+CV'] = {'y': [58, 58.5, 58.2, 59], 'color': 'green', 'marker': 'o'}
model['SN-L+V'] = {'y': [56, 57.5, 57.4, 58.1], 'color': 'orange', 'marker': 'd'}
model['SN-L+C'] = {'y': [55.6, 57.2, 56.9, 57.4], 'color': 'brown', 'marker': '>'}
model['SN-L'] = {'y': [40.7, 45.5, 46.3, 47.8], 'color': 'blue', 'marker': '<'}

for name, info in model.items():
    plt.plot(x, info['y'], label=name, linewidth=2, color=info['color'], marker=info['marker'], markersize=10, alpha=0.8, mec='black', mew=0.7)

override = {
        'fontsize': 'medium',
        'verticalalignment': 'top',
        'horizontalalignment': 'center'
    }

my_x_ticks = np.array([40, 48, 56, 64])
plt.axis([39,65,39,61])
plt.xticks(my_x_ticks)

plt.xlabel('Number of Pre-defined Training Relations',fontsize=18)
plt.ylabel('F1 Score', fontsize=18)
plt.legend(numpoints=2, fontsize='large')
plt.grid(linestyle='--')
# plt.show()
plt.savefig('figure4.pdf', format='pdf',bbox_inches = 'tight')


            


