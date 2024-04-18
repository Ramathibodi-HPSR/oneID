import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

dfraw = pd.read_csv('OP66T2.csv')
df = dfraw[(dfraw['primary_province'] == 4500)]
# 9600, 5400, 4500, 7600
dfs = df[['hmain_op', 'hcode','hmain']]

dfnar = dfs.dropna()
dfc = dfnar.reset_index(drop=True)
dfc['hmain_op'] = dfc['hmain_op'].astype(int)
dfc['hmain'] = dfc['hmain'].astype(int)
dfc['hcode'] = dfc['hcode'].astype(int)

G = nx.from_pandas_edgelist(dfc, 'hmain_op', 'hcode')
degree_centrality = nx.degree_centrality(G)
#print(degree_centrality)

unique_groups = dfc['hmain'].unique()
num_groups = len(unique_groups)
colors = plt.cm.rainbow(np.linspace(0, 1, num_groups))

group_colors = dict(zip(unique_groups, colors))
node_groups = dict(zip(dfc['hmain_op'], dfc['hmain']))
node_colors = [group_colors.get(node_groups.get(node, 'default'), 'grey') for node in G.nodes()]

node_size = [1000 * degree_centrality[node] for node in G.nodes()]
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=node_size, node_color=node_colors, font_size=0.2,
        font_weight='bold', edge_color='black', width=0.1)

legend_handles = []
for group, color in group_colors.items():
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markersize=10, label=group, markerfacecolor=color))

#plt.legend(handles=legend_handles, title='hmain', loc='best', ncol=3)

plt.title("Social Network Analysis")
plt.show()


