import pandas as pd
from mpl_chord_diagram import chord_diagram
import matplotlib.pyplot as plt

dfraw = pd.read_csv('OP66T.csv')
df = dfraw[(dfraw['primary_province'] == 9600) | (dfraw['primary_province'] == 5400)
           | (dfraw['primary_province'] == 4500) | (dfraw['primary_province'] == 7600)]
dfs = df[['hcode', 'hmain_op']]

dfnar = dfs.dropna()
dfc = dfnar.reset_index(drop=True)
dfc['hmain_op'] = dfc['hmain_op'].astype(int)

res = []
for index, row in dfc.iterrows():
    hm_op = row['hmain_op']
    hc = row['hcode']
    count = dfc[(dfc['hmain_op'] == hm_op) & (dfc['hcode'] == hc)].shape[0]
    res.append((hm_op, hc, count))

dfx = pd.DataFrame(res, columns=['hm_op','hc','count'])
dfxp = dfx.pivot_table(values='count', index='hm_op', columns='hc',fill_value=0)

column_titles = dfxp.columns
row_titles = dfxp.index

all = list(set(column_titles).union(set(row_titles)))
s_all = all.sort()
dfxp = dfxp.reindex(columns=all, index=all, fill_value=0)
dflor = dfxp.values.tolist()
hs = dfxp.columns

chord_diagram(dflor, names=hs, order=None, sort="size", directed=False,
                  colors=None, cmap=None, use_gradient=False, chord_colors=None,
                  alpha=0.5, start_at=0, extent=360, width=0.1, pad=2., gap=0.03,
                  chordwidth=0.7, min_chord_width=0, fontsize=5,
                  fontcolor="k", rotate_names=True, ax=None, show=False)
plt.show()