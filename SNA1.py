import pandas as pd
from mpl_chord_diagram import chord_diagram
import matplotlib.pyplot as plt

dfraw = pd.read_csv('OP66T2.csv')
df = dfraw[(dfraw['primary_province'] == 5400)]
# 9600, 5400, 4500, 7600
dfs = df[['hcode', 'hmain_op','hmain']]

dfnar = dfs.dropna()
dfc = dfnar.reset_index(drop=True)
dfc['hmain_op'] = dfc['hmain_op'].astype(int)
dfc['hmain'] = dfc['hmain'].astype(int)

res = []
for index, row in dfc.iterrows():
    hmain_op = row['hmain_op']
    hcode = row['hcode']
    hmain = row['hmain']
    count = dfc[(dfc['hmain_op'] == hmain_op) & (dfc['hcode'] == hcode)].shape[0]
    res.append((hmain_op, hcode, count, hmain))
ures = []
for item in res:
    if item not in ures:
        ures.append(item)

dfx = pd.DataFrame(ures, columns=['hmain_op','hcode','count','hmain'])
dfxp = dfx.pivot_table(values='count', index='hmain_op', columns='hcode',fill_value=0)

print(dfx)


'''

column_titles = dfxp.columns
row_titles = dfxp.index

all = list(set(column_titles).union(set(row_titles)))
s_all = all.sort()
dfxp = dfxp.reindex(columns=all, index=all, fill_value=0)
dflor = dfxp.values.tolist()
hs = dfxp.columns

chord_diagram(dflor, names=hs, order=None, sort="size", directed=True,
                  colors=None, cmap=None, use_gradient=False, chord_colors=None,
                  alpha=0.5, start_at=0, extent=360, width=0.1, pad=2., gap=0.03,
                  chordwidth=0.7, min_chord_width=0, fontsize=5,
                  fontcolor="k", rotate_names=True, ax=None, show=False)
plt.show()

all = list(set(column_titles).union(set(row_titles)))
all.sort
afall = pd.DataFrame(all, columns=['hmain_op'])
mapraw = dfc[['hmain','hmain_op']]
mapad = (mapraw.drop_duplicates()).reset_index(drop=True)

dfmap = pd.merge(afall, mapad, on='hmain_op', how='left')
dfso = dfmap.sort_values(by='hmain')
dfso['hmain'] = dfso['hmain'].astype(int)

dfx = pd.DataFrame(res, columns=['hm_op','hc','count'])
dfxp = dfx.pivot_table(values='count', index='hm_op', columns='hc',fill_value=0)

column_titles = dfxp.columns
row_titles = dfxp.index

all = list(set(column_titles).union(set(row_titles)))
s_all = all.sort()
dfxp = dfxp.reindex(columns=all, index=all, fill_value=0)
dflor = dfxp.values.tolist()
hs = dfxp.columns

chord_diagram(dflor, names=hs, order=None, sort="size", directed=True,
                  colors=None, cmap=None, use_gradient=False, chord_colors=None,
                  alpha=0.5, start_at=0, extent=360, width=0.1, pad=2., gap=0.03,
                  chordwidth=0.7, min_chord_width=0, fontsize=5,
                  fontcolor="k", rotate_names=True, ax=None, show=False)
plt.show()
'''