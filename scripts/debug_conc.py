import numpy as np
from data_gen import generate
from simulator import _weights_from_signal
from fastexpr import parse, eval_node, Env

panel = generate(seed=11, n_stocks=600)
sig = eval_node(parse('ts_decay_linear(ts_mean(vec_avg(nws12_afterhsz_01l), 5), 10)'), Env(panel))
W = _weights_from_signal(np.where(np.isinf(sig), np.nan, sig))

sent = panel.vector_fields['nws12_afterhsz_01l'][0]
cnt = np.isfinite(sent).sum(1)
mw = np.abs(W).max(1)
thin = np.where((cnt >= 5) & (cnt < 40))[0]
print('days with 5-40 covered names:', len(thin))
for t in thin[:12]:
    print(f'day {t} names={cnt[t]} maxweight={mw[t]:.3f}')
print('overall max weight:', mw.max().round(3), 'on day', int(mw.argmax()), 'names that day:', int(cnt[mw.argmax()]))
print('days maxweight>10%:', int((mw > 0.10).sum()))
