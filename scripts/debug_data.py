import numpy as np
from data_gen import generate

panel = generate(seed=11, n_stocks=600)
f = panel.fields
call = f['implied_volatility_call_10']
put = f['implied_volatility_put_10']

print('call_10 coverage:', np.isfinite(call).mean().round(3))
print('call_10 min:', np.nanmin(call).round(3), 'p1:', np.nanpercentile(call, 1).round(3))
ratio = put / call
print('cells |ratio|>5:', int((np.abs(ratio) > 5).sum()))
print('cells ratio>20:', int((ratio > 20).sum()))
print('max ratio:', np.nanmax(ratio).round(1))

sent = panel.vector_fields['nws12_afterhsz_01l'][0]
cnt = np.isfinite(sent).sum(1)
print('sentiment names covered: min', int(cnt.min()), 'p5', int(np.percentile(cnt, 5)), 'median', int(np.median(cnt)))
print('days with <20 covered:', int((cnt < 20).sum()))
print('days with <8 covered:', int((cnt < 8).sum()))
