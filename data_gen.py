import numpy as np
from dataclasses import dataclass

EDGE = {
    'rev5': -0.00045,
    'sent': 0.00011,
    'mom120': 0.00012,
    'skew': -0.00025,
    'quality': 0.00016,
    'lowvol': 0.00008,
    'fmom': 0.00013,
}

REGIME = {2020: 1.5, 2021: 0.5, 2022: 1.0}


@dataclass
class Panel:
    dates: np.ndarray
    fields: dict
    vector_fields: dict
    groups: dict
    subuniverse: np.ndarray


def _roll_mean(x, w):
    T, N = x.shape
    xm = np.where(np.isfinite(x), x, 0.0)
    vm = np.isfinite(x).astype(np.float64)
    cx = np.vstack([np.zeros((1, N)), np.cumsum(xm, 0)])
    cv = np.vstack([np.zeros((1, N)), np.cumsum(vm, 0)])
    S = cx[w:] - cx[:-w]
    C = cv[w:] - cv[:-w]
    out = np.full_like(x, np.nan)
    out[w - 1:] = np.where(C > 0, S / np.maximum(C, 1e-12), np.nan)
    return out


def _roll_std(x, w):
    T, N = x.shape
    xm = np.where(np.isfinite(x), x, 0.0)
    x2 = xm * xm
    vm = np.isfinite(x).astype(np.float64)
    cx = np.vstack([np.zeros((1, N)), np.cumsum(xm, 0)])
    cx2 = np.vstack([np.zeros((1, N)), np.cumsum(x2, 0)])
    cv = np.vstack([np.zeros((1, N)), np.cumsum(vm, 0)])
    S = cx[w:] - cx[:-w]
    S2 = cx2[w:] - cx2[:-w]
    C = cv[w:] - cv[:-w]
    out = np.full_like(x, np.nan)
    mean = np.where(C > 0, S / np.maximum(C, 1e-12), 0.0)
    var = np.maximum(S2 / np.maximum(C, 1e-12) - mean * mean, 0.0)
    out[w - 1:] = np.where(C > 0, np.sqrt(var), np.nan)
    return out


def _cs_z(x):
    m = np.isfinite(x)
    cnt = m.sum(1, keepdims=True)
    s = np.where(m, x, 0.0).sum(1, keepdims=True)
    mean = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    v = np.where(m, (x - mean) ** 2, 0.0).sum(1, keepdims=True)
    sd = np.sqrt(v / np.maximum(cnt - 1, 1))
    sd = np.where(sd > 0, sd, 1.0)
    z = (x - mean) / sd
    return np.where(m, z, np.nan)


def _rank1d(v):
    order = np.argsort(v, kind='stable')
    ranks = np.empty(len(v))
    ranks[order] = np.arange(len(v))
    return ranks / (len(v) - 1)


def _blackout_masks(rng, T, N, day_prob, min_cov, max_dur):
    blackout = np.zeros(T, dtype=bool)
    t = 0
    while t < T:
        if rng.random() < day_prob:
            dur = int(rng.integers(1, max_dur + 1))
            blackout[t:t + dur] = True
            t += dur
        else:
            t += 1
    return blackout


def generate(seed=7, n_stocks=1000, start='2020-01-01', end='2022-12-31'):
    rng = np.random.default_rng(seed)
    all_days = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, 'D'), dtype='datetime64[D]')
    dates = all_days[np.is_busday(all_days)]
    T = len(dates)
    N = n_stocks

    sector = rng.integers(0, 10, N)
    industry = sector * 3 + rng.integers(0, 3, N)
    beta_m = rng.normal(1.0, 0.25, N)
    beta_s = rng.normal(1.0, 0.35, N)
    beta_i = rng.normal(0.5, 0.25, N)
    sig_idio = np.exp(rng.normal(np.log(0.016), 0.35, N))

    f_m = rng.normal(0.0003, 0.008, T)
    f_s = rng.normal(0.0, 0.0035, (T, 10))[:, sector]
    f_i = rng.normal(0.0, 0.0015, (T, 30))[:, industry]
    idio = rng.normal(0.0, 1.0, (T, N)) * sig_idio
    base = beta_m * f_m[:, None] + beta_s * f_s + beta_i * f_i + idio

    size = np.exp(rng.normal(10.0, 1.0, N))
    liq_rank = _rank1d(size)
    illiq = (1.8 - 1.3 * liq_rank)[None, :]
    liqf = (0.5 + liq_rank)[None, :]

    rev5 = np.nan_to_num(_cs_z(_roll_mean(base, 5)), nan=0.0)
    mom120 = np.nan_to_num(_cs_z(_roll_mean(base, 120)), nan=0.0)
    rv20 = _roll_std(base, 20)
    med_rv = np.nanmedian(rv20, axis=0)
    rv20 = np.where(np.isfinite(rv20), rv20, med_rv)
    lowvol_z = np.nan_to_num(_cs_z(rv20), nan=0.0)

    margin = rng.beta(3.0, 12.0, N) * 0.35
    lev = np.clip(rng.normal(0.35, 0.15, N), 0.02, 0.9)
    skew0 = rng.normal(0.03, 0.05, N)
    quality_z = (_rank1d(margin) - 0.5) * 2.0
    skew_z = (_rank1d(skew0) - 0.5) * 2.0

    phi = 0.97
    sent_latent = np.zeros((T, N))
    s = rng.normal(0.0, 1.0, N)
    innov_scale = np.sqrt(1 - phi * phi)
    for t in range(T):
        s = phi * s + innov_scale * rng.normal(0.0, 1.0, N)
        sent_latent[t] = s
    sent_z = np.nan_to_num(_cs_z(sent_latent), nan=0.0)

    years = dates.astype('datetime64[Y]').astype(int) + 1970
    regime = np.array([REGIME.get(int(y), 1.0) for y in years])[:, None]

    nq_pre = int(np.ceil(T / 63))
    margin_q = np.clip(margin[None, :] + rng.normal(0.0, 0.008, (N, nq_pre)).T, 0.01, 0.45)
    dmargin = np.diff(margin_q, axis=0, prepend=margin_q[:1])
    dm_z = np.nan_to_num(_cs_z(dmargin), nan=0.0)
    qidx_pre = np.minimum(np.arange(T) // 63, nq_pre - 1)
    fmom_z = dm_z[qidx_pre, :]

    edge = (EDGE['rev5'] * rev5 * illiq
            + EDGE['sent'] * sent_z * illiq
            + EDGE['mom120'] * mom120
            + EDGE['skew'] * skew_z[None, :]
            + EDGE['quality'] * quality_z[None, :] * liqf
            + EDGE['lowvol'] * (-lowvol_z)
            + EDGE['fmom'] * fmom_z) * regime

    r = base + edge

    p0 = np.exp(rng.uniform(np.log(5.0), np.log(400.0), N))
    close = p0[None, :] * np.cumprod(1.0 + r, axis=0)
    open_ = np.vstack([close[:1], close[:-1]]) * (1.0 + rng.normal(0.0, 0.004, (T, N)))
    rng_hl = np.abs(rng.normal(0.0, 1.0, (T, N))) * sig_idio[None, :] * 0.7
    high = np.maximum(open_, close) * (1.0 + rng_hl)
    low = np.minimum(open_, close) * (1.0 - rng_hl)

    dollar_vol = size[None, :] * (1.0 + 0.3 * np.abs(np.nan_to_num(_cs_z(r)))) * np.exp(rng.normal(0.0, 0.35, (T, N)))
    volume = dollar_vol / close
    adv20 = _roll_mean(dollar_vol, 20)
    shares_out = size / np.exp(rng.normal(3.0, 0.5, N))
    cap = shares_out[None, :] * close

    nq = int(np.ceil(T / 63))
    sales_q = (size * 0.8)[None, :] * np.exp(rng.normal(0.0, 0.06, (N, nq))).T
    turnover_a = rng.uniform(0.4, 1.8, N)
    assets_q = sales_q / turnover_a[None, :]
    debt_q = assets_q * lev[None, :]
    curr_a = rng.uniform(0.25, 0.5, N)
    liab_c = rng.uniform(0.08, 0.3, N)
    qidx = np.minimum(np.arange(T) // 63, nq - 1)

    def _expand(aq):
        return aq[qidx, :]

    sales = _expand(sales_q)
    ebitda = sales * _expand(margin_q)
    assets = _expand(assets_q)
    debt = _expand(debt_q)
    assets_curr = assets * curr_a[None, :]
    liabilities_curr = assets * liab_c[None, :]

    iv_base = rv20 * np.sqrt(252.0) * 100.0 * (1.05 + rng.normal(0.0, 0.08, (T, N))) + 8.0
    # Sec 3.3: options coverage shrinks with shorter tenor + smaller cap.
    # 60-day IV prints broadly; 10-day IV is a strict sparser subset —
    # near-full for TOP500-equivalent names, sharp drop for illiquid names.
    p_iv = 0.10 + 0.85 * liq_rank[None, :] ** 2
    iv_cov = rng.random((T, N)) < p_iv
    iv_black = _blackout_masks(rng, T, 1, 0.008, 0.1, 5)
    iv_cov &= ~iv_black[:, None]
    iv_cov_60 = iv_cov
    short_keep = (0.35 + 0.60 * liq_rank[None, :])  # tenor gradient
    iv_cov_10 = iv_cov_60 & (rng.random((T, N)) < short_keep)

    def _tiny_scale(mask):
        hit = mask & (rng.random((T, N)) < 0.004) & (liq_rank[None, :] < 0.35)
        return np.where(hit, rng.uniform(0.02, 0.1, (T, N)), 1.0)

    call_10 = iv_base * _tiny_scale(iv_cov_10) * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    call_60 = iv_base * _tiny_scale(iv_cov_60) * (1.0 + rng.normal(0.0, 0.05, (T, N)))
    put_10 = iv_base * (1.0 + skew0[None, :]) * _tiny_scale(iv_cov_10) * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    put_60 = iv_base * (1.0 + skew0[None, :]) * _tiny_scale(iv_cov_60) * (1.0 + rng.normal(0.0, 0.05, (T, N)))
    call_10 = np.where(iv_cov_10, call_10, np.nan)
    call_60 = np.where(iv_cov_60, call_60, np.nan)
    put_10 = np.where(iv_cov_10, put_10, np.nan)
    put_60 = np.where(iv_cov_60, put_60, np.nan)

    p_cov = 0.25 + 0.5 * liq_rank[None, :]
    sent_black = _blackout_masks(rng, T, 1, 0.015, 0.01, 18)
    in_black = sent_black[:, None]
    sent_cov = (rng.random((T, N)) < p_cov) & (~in_black | (rng.random((T, N)) < 0.03))
    jumps = np.where(rng.random((T, N)) < 0.002, rng.choice([-1.0, 1.0], (T, N)) * rng.uniform(3.0, 7.0, (T, N)), 0.0)
    sent_obs = np.where(sent_cov, sent_z * 0.9 + rng.normal(0.0, 0.5, (T, N)) + jumps, np.nan)
    vec_parts = [np.where(sent_cov, sent_obs + rng.normal(0.0, 0.35, (T, N)), np.nan) for _ in range(3)]
    buzz = np.where(sent_cov, 1.5 + np.abs(sent_z) * 1.5 + rng.normal(0.0, 0.7, (T, N)), np.nan)

    fields = {
        'close': close,
        'open': open_,
        'high': high,
        'low': low,
        'volume': volume,
        'returns': r,
        'adv20': adv20,
        'cap': cap,
        'ebitda': ebitda,
        'sales': sales,
        'debt': debt,
        'assets': assets,
        'liabilities_curr': liabilities_curr,
        'assets_curr': assets_curr,
        'implied_volatility_call_10': call_10,
        'implied_volatility_call_60': call_60,
        'implied_volatility_put_10': put_10,
        'implied_volatility_put_60': put_60,
        'buzz': buzz,
    }
    vector_fields = {'nws12_afterhsz_01l': vec_parts}
    groups = {'sector': sector, 'industry': industry}
    n_sub = max(1, int(N * 0.5))
    subuniverse = liq_rank >= np.sort(liq_rank)[-n_sub]

    return Panel(dates=dates, fields=fields, vector_fields=vector_fields, groups=groups, subuniverse=subuniverse)
