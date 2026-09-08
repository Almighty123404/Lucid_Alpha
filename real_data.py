import os
import pickle

import numpy as np
import yfinance as yf

from data_gen import Panel, _roll_mean, _roll_std, _cs_z, _rank1d, _blackout_masks, validate_panel, fingerprint_panel


DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","AVGO","LLY",
    "JPM","UNH","V","MA","HD","COST","PG","JNJ","ABBV","CRM",
    "BAC","XOM","WMT","NFLX","ORCL","CVX","KO","AMD","TMO","ACN",
    "LIN","PEP","MRK","CSCO","ADBE","WFC","MCD","ABT","GE","CAT",
    "GS","IBM","MS","TXN","QCOM","INTU","AMGN","BKNG","ISRG","GILD",
    "HON","AMAT","VRTX","ADI","LRCX","REGN","MDLZ","ADP","CHTR","MU",
    "PANW","KLAC","SNPS","CDNS","CSX","MAR","ORLY","MCHP","CTAS","ADSK",
    "AON","AZO","ROP","PAYX","NUE","WDAY","GWW","CPRT","MNST","BIIB",
    "TT","CMG","DXCM","IDXX","FTNT","ANSS","ODFL","FAST","CTSH","VRSK",
    "BK","AXP","C","BLK","SPG","SCHW","CB","MMC","PGR","AIG",
    "MET","PRU","ALL","TRV","AFL","HIG","PFG","LNC","BEN","IVZ",
    "TROW","NTRS","STT","AMP","NTRS","UNM","GL","RJF","CINF","RE",
    "DUK","SO","NEE","D","AEP","SRE","EXC","XEL","PEG","ED",
    "WEC","ES","ETR","CMS","NI","LNT","AEE","FE","PPL","DTE",
    "TSLA","F","GM","HMC","TM","STLA","RIVN","LCID","NIO","XPEV",
    "LI","BABA","JD","PDD","BIDU","NTES","TME","VIPS","IQ","BILI",
    "SE","MELI","NU","SHOP","SQ","PYPL","COIN","HOOD","SOFI","AFRM",
    "UBER","LYFT","DASH","ABNB","EXPE","RCL","CCL","NCLH","MAR","HLT",
    "WYNN","MGM","LVS","CZR","PENN","DKNG","DIS","CMCSA","FOXA","NFLX",
    "WBD","PARA","DISCK","CHTR","TMUS","VZ","T","CMCSA","LUMN","DISH",
]


def _download_ohlcv(tickers, start, end):
    tickers = list(dict.fromkeys(tickers))
    data = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True, group_by="ticker",
    )
    if data.empty:
        raise RuntimeError("yfinance returned empty data")
    if len(tickers) == 1:
        t = tickers[0]
        cols = {}
        for k in ("Open", "High", "Low", "Close", "Volume"):
            if k in data.columns:
                cols[k.lower()] = data[k].values[:, None]
        idx = data.index.values.astype("datetime64[D]")
        return idx, cols, tickers

    idx = data.columns.levels[1].values if hasattr(data.columns, "levels") else None
    dates = data.index.values.astype("datetime64[D]")
    T = len(dates)
    N = len(tickers)
    opens = np.full((T, N), np.nan)
    highs = np.full((T, N), np.nan)
    lows = np.full((T, N), np.nan)
    closes = np.full((T, N), np.nan)
    volumes = np.full((T, N), np.nan)
    for j, t in enumerate(tickers):
        try:
            sub = data[t] if t in data.columns.get_level_values(0) else None
            if sub is None or sub.empty:
                continue
            opens[:, j] = sub["Open"].values if "Open" in sub else np.nan
            highs[:, j] = sub["High"].values if "High" in sub else np.nan
            lows[:, j] = sub["Low"].values if "Low" in sub else np.nan
            closes[:, j] = sub["Close"].values if "Close" in sub else np.nan
            volumes[:, j] = sub["Volume"].values if "Volume" in sub else np.nan
        except Exception:
            continue
    cols = {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    return dates, cols, tickers


def _filter_universe(dates, cols, tickers, min_coverage=0.7):
    closes = cols["close"]
    T = closes.shape[0]
    keep = []
    for j in range(closes.shape[1]):
        cov = np.isfinite(closes[:, j]).mean()
        if cov >= min_coverage and np.isfinite(closes[-1, j]):
            keep.append(j)
    if not keep:
        raise RuntimeError("No tickers passed coverage filter")
    idx = np.array(keep)
    filtered_tickers = [tickers[j] for j in idx]
    filtered_cols = {k: v[:, idx] for k, v in cols.items()}
    filtered_closes = filtered_cols["close"]
    valid_days = np.isfinite(filtered_closes).sum(1) >= max(5, len(filtered_tickers) // 10)
    dates_f = dates[valid_days]
    cols_f = {k: v[valid_days] for k, v in filtered_cols.items()}
    return dates_f, cols_f, filtered_tickers


def generate_real_panel(seed=7, n_stocks=200, start="2020-01-01", end="2022-12-31",
                        tickers=None, cache_path=None, use_cache=True):
    rng = np.random.default_rng(seed)

    if tickers is None:
        tickers = DEFAULT_TICKERS[: max(n_stocks, len(DEFAULT_TICKERS))]
        tickers = tickers[:n_stocks]
    else:
        tickers = list(tickers)[:n_stocks]

    if cache_path is None:
        import hashlib
        ticker_key = hashlib.sha256(",".join(sorted(tickers)).encode()).hexdigest()[:12]
        safe = f"{start}_{end}_{len(tickers)}_{ticker_key}"
        cache_path = os.path.join(os.path.dirname(__file__), "cache", f"yfinance_{safe}.pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if sorted(cached.get("tickers", [])) != sorted(tickers):
            raise ValueError("real-data cache ticker metadata does not match request")
        dates, cols, kept = cached["dates"], cached["cols"], cached["tickers"]
    else:
        dates, cols, kept = _download_ohlcv(tickers, start, end)
        dates, cols, kept = _filter_universe(dates, cols, kept)
        if use_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"dates": dates, "cols": cols, "tickers": kept}, f)

    T, N = cols["close"].shape
    close = cols["close"].astype(np.float64)
    open_ = cols["open"].astype(np.float64)
    high = cols["high"].astype(np.float64)
    low = cols["low"].astype(np.float64)
    volume = cols["volume"].astype(np.float64)

    for arr in (close, open_, high, low):
        mask = ~np.isfinite(arr)
        if mask.any():
            for j in range(N):
                col = arr[:, j]
                valid = np.isfinite(col)
                if valid.any():
                    last = np.nan
                    for t in range(T):
                        if valid[t]:
                            last = col[t]
                        elif np.isfinite(last):
                            col[t] = last
                    arr[:, j] = col

    returns = np.full_like(close, np.nan)
    returns[1:] = close[1:] / close[:-1] - 1.0
    returns[0] = 0.0
    returns = np.where(np.isfinite(returns), returns, 0.0)

    dollar_vol = close * volume
    adv20 = _roll_mean(np.where(np.isfinite(dollar_vol), dollar_vol, np.nan), 20)

    cap = close * 1e8

    sector = rng.integers(0, 10, N)
    industry = sector * 3 + rng.integers(0, 3, N)
    subindustry = industry * 2 + (np.arange(N) % 2)  # deterministic: no RNG (see data_gen note)

    liq = np.nanmean(dollar_vol, axis=0)
    liq_rank = _rank1d(np.where(np.isfinite(liq), liq, 0.0))

    # Sec 3.3: short-tenor IV is a sparser subset of longer-tenor coverage.
    p_iv = 0.10 + 0.85 * liq_rank[None, :] ** 2
    rv20 = _roll_std(returns, 20)
    med_rv = np.nanmedian(rv20, axis=0)
    rv20 = np.where(np.isfinite(rv20), rv20, med_rv)
    iv_base = rv20 * np.sqrt(252.0) * 100.0 * (1.05 + rng.normal(0.0, 0.08, (T, N))) + 8.0
    # GK anchor mirrors data_gen (range estimator on executed OHLC; rv20 stays
    # close-based for hv_20 definition stability). Real OHLC includes genuine
    # overnight gaps, so GK earns its keep here more than on synthetic bars.
    from iv_surface import garman_klass_estimate as _gk
    from data_gen import _roll_mean as _rm
    _gk_ann = np.sqrt(_rm(np.where(np.isfinite(close),
                                   _gk(open_, high, low, close) ** 2, np.nan), 20))
    _gk_ann = np.where(np.isfinite(_gk_ann), _gk_ann, np.nanmedian(_gk_ann, axis=0))
    iv_base = _gk_ann * np.sqrt(252.0) * 100.0 * (1.05 + rng.normal(0.0, 0.08, (T, N))) + 8.0
    # C6 SSVI mirror (see data_gen): arbitrage-free slices, not ad-hoc
    # multiples. Asymmetry (documented): no regime mult here — realized real
    # vols already embed crises (crisis=zeros); data_gen scales its synthetic
    # anchor instead. Masks/noise structure identical; asserts run every build.
    # Surface computed below once skew0 exists (side-effect-free import only).
    from iv_surface import (generate_iv_surface as _geniv,
                            assert_no_calendar_arb as _calarb,
                            assert_coverage_nesting as _nest)
    iv_cov = rng.random((T, N)) < p_iv
    iv_black = _blackout_masks(rng, T, 1, 0.008, 0.1, 5)
    iv_cov &= ~iv_black[:, None]
    iv_cov_60 = iv_cov
    short_keep = (0.35 + 0.60 * liq_rank[None, :])
    iv_cov_10 = iv_cov_60 & (rng.random((T, N)) < short_keep)

    def _tiny_scale(mask):
        hit = mask & (rng.random((T, N)) < 0.004) & (liq_rank[None, :] < 0.35)
        return np.where(hit, rng.uniform(0.02, 0.1, (T, N)), 1.0)

    skew0 = rng.normal(0.03, 0.05, N)
    _surf, _thetas = _geniv(iv_base, liq_rank, skew0, np.zeros(T))
    _calarb(_thetas)
    # Sticky-ratio puts mirror data_gen: static per-name wing markup from the
    # surface at mean theta (put/call persistence, not level-chasing).
    from iv_surface import term_mult as _tm, ssvi_total_var as _ssviw, ssvi_params as _ssvip
    _rho_s, _eta_s = _ssvip(liq_rank, skew0, np.zeros(N))
    _thm = 0.25 ** 2  # reference-vol evaluation mirrors data_gen (static ranks)
    _s, _sk = {}, {10.0: -0.15, 60.0: -0.25, 720.0: -0.35}
    for _Td, _Ty in ((10.0, 10.0 / 365.0), (60.0, 60.0 / 365.0), (720.0, 720.0 / 365.0)):
        _th = np.maximum(_thm * _tm(float(_Td)) ** 2, 1e-8)
        _wW = _ssviw(_sk[float(_Td)], _th, _rho_s, _eta_s)
        _wA = _ssviw(0.0, _th, _rho_s, _eta_s)
        _s[float(_Td)] = np.sqrt(_wW / np.maximum(_wA, 1e-12)) - 1.0
    call_10 = _surf[10.0][:, :, 2] * _tiny_scale(iv_cov_10) * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    call_60 = _surf[60.0][:, :, 2] * _tiny_scale(iv_cov_60) * (1.0 + rng.normal(0.0, 0.05, (T, N)))
    put_10 = _surf[10.0][:, :, 2] * (1.0 + _s[10.0])[None, :] * _tiny_scale(iv_cov_10) * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    put_60 = _surf[60.0][:, :, 2] * (1.0 + _s[60.0])[None, :] * _tiny_scale(iv_cov_60) * (1.0 + rng.normal(0.0, 0.05, (T, N)))
    call_10 = np.where(iv_cov_10, call_10, np.nan)
    call_60 = np.where(iv_cov_60, call_60, np.nan)
    put_10 = np.where(iv_cov_10, put_10, np.nan)
    put_60 = np.where(iv_cov_60, put_60, np.nan)
    # 720d tenor mirrors data_gen (SSVI ATM slice, term premium via term_mult).
    long_keep = 0.15 + 0.45 * liq_rank[None, :]
    iv_cov_720 = iv_cov_60 & (rng.random((T, N)) < long_keep)
    call_720 = np.where(iv_cov_720, _surf[720.0][:, :, 2] * (1.0 + rng.normal(0.0, 0.02, (T, N))), np.nan)
    put_720 = np.where(iv_cov_720, _surf[720.0][:, :, 2] * (1.0 + _s[720.0])[None, :] * (1.0 + rng.normal(0.0, 0.02, (T, N))), np.nan)
    _nest(iv_cov_60, iv_cov_10, iv_cov_720)

    margin = rng.beta(3.0, 12.0, N) * 0.35
    lev = np.clip(rng.normal(0.35, 0.15, N), 0.02, 0.9)
    nq = int(np.ceil(T / 63))
    # Parent unification mirror (see data_gen): generator runs FIRST, margin_q
    # from its E states; the beta-jitter draw is removed (downstream shift ok).
    from fundamentals import generate_fundamentals_q as _genfund, apply_equity_floor as _eqfloor
    from fundamentals import FUND_DEFAULTS as _FUND_DEFAULTS
    _size_proxy = np.nan_to_num(liq / np.nanmedian(liq) * 1e9, nan=1e8, posinf=1e9, neginf=1e8)
    _size_proxy = np.maximum(_size_proxy, 1e7)
    _rng_fund = np.random.default_rng([seed, 0xF17D])
    _lev_clip = np.clip(lev, 0.03, 0.89)
    _fparams = dict(_FUND_DEFAULTS,
                    mu_e=np.clip(margin, 0.02, 0.40),
                    mu_c=np.clip(margin - 0.035, -0.25, 0.40),
                    mu_l=np.log(_lev_clip / (1.0 - _lev_clip)))
    _fund = _genfund(_rng_fund, _size_proxy, nq, None, _fparams)
    margin_q = (_fund["ebitda_q"] / np.maximum(_fund["sales_q"], 1e-12))
    sales_q, assets_q, cfo_q = _fund["sales_q"], _fund["assets_q"], _fund["cfo_q"]
    # WC positivity mirrors data_gen (corr-leg review #40).
    liab_c = rng.uniform(0.08, 0.3, N)
    curr_a = liab_c + rng.uniform(0.05, 0.30, N)
    debt_q, _, _distress_q = _eqfloor(assets_q, assets_q * _fund["lev_q"], assets_q * liab_c[None, :])

    # Phase 4 PIT (mirrors data_gen): overlays are synthetic, so they get the
    # same knowledge-bounded serving (v2: size-graded lags, two-stage
    # restatements) rather than zero-lag forward-fill.
    from pit import build_revision_log_v2 as _brl, pit_asof_multi as _am, validate_pit as _vpit
    _pe = np.array([dates[min((q + 1) * 63 - 1, T - 1)] for q in range(nq)])
    _is_q4 = np.array([(q % 4 == 3) for q in range(nq)])
    _pit4 = _am({"sales": _brl(_pe, sales_q, "sales", rng, liq_rank=liq_rank, is_q4=_is_q4),
                 "margin": _brl(_pe, margin_q, "margin", rng, liq_rank=liq_rank, is_q4=_is_q4),
                 "assets": _brl(_pe, assets_q, "assets", rng, liq_rank=liq_rank, is_q4=_is_q4),
                 "debt": _brl(_pe, debt_q, "debt", rng, liq_rank=liq_rank, is_q4=_is_q4, distress=_distress_q),
                 "cashflow_op": _brl(_pe, cfo_q, "cashflow_op", rng, liq_rank=liq_rank, is_q4=_is_q4, distress=_distress_q)}, dates,
                 return_provenance=True)
    sales, _skn, _spe = _pit4["sales"]
    _marg_pit, _mkn, _mpe = _pit4["margin"]
    assets, _akn, _ape = _pit4["assets"]
    debt, _dkn, _dpe = _pit4["debt"]
    cashflow_op, _ckn, _cpe = _pit4["cashflow_op"]
    _vpit(sales, _skn, dates)
    _vpit(_marg_pit, _mkn, dates)
    _vpit(assets, _akn, dates)
    _vpit(debt, _dkn, dates)
    _vpit(cashflow_op, _ckn, dates)
    ebitda = sales * _marg_pit
    assets_curr = assets * curr_a[None, :]
    liabilities_curr = assets * liab_c[None, :]

    phi = 0.97
    sent_latent = np.zeros((T, N))
    s = rng.normal(0.0, 1.0, N)
    innov_scale = np.sqrt(1 - phi * phi)
    for t in range(T):
        s = phi * s + innov_scale * rng.normal(0.0, 1.0, N)
        sent_latent[t] = s
    sent_z = np.nan_to_num(_cs_z(sent_latent), nan=0.0)
    # Same sparsity recalibration as data_gen.py 2026-09-06 (Bug: sentiment
    # overlays printed ~47%; real nws12 prints ~1-5%).
    p_cov = 0.02 + 0.10 * liq_rank[None, :] ** 2
    sent_black = _blackout_masks(rng, T, 1, 0.03, 0.01, 30)
    in_black = sent_black[:, None]
    sent_cov = (rng.random((T, N)) < p_cov) & (~in_black | (rng.random((T, N)) < 0.01))
    jumps = np.where(rng.random((T, N)) < 0.002, rng.choice([-1.0, 1.0], (T, N)) * rng.uniform(3.0, 7.0, (T, N)), 0.0)
    sent_obs = np.where(sent_cov, sent_z * 0.9 + rng.normal(0.0, 0.5, (T, N)) + jumps, np.nan)
    vec_parts = [np.where(sent_cov, sent_obs + rng.normal(0.0, 0.35, (T, N)), np.nan) for _ in range(3)]
    buzz = np.where(sent_cov, 1.5 + np.abs(sent_z) * 1.5 + rng.normal(0.0, 0.7, (T, N)), np.nan)
    # Second sentiment process mirrors data_gen (independent AR(1), own mask).
    _phi2, _s2 = 0.90, rng.normal(0.0, 1.0, N)
    _lat2 = np.zeros((T, N))
    _is2 = np.sqrt(1 - _phi2 * _phi2)
    for _t in range(T):
        _s2 = _phi2 * _s2 + _is2 * rng.normal(0.0, 1.0, N)
        _lat2[_t] = _s2
    _sent2_z = np.nan_to_num(_cs_z(_lat2), nan=0.0)
    _p2 = 0.015 + 0.08 * liq_rank[None, :] ** 2
    _blk2 = _blackout_masks(rng, T, 1, 0.02, 0.01, 12)
    _cov2 = (rng.random((T, N)) < _p2) & (~_blk2[:, None] | (rng.random((T, N)) < 0.01))
    _obs2 = np.where(_cov2, _sent2_z * 0.9 + rng.normal(0.0, 0.5, (T, N)), np.nan)
    scl12_parts = [np.where(_cov2, _obs2 + rng.normal(0.0, 0.35, (T, N)), np.nan) for _ in range(3)]

    # Codebook extension mirrors data_gen (same documented proxies; overlays
    # stay synthetic APPROX). Real panel specifics: shares_out is constant
    # 1e8 (cap = close*1e8 by construction); size proxy = liq_rank scale.
    _fin = np.isfinite(close)
    _shares = np.full((T, N), 1e8)
    vwap = np.where(_fin, (high + low + close) / 3.0, np.nan)
    shares_out = np.where(_fin, _shares, np.nan)
    adv60 = _roll_mean(np.where(np.isfinite(dollar_vol), dollar_vol, np.nan), 60)
    _gm = rng.uniform(0.25, 0.6, N)[None, :]
    cogs = sales * (1.0 - _gm)
    gross_profit = sales - cogs
    operating_income = ebitda - 0.15 * ebitda
    _interest = debt * (0.04 / 252.0)
    _ebt = ebitda - 0.15 * ebitda - _interest
    tax_expense = 0.21 * np.maximum(_ebt, 0.0)
    net_income = _ebt - tax_expense
    eps = net_income / 1e8
    liabilities = debt + liabilities_curr
    # C2/C3 daily mirror of data_gen: exact identity, rescale debt on breach.
    _eq_pre = assets - liabilities
    _bad = _eq_pre < 0.05 * assets
    debt = np.where(_bad, assets * 0.95 - liabilities_curr, debt)
    liabilities = debt + liabilities_curr
    equity = assets - liabilities
    cash_and_equiv = assets_curr * 0.3
    retained_earnings = equity * 0.4
    # Goodwill burden mirror: dispersed share (static), timing via served assets.
    goodwill = assets * rng.uniform(0.02, 0.25, N)[None, :]
    working_capital = assets_curr - liabilities_curr
    # C5 single cash-flow truth (mirrors data_gen): PIT-served cashflow_op.
    operating_cash_flow = cashflow_op
    capex = sales * 0.05
    free_cash_flow = operating_cash_flow - capex
    dividends_paid = np.maximum(net_income, 0.0) * 0.3
    return_assets = net_income / np.maximum(assets, 1e-12)  # G1 mirror
    # Structural legs mirror data_gen (same distributions; stream not pinned).
    _sga_share = rng.uniform(0.10, 0.30, N)
    _ebit_wedge = rng.normal(0.0, 0.005, N)
    operating_expense = cogs + sales * _sga_share[None, :]
    ebit = operating_income + sales * _ebit_wedge[None, :]
    est_eps = eps * 4.0 * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    est_revenue = sales * 4.0 * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    est_eps_std = np.abs(est_eps) * 0.15
    recommendation = np.clip(3.0 + rng.normal(0.0, 0.8, (T, N)), 1.0, 5.0)
    eps_surprise = rng.normal(0.0, 0.05, (T, N))
    snt_news = np.where(sent_cov, np.clip(sent_z * 0.3 + rng.normal(0.0, 0.2, (T, N)), -1.0, 1.0), np.nan)
    snt_social = np.where(sent_cov, np.clip(sent_z * 0.2 + rng.normal(0.0, 0.3, (T, N)), -1.0, 1.0), np.nan)
    news_volume = np.where(sent_cov, np.maximum(np.round(np.where(np.isfinite(buzz), buzz, 0.0) * 2.0), 0.0), np.nan)
    iv_10 = np.where(iv_cov_10, (call_10 + put_10) / 2.0, np.nan)
    iv_30 = np.where(iv_cov_60, (call_60 + put_60) / 2.0, np.nan)
    hv_20 = rv20 * np.sqrt(252.0) * 100.0
    put_call_ratio = 0.7 + 2.0 * np.clip(skew0[None, :], -0.1, 0.2) + rng.normal(0.0, 0.1, (T, N))
    opt_open_interest = (liq_rank[None, :] * 1e6) * (1.0 + rng.normal(0.0, 0.2, (T, N)))
    _short_frac = np.clip(np.abs(rng.normal(0.02, 0.02, N))[None, :] + rng.normal(0.0, 0.002, (T, N)), 0.0, 0.2)
    short_interest = np.where(_fin, _shares * _short_frac, np.nan)
    days_to_cover = np.where(_fin, (short_interest * close) / np.maximum(adv20, 1e-12), np.nan)
    borrow_fee = np.clip(0.0025 + _short_frac * 0.5 + rng.normal(0.0, 0.002, (T, N)), 0.0025, None)
    # Anticipatory insider mirror (SSRN-278055): deterioration-loaded selling.
    _det = _marg_pit - np.roll(_marg_pit, 63, axis=0)
    _det[:63] = 0.0
    _detf = np.where(np.isfinite(_det), _det, np.nan)
    _has = np.isfinite(_detf).sum(axis=1) > 0
    _q25 = np.zeros(T)
    _q25[_has] = np.nanquantile(_detf[_has], 0.25, axis=1)
    _sell_p = 0.01 + 0.03 * (np.isfinite(_det) & (_det < _q25[:, None]))
    _ins = rng.random((T, N))
    _sz = np.nanmean(dollar_vol, axis=0, keepdims=True)
    insider_buying = np.where((_ins < 0.02) & _fin, np.abs(rng.normal(0.0, 1.0, (T, N))) * _sz * 1e-4, 0.0)
    insider_selling = np.where((_ins < _sell_p) & _fin, -np.abs(rng.normal(0.0, 1.0, (T, N))) * _sz * 1e-4, 0.0)
    _est_q = [est_eps * (1.0 + (q + 1) * 0.02 + rng.normal(0.0, 0.02, (T, N))) for q in range(4)]
    _skew_mid = np.where(np.isfinite(call_60) & np.isfinite(put_60), (call_60 + put_60) / 2.0, np.nan)
    _surf = [put_60 * 1.15, put_60 * 1.05, _skew_mid, call_60 * 1.05, call_60 * 1.15]
    _seg_w = rng.dirichlet([1.0, 1.0, 1.0, 1.0], N).T
    _seg_rev = [sales * _seg_w[q][None, :] for q in range(4)]
    _intra = [volume / 6.0 * (1.0 + rng.normal(0.0, 0.1, (T, N))) for _ in range(6)]
    fields = {
        "close": close, "open": open_, "high": high, "low": low,
        "volume": volume, "returns": returns, "adv20": adv20, "cap": cap,
        "ebitda": ebitda, "sales": sales, "debt": debt, "assets": assets,
        "cashflow_op": cashflow_op,
        "liabilities_curr": liabilities_curr, "assets_curr": assets_curr,
        "implied_volatility_call_10": call_10, "implied_volatility_call_60": call_60,
        "implied_volatility_put_10": put_10, "implied_volatility_put_60": put_60,
        "implied_volatility_call_720": call_720,
        "implied_volatility_put_720": put_720,
        "buzz": buzz,
        "vwap": vwap, "shares_out": shares_out, "adv60": adv60,
        "cogs": cogs, "gross_profit": gross_profit, "operating_income": operating_income,
        "net_income": net_income, "eps": eps, "tax_expense": tax_expense,
        "liabilities": liabilities, "equity": equity, "cash_and_equiv": cash_and_equiv,
        "retained_earnings": retained_earnings, "goodwill": goodwill,
        "working_capital": working_capital, "operating_cash_flow": operating_cash_flow,
        "capex": capex, "free_cash_flow": free_cash_flow, "dividends_paid": dividends_paid,
        "return_assets": return_assets, "operating_expense": operating_expense, "ebit": ebit,
        "est_eps": est_eps, "est_revenue": est_revenue, "est_eps_std": est_eps_std,
        "recommendation": recommendation, "eps_surprise": eps_surprise,
        "snt_news": snt_news, "snt_social": snt_social, "news_volume": news_volume,
        "iv_10": iv_10, "iv_30": iv_30, "hv_20": hv_20, "put_call_ratio": put_call_ratio,
        "opt_open_interest": opt_open_interest, "short_interest": short_interest,
        "days_to_cover": days_to_cover, "borrow_fee": borrow_fee,
        "insider_buying": insider_buying, "insider_selling": insider_selling,
        "revenue": sales, "op_income": operating_income, "ni": net_income,
        "total_debt": debt, "ocf": operating_cash_flow, "fcf": free_cash_flow,
        "pcr": put_call_ratio, "implied_volatility_10": iv_10,
        "implied_volatility_30": iv_30, "historical_volatility_20": hv_20,
    }
    vector_fields = {"nws12_afterhsz_01l": vec_parts,
                     "scl12_alltype_buzzvec": scl12_parts,
                     "scl12_buzzvec": scl12_parts,  # alias, mirrors data_gen
                     "analyst_eps_estimates": _est_q,
                     "option_implied_vol_surface": _surf,
                     "segment_revenue": _seg_rev,
                     "price_volume_intraday": _intra}
    groups = {"sector": sector, "industry": industry, "subindustry": subindustry,
              "market": np.zeros(N, dtype=int)}
    n_sub = max(1, int(N * 0.5))
    subuniverse = liq_rank >= np.sort(liq_rank)[-n_sub]

    panel = Panel(dates=dates, fields=fields, vector_fields=vector_fields, groups=groups, subuniverse=subuniverse,
                  knowledge_ts={'sales': _skn, 'assets': _akn,
                                'debt': _dkn, 'cashflow_op': _ckn},
                  period_end={'sales': _spe, 'assets': _ape,
                              'debt': _dpe, 'cashflow_op': _cpe},
                  report_lag_days={'sales': 0, 'margin': 0, 'assets': 0,
                                   'debt': 0, 'cashflow_op': 0})
    panel.tickers = kept
    validate_panel(panel)  # Phase 2: fail loudly on malformed panels
    panel.dataset_id = fingerprint_panel(dates, fields, vector_fields, groups=groups,
                                         subuniverse=subuniverse, backend="yfinance",
                                         tickers=kept, start=str(start), end=str(end), seed=seed)
    return panel
