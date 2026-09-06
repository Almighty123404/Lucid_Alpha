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
        safe = f"{start}_{end}_{len(tickers)}"
        cache_path = os.path.join(os.path.dirname(__file__), "cache", f"yfinance_{safe}.pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
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

    liq = np.nanmean(dollar_vol, axis=0)
    liq_rank = _rank1d(np.where(np.isfinite(liq), liq, 0.0))

    # Sec 3.3: short-tenor IV is a sparser subset of longer-tenor coverage.
    p_iv = 0.10 + 0.85 * liq_rank[None, :] ** 2
    rv20 = _roll_std(returns, 20)
    med_rv = np.nanmedian(rv20, axis=0)
    rv20 = np.where(np.isfinite(rv20), rv20, med_rv)
    iv_base = rv20 * np.sqrt(252.0) * 100.0 * (1.05 + rng.normal(0.0, 0.08, (T, N))) + 8.0
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
    call_10 = iv_base * _tiny_scale(iv_cov_10) * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    call_60 = iv_base * _tiny_scale(iv_cov_60) * (1.0 + rng.normal(0.0, 0.05, (T, N)))
    put_10 = iv_base * (1.0 + skew0[None, :]) * _tiny_scale(iv_cov_10) * (1.0 + rng.normal(0.0, 0.03, (T, N)))
    put_60 = iv_base * (1.0 + skew0[None, :]) * _tiny_scale(iv_cov_60) * (1.0 + rng.normal(0.0, 0.05, (T, N)))
    call_10 = np.where(iv_cov_10, call_10, np.nan)
    call_60 = np.where(iv_cov_60, call_60, np.nan)
    put_10 = np.where(iv_cov_10, put_10, np.nan)
    put_60 = np.where(iv_cov_60, put_60, np.nan)

    margin = rng.beta(3.0, 12.0, N) * 0.35
    lev = np.clip(rng.normal(0.35, 0.15, N), 0.02, 0.9)
    nq = int(np.ceil(T / 63))
    sales_q = (liq * 0.8 / np.nanmedian(liq) * 1e9)[None, :] if np.nanmedian(liq) > 0 else rng.uniform(1e8, 1e9, (1, N))
    if sales_q.shape[1] != N:
        sales_q = np.broadcast_to(sales_q, (1, N))
    sales_q = np.repeat(sales_q, nq, axis=0) * np.exp(rng.normal(0.0, 0.06, (nq, N)))
    margin_q = np.clip(margin[None, :] + rng.normal(0.0, 0.008, (nq, N)), 0.01, 0.45)
    turnover_a = rng.uniform(0.4, 1.8, N)
    assets_q = sales_q / turnover_a[None, :]
    debt_q = assets_q * lev[None, :]
    curr_a = rng.uniform(0.25, 0.5, N)
    liab_c = rng.uniform(0.08, 0.3, N)

    # Phase 4 PIT (mirrors data_gen): overlays are synthetic, so they get the
    # same knowledge-bounded serving (20-45d lag + 2% restatements) rather
    # than zero-lag forward-fill. ebitda uses PIT margin for consistency.
    from pit import build_revision_log as _brl, pit_asof_multi as _am, validate_pit as _vpit
    _pe = np.array([dates[min((q + 1) * 63 - 1, T - 1)] for q in range(nq)])
    _pit4 = _am({"sales": _brl(_pe, sales_q, "sales", rng),
                 "margin": _brl(_pe, margin_q, "margin", rng),
                 "assets": _brl(_pe, assets_q, "assets", rng),
                 "debt": _brl(_pe, debt_q, "debt", rng)}, dates)
    sales, _skn = _pit4["sales"]
    _marg_pit, _mkn = _pit4["margin"]
    assets, _akn = _pit4["assets"]
    debt, _dkn = _pit4["debt"]
    _vpit(sales, _skn, dates)
    _vpit(_marg_pit, _mkn, dates)
    _vpit(assets, _akn, dates)
    _vpit(debt, _dkn, dates)
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
    operating_income = ebitda * 0.85
    net_income = ebitda * 0.6
    eps = net_income / 1e8
    tax_expense = ebitda * 0.15
    liabilities = debt + liabilities_curr
    equity = assets - liabilities
    cash_and_equiv = assets_curr * 0.3
    retained_earnings = equity * 0.4
    goodwill = assets * 0.1
    working_capital = assets_curr - liabilities_curr
    operating_cash_flow = ebitda * 0.9
    capex = sales * 0.05
    free_cash_flow = operating_cash_flow - capex
    dividends_paid = np.maximum(net_income, 0.0) * 0.3
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
    _ins = rng.random((T, N)) < 0.02
    _sz = np.nanmean(dollar_vol, axis=0, keepdims=True)
    insider_buying = np.where(_ins & _fin, np.abs(rng.normal(0.0, 1.0, (T, N))) * _sz * 1e-4, 0.0)
    insider_selling = np.where(_ins & _fin, -np.abs(rng.normal(0.0, 1.0, (T, N))) * _sz * 1e-4, 0.0)
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
        "liabilities_curr": liabilities_curr, "assets_curr": assets_curr,
        "implied_volatility_call_10": call_10, "implied_volatility_call_60": call_60,
        "implied_volatility_put_10": put_10, "implied_volatility_put_60": put_60,
        "buzz": buzz,
        "vwap": vwap, "shares_out": shares_out, "adv60": adv60,
        "cogs": cogs, "gross_profit": gross_profit, "operating_income": operating_income,
        "net_income": net_income, "eps": eps, "tax_expense": tax_expense,
        "liabilities": liabilities, "equity": equity, "cash_and_equiv": cash_and_equiv,
        "retained_earnings": retained_earnings, "goodwill": goodwill,
        "working_capital": working_capital, "operating_cash_flow": operating_cash_flow,
        "capex": capex, "free_cash_flow": free_cash_flow, "dividends_paid": dividends_paid,
        "est_eps": est_eps, "est_revenue": est_revenue, "est_eps_std": est_eps_std,
        "recommendation": recommendation, "eps_surprise": eps_surprise,
        "snt_news": snt_news, "snt_social": snt_social, "news_volume": news_volume,
        "iv_10": iv_10, "iv_30": iv_30, "hv_20": hv_20, "put_call_ratio": put_call_ratio,
        "opt_open_interest": opt_open_interest, "short_interest": short_interest,
        "days_to_cover": days_to_cover, "borrow_fee": borrow_fee,
        "insider_buying": insider_buying, "insider_selling": insider_selling,
    }
    vector_fields = {"nws12_afterhsz_01l": vec_parts,
                     "analyst_eps_estimates": _est_q,
                     "option_implied_vol_surface": _surf,
                     "segment_revenue": _seg_rev,
                     "price_volume_intraday": _intra}
    groups = {"sector": sector, "industry": industry,
              "market": np.zeros(N, dtype=int)}
    n_sub = max(1, int(N * 0.5))
    subuniverse = liq_rank >= np.sort(liq_rank)[-n_sub]

    panel = Panel(dates=dates, fields=fields, vector_fields=vector_fields, groups=groups, subuniverse=subuniverse)
    panel.tickers = kept
    validate_panel(panel)  # Phase 2: fail loudly on malformed panels
    panel.dataset_id = fingerprint_panel(dates, fields, vector_fields, backend="yfinance",
                                         tickers=kept, start=str(start), end=str(end), seed=seed)
    return panel
