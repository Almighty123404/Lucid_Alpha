import os
import pickle

import numpy as np
import yfinance as yf

from data_gen import Panel, _roll_mean, _roll_std, _cs_z, _rank1d, _blackout_masks


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
    qidx = np.minimum(np.arange(T) // 63, nq - 1)

    def _expand(aq):
        return aq[qidx, :]

    sales = _expand(sales_q)
    ebitda = sales * _expand(margin_q)
    assets = _expand(assets_q)
    debt = _expand(debt_q)
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
    p_cov = 0.25 + 0.5 * liq_rank[None, :]
    sent_black = _blackout_masks(rng, T, 1, 0.015, 0.01, 18)
    in_black = sent_black[:, None]
    sent_cov = (rng.random((T, N)) < p_cov) & (~in_black | (rng.random((T, N)) < 0.03))
    jumps = np.where(rng.random((T, N)) < 0.002, rng.choice([-1.0, 1.0], (T, N)) * rng.uniform(3.0, 7.0, (T, N)), 0.0)
    sent_obs = np.where(sent_cov, sent_z * 0.9 + rng.normal(0.0, 0.5, (T, N)) + jumps, np.nan)
    vec_parts = [np.where(sent_cov, sent_obs + rng.normal(0.0, 0.35, (T, N)), np.nan) for _ in range(3)]
    buzz = np.where(sent_cov, 1.5 + np.abs(sent_z) * 1.5 + rng.normal(0.0, 0.7, (T, N)), np.nan)

    fields = {
        "close": close, "open": open_, "high": high, "low": low,
        "volume": volume, "returns": returns, "adv20": adv20, "cap": cap,
        "ebitda": ebitda, "sales": sales, "debt": debt, "assets": assets,
        "liabilities_curr": liabilities_curr, "assets_curr": assets_curr,
        "implied_volatility_call_10": call_10, "implied_volatility_call_60": call_60,
        "implied_volatility_put_10": put_10, "implied_volatility_put_60": put_60,
        "buzz": buzz,
    }
    vector_fields = {"nws12_afterhsz_01l": vec_parts}
    groups = {"sector": sector, "industry": industry}
    n_sub = max(1, int(N * 0.5))
    subuniverse = liq_rank >= np.sort(liq_rank)[-n_sub]

    panel = Panel(dates=dates, fields=fields, vector_fields=vector_fields, groups=groups, subuniverse=subuniverse)
    panel.tickers = kept
    return panel
