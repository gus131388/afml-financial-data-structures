"""
═══════════════════════════════════════════════════════════════════════════════
afml_core — nucleo de metodologia AFML (Lopez de Prado)
═══════════════════════════════════════════════════════════════════════════════

Este modulo contiene la DEFINICION de los objetos de AFML. No ejecuta ningun
experimento: los guiones (Exercise_*.py) lo importan y deciden que preguntas
hacerle.

REGLA DE ORO
------------
Las ecuaciones de AFML no se modifican. En particular, la condicion de cierre de
las imbalance bars es literalmente

        T* = arg min_T { |theta_T| >= E_0[T] * |2v+ - E_0[v_t]| }

implementada como

        |theta_T| >= E_0[T] * |E_0[b_t * v_t]|

sin suelos, techos, tamanos objetivo ni escalado por volatilidad. Cuando el
estimador degenera, este modulo lo DETECTA y lo REPORTA mediante codigos de
estado; nunca reajusta el umbral para "salvar" la serie. Una serie degenerada es
un resultado, no un error a tapar.

Referencias de seccion: AFML §2.3.2 (imbalance bars), §2.4.1 (ETF trick).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# ═══════════════════════════════════════════════════════════════════════════════
# Constantes de contrato
# ═══════════════════════════════════════════════════════════════════════════════
CONTRACT_MULTIPLIER_ES = 50.0        # E-mini S&P 500: 50 USD por punto de indice


class AFMLDataError(Exception):
    """Los datos violan una invariante que el resto del pipeline da por cierta."""


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — Carga, rolls y serie continua
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class RollReport:
    """Todo lo que hay que saber sobre como se empalmaron los contratos."""
    mode: str
    symbols: list[str]
    n_raw: dict[str, int]
    n_kept: dict[str, int]
    roll_ts: list[pd.Timestamp]
    overlap_seconds: list[float]
    gap_seconds: list[float]
    price_old: list[float]
    price_new: list[float]
    notes: list[str] = field(default_factory=list)


def _load_contract(path: Path, multiplier: float,
                   max_rows: int | None = None) -> pd.DataFrame:
    """Lee un parquet de trades de Databento y normaliza columnas."""
    if not Path(path).exists():
        raise AFMLDataError(f"No existe el fichero de datos: {path}")

    cols = ["ts_event", "action", "price", "size", "symbol"]
    df = pd.read_parquet(path, columns=cols, engine="pyarrow")

    # Databento marca las ejecuciones con action == 'T'. Si el fichero ya viene
    # filtrado, la mascara no elimina nada; si no, evita meter cancelaciones.
    if "action" in df.columns:
        df = df[df["action"] == "T"]
    df = df.drop(columns=["action"], errors="ignore")

    df = df[df["size"] > 0]
    df = df.sort_values("ts_event", kind="mergesort").reset_index(drop=True)
    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)

    df = df.rename(columns={"price": "price_raw", "size": "size_contracts"})
    df["size_contracts"] = df["size_contracts"].astype("int64")
    df["dollar_value_usd"] = (df["price_raw"].to_numpy(np.float64)
                              * df["size_contracts"].to_numpy(np.float64)
                              * float(multiplier))
    return df


def build_continuous_series(contracts: list[dict], multiplier: float,
                            max_rows_per_contract: int | None = None
                            ) -> tuple[pd.DataFrame, RollReport]:
    """
    Empalma varios contratos en una unica serie temporal sin solapamiento.

    Criterio de roll (modo INPUT): la frontera entre el contrato i y el i+1 es el
    PRIMER ts_event del contrato i+1. Del contrato i se retienen los ticks
    estrictamente anteriores a esa frontera. Es un criterio determinista y
    auditable; no se elige por volumen ni por open interest, porque esos datos no
    estan en los ficheros crudos.

    NOTA: este empalme NO ajusta el gap de precio entre contratos. El ajuste lo
    hace el ETF Trick (§2.4.1), que es donde corresponde.
    """
    frames, n_raw, n_kept = [], {}, {}
    for c in contracts:
        d = _load_contract(Path(c["file"]), multiplier, max_rows_per_contract)
        d["symbol"] = c["symbol"]
        n_raw[c["symbol"]] = len(d)
        frames.append(d)

    if len(frames) < 1:
        raise AFMLDataError("No se ha cargado ningun contrato")

    syms = [c["symbol"] for c in contracts]
    roll_ts, overlap, gap, p_old, p_new, notes = [], [], [], [], [], []

    for i in range(len(frames) - 1):
        cur, nxt = frames[i], frames[i + 1]
        boundary = nxt["ts_event"].iloc[0]
        roll_ts.append(boundary)

        # Solapamiento real: cuanto tiempo cotizan ambos a la vez en los ficheros.
        ov = (cur["ts_event"].iloc[-1] - boundary).total_seconds()
        overlap.append(max(0.0, ov))

        keep = cur["ts_event"] < boundary
        frames[i] = cur[keep].reset_index(drop=True)
        if len(frames[i]) == 0:
            raise AFMLDataError(
                f"El contrato {syms[i]} queda vacio tras recortar en {boundary}")

        # Hueco: distancia entre el ultimo tick retenido y el primero del nuevo.
        g = (boundary - frames[i]["ts_event"].iloc[-1]).total_seconds()
        gap.append(g)
        p_old.append(float(frames[i]["price_raw"].iloc[-1]))
        p_new.append(float(nxt["price_raw"].iloc[0]))

        if ov <= 0:
            notes.append(
                f"{syms[i]} -> {syms[i+1]}: no hay solape entre ficheros. El "
                f"contrato nuevo SI cotizaba antes de {boundary}, pero esos ticks "
                f"no estan en los datos crudos. El hueco de {g/3600:.2f} h es una "
                f"carencia de datos, no una parada de mercado.")

    for i, f in enumerate(frames):
        n_kept[syms[i]] = len(f)

    df = pd.concat(frames, ignore_index=True)
    if not df["ts_event"].is_monotonic_increasing:
        raise AFMLDataError("La serie combinada no es monotona en ts_event")

    rep = RollReport(
        mode="frontera = primer ts_event del contrato siguiente (INPUT, determinista)",
        symbols=syms, n_raw=n_raw, n_kept=n_kept, roll_ts=roll_ts,
        overlap_seconds=overlap, gap_seconds=gap,
        price_old=p_old, price_new=p_new, notes=notes)
    return df, rep


def audit_segments(df: pd.DataFrame, rep: RollReport) -> pd.DataFrame:
    """
    Test INDEPENDIENTE del empalme: se recalcula todo desde el resultado final.

    Deliberadamente no reutiliza los contadores de build_continuous_series. Si
    ambos coinciden, el arqueo es real; si se reciclaran, seria tautologico.
    """
    rows = []
    for sym, g in df.groupby("symbol", observed=True, sort=False):
        rows.append({
            "symbol": sym,
            "n_ticks": len(g),
            "ts_first": g["ts_event"].iloc[0],
            "ts_last": g["ts_event"].iloc[-1],
            "monotona": bool(g["ts_event"].is_monotonic_increasing),
            "price_min": float(g["price_raw"].min()),
            "price_max": float(g["price_raw"].max()),
            "n_kept_esperado": rep.n_kept.get(sym, -1),
            "cuadra": len(g) == rep.n_kept.get(sym, -1),
        })
    out = pd.DataFrame(rows)
    # Ningun segmento puede empezar antes de que acabe el anterior.
    out["sin_solape_con_anterior"] = [True] + [
        out["ts_first"].iloc[i] > out["ts_last"].iloc[i - 1]
        for i in range(1, len(out))]
    return out


def count_duplicate_timestamps(df: pd.DataFrame) -> int:
    """Ticks cuyo ts_event coincide con el del tick anterior."""
    ts = df["ts_event"].to_numpy()
    return int((ts[1:] == ts[:-1]).sum())


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — ETF Trick (AFML §2.4.1) y tick rule (AFML §2.3.2.1)
# ═══════════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _etf_trick_kernel(price, phi, is_roll, K0):
    n = price.shape[0]
    K = np.empty(n, np.float64)
    h = np.empty(n, np.float64)
    K[0] = K0
    h[0] = K0 / (price[0] * phi)
    for t in range(1, n):
        if is_roll[t]:
            # En la frontera de roll el gap de carry NO es P&L: es un cambio de
            # instrumento. Se transporta K y se recalcula la posicion h al precio
            # del contrato nuevo.
            K[t] = K[t - 1]
        else:
            K[t] = K[t - 1] + h[t - 1] * phi * (price[t] - price[t - 1])
        if K[t] <= 0.0:
            K[t] = 1e-12          # el validador lo detectara; no se enmascara
        h[t] = K[t] / (price[t] * phi)
    return K, h


def etf_trick(df: pd.DataFrame, multiplier: float, K0: float = 1.0
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Serie de valor K_t de un ETF sintetico sobre la serie continua de futuros.

    Con un unico instrumento y peso 1, la posicion es h_t = K_t / (p_t * phi), de
    modo que dentro de un mismo contrato K_t/K_{t-1} = p_t/p_{t-1}: K es el
    producto acumulado de los retornos simples. En cada roll se salta el gap de
    carry.

    INTERPRETACION: al excluir el gap de carry y no acreditar rendimiento del
    colateral, K es un indice de PRECIO / EXCESO DE RETORNO (equivalente a spot
    ex-dividendo), NO de retorno total.
    """
    price = df["price_raw"].to_numpy(np.float64)
    if (price <= 0).any():
        raise AFMLDataError("Hay precios <= 0 en la serie")

    sym = df["symbol"].to_numpy()
    is_roll = np.zeros(len(df), np.bool_)
    is_roll[1:] = sym[1:] != sym[:-1]

    K, h = _etf_trick_kernel(price, float(multiplier), is_roll, float(K0))
    return K, h, is_roll


@njit(cache=True)
def _tick_rule_kernel(x, b_init):
    n = x.shape[0]
    b = np.empty(n, np.int8)
    prev = b_init
    for t in range(n):
        if t == 0:
            b[0] = np.int8(prev)          # condicion de frontera b_0
            continue
        d = x[t] - x[t - 1]
        if d > 0.0:
            prev = 1.0
        elif d < 0.0:
            prev = -1.0
        # d == 0  ->  b_t = b_{t-1}  (se arrastra prev)
        b[t] = np.int8(prev)
    return b


def tick_rule_on_etf_k(K: np.ndarray, b_init: float = 1.0) -> np.ndarray:
    """
    Tick rule (AFML §2.3.2.1) aplicada a la serie del ETF Trick:

        b_t = b_{t-1}            si  dK_t == 0
        b_t = |dK_t| / dK_t      si  dK_t != 0

    b_0 = +1 por convencion documentada: no existe tick previo del que heredar
    el signo, y hay que elegir uno. La eleccion se declara aqui explicitamente en
    lugar de quedar implicita.
    """
    if b_init not in (1.0, -1.0):
        raise ValueError("b_init debe ser +1 o -1")
    return _tick_rule_kernel(np.asarray(K, np.float64), float(b_init))


def validate_etf_trick(df: pd.DataFrame, K: np.ndarray, b: np.ndarray,
                       is_roll: np.ndarray) -> pd.DataFrame:
    """
    Validacion INDEPENDIENTE: cada test se recalcula desde cero, no se reutiliza
    nada de etf_trick(). Un test que reutilice el calculo que pretende comprobar
    no es un test.
    """
    price = df["price_raw"].to_numpy(np.float64)
    tests = []

    tests.append(("K estrictamente positivo",
                  float(-min(0.0, K.min())), 0.0))

    # Dentro de contrato: K_t/K_{t-1} debe igualar p_t/p_{t-1}.
    intra = ~is_roll[1:]
    if intra.any():
        r_K = K[1:][intra] / K[:-1][intra]
        r_p = price[1:][intra] / price[:-1][intra]
        tests.append(("Intra-contrato: K_t/K_{t-1} == p_t/p_{t-1}",
                      float(np.abs(r_K - r_p).max()), 1e-9))

    # En el roll: K se transporta sin P&L.
    if is_roll[1:].any():
        m = is_roll[1:]
        tests.append(("En roll: K_t == K_{t-1} (gap de carry excluido)",
                      float(np.abs(K[1:][m] / K[:-1][m] - 1.0).max()), 1e-12))

    # Retorno total acumulado reconstruido tick a tick.
    rec = np.exp(np.log(K[1:][intra] / K[:-1][intra]).sum())
    tests.append(("Producto de retornos intra-contrato == K_fin/K_ini",
                  float(abs(rec - K[-1] / K[0])) / max(1e-12, abs(K[-1] / K[0])),
                  1e-8))

    tests.append(("b_t pertenece a {-1,+1}",
                  float(np.abs(np.abs(b.astype(np.float64)) - 1.0).max()), 0.0))

    out = pd.DataFrame(tests, columns=["test", "valor", "tolerancia"])
    out["ok"] = out["valor"] <= out["tolerancia"]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — Barras estandar (tick / volume / dollar), AFML §2.3.1
# ═══════════════════════════════════════════════════════════════════════════════
def calibrate_thresholds(df_train: pd.DataFrame, target_bars: int) -> dict:
    """
    Umbrales que producen `target_bars` barras en la muestra de ENTRENAMIENTO.

    Sin esto, comparar curtosis o autocorrelacion entre tipos de barra mide el
    numero de barras, no el metodo de muestreo. Los umbrales se congelan aqui y
    se aplican tal cual a la muestra de evaluacion (sin re-ajustar): re-ajustar
    seria look-ahead.
    """
    if target_bars < 1:
        raise ValueError("target_bars debe ser >= 1")
    n = len(df_train)
    vol = float(df_train["size_contracts"].sum())
    dol = float(df_train["dollar_value_usd"].sum())
    return {"tick": n / target_bars,
            "volume": vol / target_bars,
            "dollar": dol / target_bars}


@njit(cache=True)
def _standard_bars_kernel(K, ts_i8, price, vol, dol, b, kind, threshold):
    """kind: 0 = tick, 1 = volume, 2 = dollar."""
    n = K.shape[0]
    cap = n + 1
    o = np.empty(cap, np.float64); hi = np.empty(cap, np.float64)
    lo = np.empty(cap, np.float64); c = np.empty(cap, np.float64)
    vwK = np.empty(cap, np.float64); vwP = np.empty(cap, np.float64)
    vv = np.empty(cap, np.float64); dd = np.empty(cap, np.float64)
    nt = np.empty(cap, np.int64); imb = np.empty(cap, np.float64)
    t_o = np.empty(cap, np.int64); t_c = np.empty(cap, np.int64)

    nb = 0
    acc = 0.0
    start = 0
    sK = 0.0; sP = 0.0; sV = 0.0; sD = 0.0; sI = 0.0
    h_ = -1e300; l_ = 1e300

    for i in range(n):
        if i == start:
            h_ = -1e300; l_ = 1e300
            sK = 0.0; sP = 0.0; sV = 0.0; sD = 0.0; sI = 0.0
        if K[i] > h_:
            h_ = K[i]
        if K[i] < l_:
            l_ = K[i]
        w = vol[i]
        sK += K[i] * w
        sP += price[i] * w
        sV += w
        sD += dol[i]
        sI += b[i] * dol[i]

        if kind == 0:
            acc += 1.0
        elif kind == 1:
            acc += vol[i]
        else:
            acc += dol[i]

        if acc >= threshold:
            o[nb] = K[start]; hi[nb] = h_; lo[nb] = l_; c[nb] = K[i]
            vwK[nb] = sK / sV if sV > 0.0 else K[i]
            vwP[nb] = sP / sV if sV > 0.0 else price[i]
            vv[nb] = sV; dd[nb] = sD; nt[nb] = i - start + 1; imb[nb] = sI
            t_o[nb] = ts_i8[start]; t_c[nb] = ts_i8[i]
            nb += 1
            acc = 0.0
            start = i + 1

    return (o[:nb], hi[:nb], lo[:nb], c[:nb], vwK[:nb], vwP[:nb],
            vv[:nb], dd[:nb], nt[:nb], imb[:nb], t_o[:nb], t_c[:nb], start)


def build_standard_bars(df: pd.DataFrame, K: np.ndarray, b: np.ndarray,
                        kind: str, threshold: float) -> pd.DataFrame:
    """
    Barras de tick, volumen o dolar sobre la serie del ETF Trick.

    Los ticks sobrantes al final (que no completan una barra) NO se descartan en
    silencio: se cuentan en attrs['n_residual_ticks'] y en attrs['coverage'].
    """
    kmap = {"tick": 0, "volume": 1, "dollar": 2}
    if kind not in kmap:
        raise ValueError(f"kind desconocido: {kind}")
    if threshold <= 0:
        raise ValueError("threshold debe ser > 0")

    res = _standard_bars_kernel(
        np.asarray(K, np.float64),
        df["ts_event"].to_numpy("datetime64[ns]").astype(np.int64),
        df["price_raw"].to_numpy(np.float64),
        df["size_contracts"].to_numpy(np.float64),
        df["dollar_value_usd"].to_numpy(np.float64),
        np.asarray(b, np.float64),
        kmap[kind], float(threshold))
    o, hi, lo, c, vwK, vwP, vv, dd, nt, imb, t_o, t_c, start = res

    tz = df["ts_event"].dt.tz
    out = pd.DataFrame({
        "ts_close": pd.to_datetime(t_c).tz_localize("UTC").tz_convert(tz),
        "open_K": o, "high_K": hi, "low_K": lo, "close_K": c,
        "vwap_K": vwK, "vwap_price": vwP,
        "volume_contracts": vv, "dollar_value_usd": dd,
        "n_ticks": nt, "imbalance_usd": imb,
        "ts_open": pd.to_datetime(t_o).tz_localize("UTC").tz_convert(tz),
    }).set_index("ts_close")

    n_res = len(df) - int(start)
    out.attrs["coverage"] = float(start) / max(1, len(df))
    out.attrs["n_residual_ticks"] = int(n_res)
    out.attrs["threshold"] = float(threshold)
    out.attrs["kind"] = kind
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — Dollar Imbalance Bars (AFML §2.3.2.2)
# ═══════════════════════════════════════════════════════════════════════════════
DIB_OK = 0
DIB_COLLAPSE_ABSORBING = 1
DIB_COLLAPSE_ONE_TICK = 2
DIB_MAX_BARS = 3
DIB_EXPLOSIVE_E_T = 4
DIB_EXPLOSIVE_RESIDUAL = 5
DIB_EXPLOSIVE_BAR = 6

DIB_STATUS_TEXT = {
    DIB_OK: "OK",
    DIB_COLLAPSE_ABSORBING:
        "umbral por debajo de min|b_t*v_t|: TODO tick cierra barra por si solo "
        "(estado absorbente T=1 demostrado)",
    DIB_COLLAPSE_ONE_TICK:
        "racha de barras de 1 tick por encima del limite (colapso practico)",
    DIB_MAX_BARS:
        "se alcanzo el tope de barras antes de agotar la muestra",
    DIB_EXPLOSIVE_E_T:
        "umbral explosivo: E[T] diverge respecto a su inicializacion",
    DIB_EXPLOSIVE_RESIDUAL:
        "barra final inacabada cubre una fraccion anomala de la muestra",
    DIB_EXPLOSIVE_BAR:
        "umbral explosivo: una barra absorbe una fraccion enorme de la muestra",
}


@njit(cache=True)
def _seed_kernel(bv, a_bv, burn):
    E = 0.0
    for k in range(burn):
        E = a_bv * bv[k] + (1.0 - a_bv) * E
    return E / (1.0 - (1.0 - a_bv) ** burn)


def dib_signed_flow(df: pd.DataFrame, b: np.ndarray) -> np.ndarray:
    """b_t * v_t tick a tick, con v_t = valor en dolares del trade (DIB)."""
    return np.asarray(b, np.float64) * df["dollar_value_usd"].to_numpy(np.float64)


def seed_E_bv(bv: np.ndarray, a_bv: float, burn: int) -> float:
    """
    E_0[b*v] sembrado con el burn-in.

    La EWMA arranca en 0, lo que sesga a la baja las primeras observaciones. El
    divisor (1 - (1-a)^burn) corrige ese sesgo, de forma que el resultado es la
    media ponderada de los `burn` ticks y no una mezcla con el cero inicial.
    """
    return float(_seed_kernel(np.asarray(bv, np.float64), float(a_bv), int(burn)))


class DIBDegenerateError(Exception):
    """El proceso DIB de AFML degenero. La regla NO se ha modificado."""

    def __init__(self, msg: str, report: "DIBReport"):
        super().__init__(msg)
        self.report = report


@njit(cache=True)
def _dib_kernel(bv, a_T, a_bv, E_T_init, E_bv_init, burn, max_bars,
                one_tick_run_limit, min_abs_bv, bar_span_limit, E_T_explode_limit):
    """
    Kernel literal de AFML §2.3.2.2.

    Para cada barra:
        umbral = E_0[T] * |E_0[b_t v_t]|     <- CONGELADO al abrir la barra
        theta_T = suma acumulada de b_t v_t
        cierra en el primer T con |theta_T| >= umbral

    E_0[T]  : EWMA (span_T)  de los T de barras ANTERIORES completadas.
    E_0[bv] : EWMA (span_bv) de los b_t v_t de ticks ANTERIORES. Se actualiza
              tick a tick, pero solo se LEE al abrir barra, de modo que en el
              momento de leerse contiene exclusivamente ticks de barras previas.
              Esto es lo que garantiza que no haya look-ahead intra-barra.

    Los limites (bar_span_limit, E_T_explode_limit, ...) NO alteran el umbral:
    solo detienen el proceso y devuelven un codigo de estado.
    """
    n = bv.shape[0]
    starts = np.empty(max_bars, np.int64)
    ends = np.empty(max_bars, np.int64)
    theta_c = np.empty(max_bars, np.float64)
    thresh_c = np.empty(max_bars, np.float64)
    ET_prior = np.empty(max_bars, np.float64)
    Ebv_prior = np.empty(max_bars, np.float64)

    E_T = E_T_init
    E_bv = E_bv_init
    i = burn
    nb = 0
    status = DIB_OK
    one_tick_run = 0

    while i < n:
        if nb >= max_bars:
            status = DIB_MAX_BARS
            break

        bar_start = i
        E_T_open = E_T
        E_bv_open = E_bv
        thr = E_T_open * abs(E_bv_open)          # E_0[T] * |E_0[b v]|

        # Colapso absorbente: si el umbral no llega ni al menor |b_t v_t| de la
        # muestra, el primer tick de cualquier barra la cierra. T = 1 para
        # siempre y el proceso deja de muestrear informacion.
        if thr < min_abs_bv:
            status = DIB_COLLAPSE_ABSORBING
            break

        theta = 0.0
        closed = False
        while i < n:
            theta += bv[i]
            E_bv = a_bv * bv[i] + (1.0 - a_bv) * E_bv
            i += 1
            if abs(theta) >= thr:
                closed = True
                break

        if not closed:
            break                                 # muestra agotada: barra abierta

        T = i - bar_start
        starts[nb] = bar_start
        ends[nb] = i
        theta_c[nb] = theta
        thresh_c[nb] = thr
        ET_prior[nb] = E_T_open
        Ebv_prior[nb] = E_bv_open
        nb += 1

        E_T = a_T * T + (1.0 - a_T) * E_T         # EWMA sobre barras completadas

        if T == 1:
            one_tick_run += 1
            if one_tick_run >= one_tick_run_limit:
                status = DIB_COLLAPSE_ONE_TICK
                break
        else:
            one_tick_run = 0

        if T > bar_span_limit:
            status = DIB_EXPLOSIVE_BAR
            break
        if E_T > E_T_explode_limit:
            status = DIB_EXPLOSIVE_E_T
            break

    return (starts[:nb], ends[:nb], theta_c[:nb], thresh_c[:nb],
            ET_prior[:nb], Ebv_prior[:nb], status)


@dataclass
class DIBReport:
    n_obs: int
    burn: int
    n_bars: int
    n_residual: int
    coverage: float
    status: int
    no_lookahead: bool
    E_T_init: float
    E_bv_init: float
    threshold_init: float
    min_abs_bv: float
    n_ticks_desc: dict
    frac_one_tick: float
    residual_from: object
    residual_imbalance: float
    traj: pd.DataFrame

    def to_text(self) -> str:
        L = []
        L.append("─" * 78)
        L.append("DIAGNOSTICO DOLLAR IMBALANCE BARS (AFML §2.3.2)")
        L.append("─" * 78)
        L.append(f"  Observaciones totales      : {self.n_obs:,}")
        L.append(f"  Burn-in consumido          : {self.burn:,} ticks "
                 f"({self.burn/max(1,self.n_obs):.4%}) [excluidos del loop principal]")
        L.append(f"  Barras completadas         : {self.n_bars:,}")
        L.append(f"  Ticks residuales           : {self.n_residual:,}")
        L.append(f"  Cobertura por barras       : {self.coverage:.4%}")
        L.append(f"  Estado                     : {self.status} — "
                 f"{DIB_STATUS_TEXT[int(self.status)]}")
        L.append(f"  Sin look-ahead intra-barra : {self.no_lookahead}")
        L.append("")
        L.append(f"  E_0[T]      inicial        : {self.E_T_init:,.2f} ticks/barra")
        L.append(f"  E_0[b*v]    inicial        : {self.E_bv_init:+,.2f} USD/tick")
        L.append(f"  Umbral      inicial        : {self.threshold_init:.4e} USD")
        L.append(f"  min|b_t*v_t| en la muestra : {self.min_abs_bv:,.2f} USD")
        L.append("")
        L.append(f"  Distribucion de n_ticks    : {self.n_ticks_desc}")
        L.append(f"  Barras de 1 tick           : {self.frac_one_tick:.2%}")
        L.append(f"  Residuo desde              : {self.residual_from}")
        L.append(f"  Imbalance residual         : {self.residual_imbalance:+,.2f} USD")
        return "\n".join(L)


def build_dollar_imbalance_bars(df: pd.DataFrame, K: np.ndarray, b: np.ndarray,
                                E_T_init: float, span_T: int, span_bv: int,
                                burn: int, max_bars: int,
                                one_tick_run_limit: int,
                                raise_on_degenerate: bool = True
                                ) -> tuple[pd.DataFrame, DIBReport]:
    """
    Dollar Imbalance Bars segun AFML §2.3.2.2, sin modificar la regla.

    v_t = price_raw * size * phi  (valor en dolares del trade). NO se usa etf_K
    para v_t: etf_K solo alimenta la tick rule que produce b_t. Mezclar ambos
    daria un "dolar sintetico" que no corresponde a ningun importe negociado.

    El burn-in sirve UNICAMENTE para sembrar E_0[b*v] y queda excluido del bucle
    principal. AFML no especifica inicializacion; hay que elegir una y aqui se
    documenta explicitamente.
    """
    bv = dib_signed_flow(df, b)
    n = bv.shape[0]

    if burn >= n:
        raise AFMLDataError(f"burn ({burn:,}) >= observaciones ({n:,})")
    usable = n - burn

    a_T = 2.0 / (span_T + 1.0)
    a_bv = 2.0 / (span_bv + 1.0)
    E_bv0 = seed_E_bv(bv, a_bv, burn)
    min_abs = float(np.abs(bv).min())

    st, en, th, thr, ETp, Ebvp, status = _dib_kernel(
        bv, a_T, a_bv, float(E_T_init), float(E_bv0), int(burn), int(max_bars),
        int(one_tick_run_limit), min_abs, 0.20 * usable, 20.0 * float(E_T_init))

    nb = st.size
    nt = (en - st) if nb else np.array([], np.int64)
    # El residuo empieza donde acaba la ULTIMA barra completada, no donde empieza.
    # Cuando el kernel corta tras registrar una barra (estados 2,3,4,6), bar_start
    # aun apunta al inicio de esa barra ya cerrada; usarlo contaria sus ticks como
    # residuo. En la salida por muestra agotada ambos coinciden.
    resid_start = int(en[-1]) if nb else int(burn)
    n_res = n - resid_start
    cov = (float(en[-1]) - burn) / usable if nb else 0.0

    # El kernel no puede evaluar el residuo: solo se conoce al agotar la muestra.
    if status == DIB_OK and n_res > 0.10 * usable:
        status = DIB_EXPLOSIVE_RESIDUAL

    traj = pd.DataFrame({
        "bar": np.arange(nb, dtype=np.int64),
        "tick_start": st, "n_ticks": nt,
        "E_T_prior": ETp, "E_bv_prior": Ebvp,
        "threshold_usd": thr, "imbalance_usd": th,
    })

    desc = {}
    if nb:
        q = np.percentile(nt, [1, 25, 50, 75, 99])
        desc = {"mean": float(nt.mean()), "min": int(nt.min()), "max": int(nt.max()),
                "p1": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                "p75": float(q[3]), "p99": float(q[4])}

    rep = DIBReport(
        n_obs=n, burn=int(burn), n_bars=nb, n_residual=int(n_res), coverage=cov,
        status=int(status), no_lookahead=True,
        E_T_init=float(E_T_init), E_bv_init=float(E_bv0),
        threshold_init=float(E_T_init) * abs(E_bv0), min_abs_bv=min_abs,
        n_ticks_desc=desc,
        frac_one_tick=float((nt == 1).mean()) if nb else 0.0,
        residual_from=(df["ts_event"].iloc[resid_start]
                       if resid_start < n else None),
        residual_imbalance=float(bv[resid_start:].sum()),
        traj=traj)

    bars = _bars_from_index(df, K, b, st, en) if nb else _empty_bars(df)

    if status != DIB_OK and raise_on_degenerate:
        raise DIBDegenerateError(
            f"El proceso DIB de AFML degenero: {DIB_STATUS_TEXT[int(status)]}. "
            f"Barras={nb}, 1-tick={rep.frac_one_tick:.2%}, cobertura={cov:.4%}. "
            f"La regla de AFML NO se ha modificado; la serie queda marcada "
            f"INVALIDA y debe excluirse de toda comparacion estadistica "
            f"posterior.", rep)
    return bars, rep


def _empty_bars(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["open_K", "high_K", "low_K", "close_K", "vwap_K", "vwap_price",
            "volume_contracts", "dollar_value_usd", "n_ticks", "imbalance_usd",
            "ts_open"]
    idx = pd.DatetimeIndex([], tz=df["ts_event"].dt.tz, name="ts_close")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols}, index=idx)


def _bars_from_index(df: pd.DataFrame, K: np.ndarray, b: np.ndarray,
                     st: np.ndarray, en: np.ndarray) -> pd.DataFrame:
    """Agrega OHLC/VWAP dados los indices de inicio y fin de cada barra."""
    K = np.asarray(K, np.float64)
    price = df["price_raw"].to_numpy(np.float64)
    vol = df["size_contracts"].to_numpy(np.float64)
    dol = df["dollar_value_usd"].to_numpy(np.float64)
    bv = np.asarray(b, np.float64) * dol
    ts = df["ts_event"].to_numpy()

    cK = np.cumsum(np.concatenate(([0.0], K * vol)))
    cP = np.cumsum(np.concatenate(([0.0], price * vol)))
    cV = np.cumsum(np.concatenate(([0.0], vol)))
    cD = np.cumsum(np.concatenate(([0.0], dol)))
    cI = np.cumsum(np.concatenate(([0.0], bv)))

    sV = cV[en] - cV[st]
    sV_safe = np.where(sV > 0, sV, 1.0)

    return pd.DataFrame({
        "ts_close": ts[en - 1],
        "open_K": K[st],
        "high_K": np.array([K[s:e].max() for s, e in zip(st, en)]),
        "low_K": np.array([K[s:e].min() for s, e in zip(st, en)]),
        "close_K": K[en - 1],
        "vwap_K": (cK[en] - cK[st]) / sV_safe,
        "vwap_price": (cP[en] - cP[st]) / sV_safe,
        "volume_contracts": sV,
        "dollar_value_usd": cD[en] - cD[st],
        "n_ticks": (en - st).astype(np.int64),
        "imbalance_usd": cI[en] - cI[st],
        "ts_open": ts[st],
    }).set_index("ts_close")


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — Evaluacion estadistica de retornos
# ═══════════════════════════════════════════════════════════════════════════════
def log_returns(bars: pd.DataFrame, col: str = "close_K") -> pd.DataFrame:
    """Retornos logaritmicos barra a barra. Descarta no finitos y lo reporta."""
    x = bars[col].to_numpy(np.float64)
    if len(x) < 2:
        return pd.DataFrame({"r_t": pd.Series(dtype="float64")},
                            index=bars.index[:0])
    r = np.log(x[1:] / x[:-1])
    out = pd.DataFrame({"r_t": r}, index=bars.index[1:])
    n0 = len(out)
    out = out[np.isfinite(out["r_t"])]
    out.attrs["n_dropped_nonfinite"] = int(n0 - len(out))
    return out


def evaluate_returns(r: np.ndarray, label: str, min_n: int = 30) -> dict:
    """Bateria estandar: autocorrelacion lag-1, Ljung-Box, normalidad."""
    r = np.asarray(r, np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n < min_n:
        return {"label": label, "n": n, "insufficient": True,
                "acf1": np.nan, "acf1_se": np.nan, "acf1_t": np.nan,
                "acf1_ci95": (np.nan, np.nan), "acf1_significant": False,
                "ljung_box": pd.DataFrame(), "skew": np.nan,
                "excess_kurtosis": np.nan, "jb_stat": np.nan,
                "jb_log10p": np.nan, "jb_stat_per_obs": np.nan}

    x = r - r.mean()
    acf1 = float((x[1:] * x[:-1]).sum() / (x * x).sum())
    se = 1.0 / np.sqrt(n)

    # IC bootstrap por bloques: preserva la dependencia serial que el SE
    # asintotico ignora. Sin esto, el IC es demasiado estrecho.
    rng = np.random.default_rng(12345)
    L = max(10, int(n ** (1 / 3)))
    nblk = n // L
    boot = np.empty(400, np.float64)
    for k in range(400):
        idx = rng.integers(0, n - L, size=nblk)
        s = np.concatenate([r[i:i + L] for i in idx])
        s = s - s.mean()
        d = (s * s).sum()
        boot[k] = (s[1:] * s[:-1]).sum() / d if d > 0 else np.nan
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])

    lb = acorr_ljungbox(r, lags=[1, 5, 10, 20], return_df=True)

    sk = float(stats.skew(r))
    ku = float(stats.kurtosis(r))                     # ya es exceso
    jb = n / 6.0 * (sk ** 2 + 0.25 * ku ** 2)
    # log-survival: con miles de barras el p-value hace underflow a 0.0 y deja de
    # informar. En log10 sigue siendo comparable entre tipos de barra.
    jb_log10p = float(stats.chi2.logsf(jb, 2) / np.log(10.0))

    return {"label": label, "n": n, "insufficient": False,
            "acf1": acf1, "acf1_se": se, "acf1_t": acf1 / se,
            "acf1_ci95": (float(lo), float(hi)),
            "acf1_significant": bool(lo > 0 or hi < 0),
            "ljung_box": lb, "skew": sk, "excess_kurtosis": ku,
            "jb_stat": float(jb), "jb_log10p": jb_log10p,
            "jb_stat_per_obs": float(jb / n)}


def variance_of_variances(rets: pd.DataFrame, freq: str = "M") -> dict:
    """
    Estabilidad de la varianza entre periodos.

    Se devuelve la version CRUDA y la ADIMENSIONAL (VoV / media^2). La cruda
    depende del nivel de varianza, que a su vez depende del numero de barras: no
    es comparable entre tipos de barra. Prevalece la adimensional.
    """
    if rets.empty:
        return {"n_periods": 0, "mean_var": np.nan, "vov_raw": np.nan,
                "vov_dimensionless": np.nan, "cv_of_var": np.nan}
    g = rets["r_t"].groupby(rets.index.to_period(freq))
    v = g.var().dropna()
    v = v[v > 0]
    if len(v) < 2:
        return {"n_periods": int(len(v)), "mean_var": float(v.mean()) if len(v) else np.nan,
                "vov_raw": np.nan, "vov_dimensionless": np.nan, "cv_of_var": np.nan}
    m = float(v.mean())
    vv = float(v.var(ddof=1))
    return {"n_periods": int(len(v)), "mean_var": m, "vov_raw": vv,
            "vov_dimensionless": vv / (m ** 2), "cv_of_var": float(v.std(ddof=1) / m)}
