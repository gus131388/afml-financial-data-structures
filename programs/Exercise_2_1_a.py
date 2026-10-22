"""
═══════════════════════════════════════════════════════════════════════════════
AFML — Capitulo 2, Ejercicio 2.1(a)
ETF Trick + barras alternativas + Dollar Imbalance Bars (AFML §2.3.2, §2.4.1)
E-mini S&P 500 (ES), contratos ESH4 / ESM4 / ESU4 / ESZ4, ano 2024 completo
═══════════════════════════════════════════════════════════════════════════════

Uso:
    python Exercise_2_1_a.py                 # muestra completa
    python Exercise_2_1_a.py --max-ticks 3e6 # prueba rapida

ALCANCE DE ESTA VERSION
-----------------------
Se corrigen errores de implementacion, look-ahead, manejo de rolls, validaciones
tautologicas y comparaciones estadisticas confundidas. Las ECUACIONES de AFML
NO se modifican. En concreto, la regla de cierre de las Dollar Imbalance Bars
sigue siendo literalmente

        |θ_T| ≥ E_0[T] · |E_0[b_t·v_t]|

sin suelos, techos, tamanos objetivo ni escalado por volatilidad. Si el umbral
degenera, el script lo REPORTA y marca la serie como invalida.

Los analisis exploratorios (Hurst, variance ratio, senal/ruido, comparacion con
el agresor del exchange) viven en `diagnostics_2_1_a.py` y estan etiquetados
como DIAGNOSTICO, no como parte de la definicion de AFML (§16).
"""

from __future__ import annotations

import argparse

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

for _s in (sys.stdout, sys.stderr):        # consola Windows en cp1252
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afml_core as ac                     # noqa: E402

# §17: NO se silencian todas las advertencias. Solo esta, que es conocida y
# documentada: pandas avisa al convertir a Period en series tz-aware.
warnings.filterwarnings("ignore", message=".*Converting to PeriodArray.*")

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETROS
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"
RAW_PATH = DATA_ROOT / "raw"
BARS_PATH = DATA_ROOT / "bars"
RETURNS_PATH = DATA_ROOT / "returns"
IMBALANCE_PATH = DATA_ROOT / "imbalance_bars"
ETF_TRICK_PATH = DATA_ROOT / "etf_trick"
FIGS_PATH = DATA_ROOT / "figures"
REPORTS_PATH = DATA_ROOT / "reports"

PHI = ac.CONTRACT_MULTIPLIER_ES     # 50 USD / punto de indice (ES)

# Los cuatro vencimientos trimestrales de 2024: ano natural completo y tres
# rolls. ESZ4 se consolido desde seis descargas quincenales con
# consolidar_raw.py (los trozos de Databento se solapan 6.5 h en cada frontera)
# y verificar_raw.py confirma que es intercambiable con los otros tres: mismo
# esquema, mismos dominios de valores y mismo patron temporal.
CONTRACTS = [
    {"symbol": "ESH4", "file": RAW_PATH / "ESH4_2024-01-02_2024-03-15_trades.parquet"},
    {"symbol": "ESM4", "file": RAW_PATH / "ESM4_2024-03-15_2024-06-21_trades.parquet"},
    {"symbol": "ESU4", "file": RAW_PATH / "ESU4_2024-06-21_2024-09-20_trades.parquet"},
    {"symbol": "ESZ4", "file": RAW_PATH / "ESZ4_2024-09-20_2024-12-20_trades.parquet"},
]

# §14 — calibracion para IGUAL numero de barras
TRAIN_FRACTION = 0.30          # muestra de entrenamiento (umbrales) vs evaluacion
TARGET_BARS_EVAL = 5_000       # barras objetivo en la muestra de evaluacion

# AFML §2.3.2 — Dollar Imbalance Bars
DIB_SPAN_T = 200               # span EWMA de E[T]      (en barras completadas)
DIB_SPAN_BV = 17_000           # span EWMA de E[b·v]    (en ticks)
DIB_BURN = 50_000              # ticks de burn-in, EXCLUIDOS del loop principal
DIB_E_T_INIT = 50_000.0        # convencion de inicializacion de E_0[T]
DIB_ONE_TICK_RUN_LIMIT = 1_000
DIB_E_T_INIT_SWEEP = [1e3, 1e4, 5e4, 1e5, 2e5, 5e5]   # robustez de la conclusion

GAP_WARN_HOURS = 1.0


def hr(title: str = "", ch: str = "═") -> None:
    print("\n" + ch * 78)
    if title:
        print(title)
        print(ch * 78)


# ═══════════════════════════════════════════════════════════════════════════════
def main(max_ticks: int | None = None) -> int:
    for p in (BARS_PATH, RETURNS_PATH, IMBALANCE_PATH, ETF_TRICK_PATH,
              FIGS_PATH, REPORTS_PATH):
        p.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    hr("PASO 1 — Carga, rolls y construccion de la serie continua")
    # ══════════════════════════════════════════════════════════════════════════
    if max_ticks:
        print(f"  *** MODO PRUEBA: max {int(max_ticks):,} ticks por contrato ***")
    df_all, roll_rep = ac.build_continuous_series(
        CONTRACTS, multiplier=PHI,
        max_rows_per_contract=int(max_ticks) if max_ticks else None)
    df_all["symbol"] = df_all["symbol"].astype("category")

    print(f"\n  Ticks crudos por contrato : {roll_rep.n_raw}")
    print(f"  Ticks retenidos           : {roll_rep.n_kept}")
    print(f"  Total combinado           : {len(df_all):,}")
    print(f"  Periodo                   : {df_all['ts_event'].iloc[0]}  ->  "
          f"{df_all['ts_event'].iloc[-1]}")

    # ── §12: diagnostico honesto de rolls ────────────────────────────────────
    print(f"\n  Modo de determinacion del roll : {roll_rep.mode}")
    for i, ts in enumerate(roll_rep.roll_ts):
        ov, gap = roll_rep.overlap_seconds[i], roll_rep.gap_seconds[i]
        print(f"\n  Roll {roll_rep.symbols[i]} -> {roll_rep.symbols[i+1]}")
        print(f"    Limite de roll (INPUT)   : {ts}")
        print(f"    Solapamiento real        : {ov:,.0f} s"
              f"  {'(HAY solape)' if ov > 0 else '(NO hay solape: es un HUECO)'}")
        print(f"    Hueco entre contratos    : {gap:,.0f} s = {gap/3600:.2f} h")
        print(f"    Ultimo precio {roll_rep.symbols[i]}     : "
              f"{roll_rep.price_old[i]:.2f}")
        print(f"    Primer precio {roll_rep.symbols[i+1]}    : "
              f"{roll_rep.price_new[i]:.2f}")
        print(f"    Gap de carry             : "
              f"{roll_rep.price_new[i] - roll_rep.price_old[i]:+.2f} puntos")
    for n in roll_rep.notes:
        print(f"\n  [AVISO] {n}")

    # ── §13: tests independientes de segmentacion y arqueo ───────────────────
    seg = ac.audit_segments(df_all, roll_rep)
    print("\n  Auditoria de segmentos (test independiente, puede fallar):")
    print(seg.to_string())
    n_dup = ac.count_duplicate_timestamps(df_all)
    print(f"\n  ts_event duplicados: {n_dup:,} ({n_dup/len(df_all):.3%}) — "
          f"esperado: CME emite varios prints por orden agresora que barre niveles.")

    # ── Continuidad temporal, consciente de los limites de roll ──────────────
    validate_continuity(df_all, "ES_combinado", set(roll_rep.roll_ts))

    # ══════════════════════════════════════════════════════════════════════════
    hr("PASO 2 — ETF Trick (AFML §2.4.1) y tick rule (AFML §2.3.2)")
    # ══════════════════════════════════════════════════════════════════════════
    etf_K, etf_h, is_roll = ac.etf_trick(df_all, multiplier=PHI)
    b_t = ac.tick_rule_on_etf_k(etf_K, b_init=1.0)
    df_all["etf_K"] = etf_K
    df_all["etf_h"] = etf_h
    df_all["b_t"] = b_t
    df_all["roll_flag"] = is_roll.astype(np.int8)

    print(f"  K inicial     : {etf_K[0]:.10f}")
    print(f"  K final       : {etf_K[-1]:.10f}")
    print(f"  Retorno total : {(etf_K[-1]/etf_K[0]-1)*100:+.2f}%")
    print("\n  INTERPRETACION (§11): al excluir el gap de carry en el roll y no")
    print("  acreditar rendimiento del colateral, etf_K es un indice de PRECIO /")
    print("  EXCESO DE RETORNO (equivalente a spot ex-dividendo), NO de retorno total.")

    p_buy = float((b_t == 1).mean())
    print(f"\n  Tick rule sobre etf_K: P(b=+1) = {p_buy:.6f}  ->  E[b] = {2*p_buy-1:+.6f}")
    print("  b_0 = +1 por convencion documentada (no hay tick previo).")

    val = ac.validate_etf_trick(df_all, etf_K, b_t, is_roll)
    print("\n  Validacion INDEPENDIENTE del ETF Trick (§13):")
    for _, r in val.iterrows():
        print(f"    [{'OK ' if r['ok'] else 'FALLO'}] {r['test']:<58} "
              f"{r['valor']:.3e}  (tol {r['tolerancia']:.0e})")
    if not val["ok"].all():
        raise ac.AFMLDataError("Validacion del ETF Trick fallida — ver tabla arriba")

    etf_file = ETF_TRICK_PATH / "ES_2024_etf_series.parquet"
    cols = ["ts_event", "symbol", "price_raw", "size_contracts", "dollar_value_usd",
            "etf_K", "etf_h", "b_t", "roll_flag"]
    df_all[cols].to_parquet(etf_file, engine="pyarrow", compression="snappy", index=False)
    print(f"\n  Guardado: {etf_file}  ({len(df_all):,} ticks)")

    # etf_h (la posicion h_t del ETF sintetico) ya esta persistida y no vuelve a
    # leerse en ningun paso posterior. Mantenerla viva cuesta el doble de lo que
    # parece: una copia dentro de df_all y otra en el array suelto que devolvio
    # etf_trick(). Con el ano completo son ~1.5 GB que no hacen nada.
    df_all.drop(columns=["etf_h"], inplace=True)
    del etf_h, is_roll

    # ══════════════════════════════════════════════════════════════════════════
    hr("PASO 3 — Barras estandar con umbrales CALIBRADOS a igual recuento (§14)")
    # ══════════════════════════════════════════════════════════════════════════
    n = len(df_all)
    n_train = int(n * TRAIN_FRACTION)
    df_train = df_all.iloc[:n_train]
    # VISTA, no copia. reset_index(drop=True) obligaba a pandas a consolidar
    # todos los bloques y duplicaba en memoria el 70% de la serie: con el ano
    # completo son ~2.7 GB mas un pico de ~2 GB al fusionar las columnas
    # float64, y era justo ahi donde la ejecucion completa moria por falta de
    # memoria. Nadie necesita el reindexado: build_standard_bars() accede por
    # nombre de columna y weekly_stability() por posicion (.iloc) y por el
    # indice del resample, nunca por el valor del indice de df_eval.
    df_eval = df_all.iloc[n_train:]
    K_eval = etf_K[n_train:]
    b_eval = b_t[n_train:]

    target_train = max(1, int(round(TARGET_BARS_EVAL * n_train / (n - n_train))))
    thresholds = ac.calibrate_thresholds(df_train, target_bars=target_train)

    print(f"  Muestra de entrenamiento : ticks [0, {n_train:,})   "
          f"({TRAIN_FRACTION:.0%})   -> umbrales")
    print(f"  Muestra de evaluacion    : ticks [{n_train:,}, {n:,})  -> comparacion")
    print(f"  Barras objetivo en eval  : {TARGET_BARS_EVAL:,}")
    print("\n  Umbrales calibrados (congelados tras el entrenamiento):")
    print(f"    tick   : {thresholds['tick']:,.1f} ticks")
    print(f"    volume : {thresholds['volume']:,.1f} contratos")
    print(f"    dollar : ${thresholds['dollar']:,.0f}")

    bars = {}
    for bt in ("tick", "volume", "dollar"):
        bars[bt] = ac.build_standard_bars(df_eval, K_eval, b_eval, bt, thresholds[bt])
        a = bars[bt].attrs
        print(f"\n  {bt:<7}: {len(bars[bt]):>7,} barras  |  cobertura {a['coverage']:.4%}  "
              f"|  residuo {a['n_residual_ticks']:,} ticks (§9: reportado, no descartado)")
        out = BARS_PATH / f"ES_2024_etf_{bt}_bars.parquet"
        bars[bt].to_parquet(out, engine="pyarrow", compression="snappy")

    counts = {k: len(v) for k, v in bars.items()}
    spread = max(counts.values()) / max(1, min(counts.values()))
    print(f"\n  Recuentos: {counts}   ratio max/min = {spread:.3f}")
    print(f"  {'OK: comparacion valida' if spread < 1.25 else 'AVISO: recuentos aun dispares'}"
          f" — sin esto, curtosis/varianza/autocorrelacion miden el numero de barras.")

    # ══════════════════════════════════════════════════════════════════════════
    hr("PASO 4 — Evaluacion estadistica (§15)")
    # ══════════════════════════════════════════════════════════════════════════
    rets, stats = {}, {}
    for bt in ("tick", "volume", "dollar"):
        rets[bt] = ac.log_returns(bars[bt], "close_K")
        rets[bt].to_parquet(RETURNS_PATH / f"ES_2024_etf_{bt}_returns.parquet",
                            engine="pyarrow", compression="snappy")
        stats[bt] = ac.evaluate_returns(rets[bt]["r_t"].to_numpy(), bt)

    print(f"\n  {'tipo':<8}{'N':>8}{'acf1':>10}{'SE':>8}{'t':>7}"
          f"{'IC95 bootstrap':>24}{'signif.':>9}")
    print("  " + "-" * 66)
    for bt, s in stats.items():
        lo, hi = s["acf1_ci95"]
        print(f"  {bt:<8}{s['n']:>8,}{s['acf1']:>+10.4f}{s['acf1_se']:>8.4f}"
              f"{s['acf1_t']:>+7.2f}{f'[{lo:+.4f}, {hi:+.4f}]':>24}"
              f"{'SI' if s['acf1_significant'] else 'no':>9}")

    print("\n  Ljung-Box (p-value) por lag:")
    print(f"  {'tipo':<8}" + "".join(f"{'lag '+str(l):>12}" for l in (1, 5, 10, 20)))
    print("  " + "-" * 56)
    for bt, s in stats.items():
        lb = s["ljung_box"]
        row = "".join(f"{lb['lb_pvalue'].get(l, np.nan):>12.4g}" for l in (1, 5, 10, 20))
        print(f"  {bt:<8}{row}")

    print("\n  Normalidad (JB con log-survival: evita el underflow de p a 0.0):")
    print(f"  {'tipo':<8}{'N':>8}{'skew':>9}{'exc.kurt':>10}{'JB':>13}"
          f"{'log10(p)':>11}{'JB/N':>9}")
    print("  " + "-" * 68)
    for bt, s in stats.items():
        print(f"  {bt:<8}{s['n']:>8,}{s['skew']:>+9.3f}{s['excess_kurtosis']:>10.3f}"
              f"{s['jb_stat']:>13.1f}{s['jb_log10p']:>11.1f}{s['jb_stat_per_obs']:>9.3f}")

    print("\n  Varianza de varianzas mensuales (§15: cruda Y adimensional):")
    print(f"  {'tipo':<8}{'periodos':>10}{'media Var':>14}{'VoV cruda':>14}"
          f"{'VoV/Var^2':>12}{'CV(Var)':>10}")
    print("  " + "-" * 68)
    vov = {}
    for bt in ("tick", "volume", "dollar"):
        vov[bt] = ac.variance_of_variances(rets[bt])
        v = vov[bt]
        print(f"  {bt:<8}{v['n_periods']:>10}{v['mean_var']:>14.4e}"
              f"{v['vov_raw']:>14.4e}{v['vov_dimensionless']:>12.4f}{v['cv_of_var']:>10.4f}")

    print("\n  " + interpret_results(stats, vov, counts))

    # ══════════════════════════════════════════════════════════════════════════
    hr("PASO 5 — Estabilidad del recuento semanal de barras")
    # ══════════════════════════════════════════════════════════════════════════
    weekly_stability(bars, df_eval, roll_rep)

    # ══════════════════════════════════════════════════════════════════════════
    hr("PASO 6 — Dollar Imbalance Bars (AFML §2.3.2) — regla SIN modificar")
    # ══════════════════════════════════════════════════════════════════════════
    print("  Regla de cierre : |θ_T| >= E_0[T] · |E_0[b_t·v_t]|   (literal, AFML §2.3.2)")
    print("  θ_T             = Σ b_t·v_t ,  b_t = tick rule sobre etf_K,")
    print(f"                    v_t = price_raw · size · {PHI:.0f}  (NO usa etf_K, §10)")
    print(f"  span E[T]       = {DIB_SPAN_T} barras   |   span E[b·v] = {DIB_SPAN_BV:,} ticks")
    print(f"  burn-in         = {DIB_BURN:,} ticks (excluidos del loop principal, §3)")
    print(f"  E_0[T] inicial  = {DIB_E_T_INIT:,.0f} ticks/barra (convencion documentada)")

    dib_bars, dib_rep, dib_valid = None, None, False
    max_bars = max(1_000, len(bars["dollar"]) * 4)
    try:
        dib_bars, dib_rep = ac.build_dollar_imbalance_bars(
            df_all, etf_K, b_t, E_T_init=DIB_E_T_INIT,
            span_T=DIB_SPAN_T, span_bv=DIB_SPAN_BV, burn=DIB_BURN,
            max_bars=max_bars, one_tick_run_limit=DIB_ONE_TICK_RUN_LIMIT,
            raise_on_degenerate=True)
        dib_valid = True
        print("\n" + dib_rep.to_text())
    except ac.DIBDegenerateError as e:
        dib_rep = e.report
        print("\n" + dib_rep.to_text())
        print("\n  " + "!" * 74)
        print("  DEGENERACION DEL ESTIMADOR AFML — la regla NO se ha modificado.")
        print(f"  {e}")
        print("  " + "!" * 74)

    # §8/§9: la serie degenerada NO se guarda con el nombre de una serie valida,
    # y queda EXCLUIDA de toda comparacion estadistica posterior.
    if dib_rep is not None and len(dib_rep.traj):
        traj_file = REPORTS_PATH / "ES_2024_dib_trajectory.parquet"
        dib_rep.traj.to_parquet(traj_file, engine="pyarrow", compression="snappy")
        print(f"\n  Trayectoria (E[T], E[b·v], umbral, θ) guardada: {traj_file}")
        show_trajectory(dib_rep)

    stale = IMBALANCE_PATH / "ES_2024_etf_dollar_imbalance_bars.parquet"
    if dib_valid:
        dib_bars.to_parquet(stale, engine="pyarrow", compression="snappy")
        print(f"  Guardado: {stale}")
        dib_ret = ac.log_returns(dib_bars, "close_K")
        dib_ret.to_parquet(RETURNS_PATH / "ES_2024_etf_dollar_imbalance_returns.parquet",
                           engine="pyarrow", compression="snappy")
        s = ac.evaluate_returns(dib_ret["r_t"].to_numpy(), "dollar_imbalance")
        if s.get("insufficient"):
            print(f"\n  DIB: solo {s['n']} retornos utiles — muestra insuficiente para")
            print("  inferencia estadistica. Comparacion con dollar bars OMITIDA (§8).")
        else:
            lo, hi = s["acf1_ci95"]
            print(f"\n  DIB acf1 = {s['acf1']:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
                  f"N={s['n']:,}  significativa={s['acf1_significant']}")
            d = stats["dollar"]
            print(f"  Comparacion con dollar bars (acf1 = {d['acf1']:+.4f}, N={d['n']:,}):")
            print(f"    {'DIB' if abs(s['acf1']) < abs(d['acf1']) else 'Dollar bars'} "
                  f"presenta menor |acf1|. Nota: los recuentos difieren "
                  f"({s['n']:,} vs {d['n']:,}), la comparacion NO esta calibrada a igual N")
            print("    porque forzar el recuento de las DIB exigiria alterar su formula (§14).")
    else:
        if dib_bars is not None and len(dib_bars):
            bad = IMBALANCE_PATH / "ES_2024_etf_dollar_imbalance_bars_DEGENERATE.parquet"
            dib_bars.to_parquet(bad, engine="pyarrow", compression="snappy")
            print(f"  Serie degenerada guardada aparte: {bad}")
        # No debe quedar en disco ningun artefacto DIB con nombre de serie valida:
        # el proximo lector no sabria que procede de un proceso degenerado.
        for f in (stale, RETURNS_PATH / "ES_2024_etf_dollar_imbalance_returns.parquet"):
            if f.exists():
                f.unlink()
                print(f"  ELIMINADO {f.name}: procede de un proceso DIB degenerado")
                print("  y estaba presentado como una serie valida.")
        print("\n  Comparaciones estadisticas con DIB: OMITIDAS (serie invalida, §8).")

    # ══════════════════════════════════════════════════════════════════════════
    hr("PASO 7 — Robustez: barrido de E_0[T] (¿es un artefacto de la inicializacion?)")
    # ══════════════════════════════════════════════════════════════════════════
    sweep_E_T_init(df_all, b_t, max_bars)

    hr("FIN")
    print(f"  Barras   : {BARS_PATH}")
    print(f"  Retornos : {RETURNS_PATH}")
    print(f"  Figuras  : {FIGS_PATH}")
    print(f"  Reportes : {REPORTS_PATH}")
    return 0 if dib_valid else 2


# ═══════════════════════════════════════════════════════════════════════════════
# Auxiliares
# ═══════════════════════════════════════════════════════════════════════════════

def validate_continuity(df: pd.DataFrame, label: str, roll_ts: set,
                        gap_hours: float = GAP_WARN_HOURS) -> None:
    """
    Continuidad temporal.

    CORRECCION (§12): la version original clasificaba como "pausa de
    mantenimiento 1-2 h" cualquier hueco en ese rango. El hueco de 1 h que
    aparece en cada frontera de contrato (el fichero del contrato nuevo empieza
    1 h despues de que expire el viejo) quedaba asi enmascarado como evento
    esperado. Aqui los huecos en fronteras de roll se clasifican por separado.
    """
    hr(f"Continuidad temporal — {label}", "─")
    ts = df["ts_event"]
    d = ts.diff()
    idx = d[d > pd.Timedelta(hours=gap_hours)].index

    n_wk = n_mnt = 0
    rolls, unexpected = [], []
    for i in idx:
        t0, t1 = ts.iloc[i - 1], ts.iloc[i]
        h = d.iloc[i].total_seconds() / 3600
        if t1 in roll_ts:
            rolls.append((t0, t1, h))
        elif t0.dayofweek in (4, 5) and h > 30:
            n_wk += 1
        elif 1.0 <= h <= 2.5:
            n_mnt += 1
        else:
            unexpected.append((t0, t1, h))

    print(f"  Primer tick : {ts.iloc[0]}")
    print(f"  Ultimo tick : {ts.iloc[-1]}")
    print(f"  Ticks       : {len(ts):,}")
    print(f"  Huecos > {gap_hours:.0f} h : {len(idx)}")
    print(f"    Fines de semana (esperados)        : {n_wk}")
    print(f"    Pausas de mantenimiento CME (1-2 h): {n_mnt}")
    if rolls:
        print(f"    EN FRONTERA DE ROLL                : {len(rolls)}  <-- HUECO DE DATOS")
        for t0, t1, h in rolls:
            print(f"        {t0}  ->  {t1}  ({h:.2f} h)")
        print("      Estos huecos NO son pausas de mantenimiento: el contrato nuevo")
        print("      SI cotizaba en esa ventana, pero no esta en los ficheros crudos.")
    if unexpected:
        print(f"    *** INESPERADOS                    : {len(unexpected)} ***")
        for t0, t1, h in unexpected[:20]:
            print(f"        {t0}  ->  {t1}  ({h:.2f} h)")


def interpret_results(stats: dict, vov: dict, counts: dict) -> str:
    """
    §15: la interpretacion se GENERA a partir de los resultados. No hay narrativa
    fija que declare un ganador.
    """
    L = ["INTERPRETACION (generada de los resultados, no predefinida):"]
    sig = [b for b, s in stats.items() if s["acf1_significant"]]
    if not sig:
        L.append("    * Ningun tipo de barra muestra autocorrelacion lag-1 distinta de")
        L.append("      cero al 95 % (IC bootstrap). Las diferencias entre metodos NO")
        L.append("      son estadisticamente distinguibles del ruido muestral.")
    else:
        L.append(f"    * Autocorrelacion lag-1 significativa al 95 % en: {', '.join(sig)}.")
        best = min(sig, key=lambda b: abs(stats[b]["acf1"]))
        L.append(f"      Menor |acf1| entre los significativos: {best}.")

    lb_rej = [b for b, s in stats.items()
              if len(s["ljung_box"]) and (s["ljung_box"]["lb_pvalue"] < 0.05).any()]
    L.append(f"    * Ljung-Box rechaza independencia (algun lag, α=0.05) en: "
             f"{', '.join(lb_rej) if lb_rej else 'ninguno'}.")

    kurt = {b: s["excess_kurtosis"] for b, s in stats.items()}
    kb = min(kurt, key=lambda b: kurt[b])
    L.append(f"    * Menor exceso de curtosis: {kb} ({kurt[kb]:.2f}). Con recuentos")
    L.append(f"      igualados ({counts}) esta comparacion ya no esta confundida por N.")
    L.append("    * Jarque-Bera rechaza normalidad en los tres casos (esperado con")
    L.append("      miles de barras); se compara por magnitud, no por p-value.")

    vr = {b: v["vov_dimensionless"] for b, v in vov.items()}
    vb = min(vr, key=lambda b: vr[b])
    vraw = min(vov, key=lambda b: vov[b]["vov_raw"])
    L.append(f"    * VoV adimensional minima: {vb}. VoV cruda minima: {vraw}.")
    if vb != vraw:
        L.append("      DISCREPAN: la VoV cruda depende del nivel de varianza y por")
        L.append("      tanto del numero de barras. Prevalece la version adimensional.")
    return "\n  ".join(L)


def weekly_stability(bars: dict, df_eval: pd.DataFrame, roll_rep) -> None:
    """CV del recuento semanal, con tratamiento consistente de semanas parciales (§15)."""
    tick_w = df_eval.resample("W-FRI", on="ts_event").size()
    tick_w = tick_w[tick_w > 0]

    first_ts, last_ts = df_eval["ts_event"].iloc[0], df_eval["ts_event"].iloc[-1]
    drop_first = first_ts.dayofweek != 0 or first_ts.hour > 1
    # CORRECCION: no basta con que el ultimo tick caiga en viernes; ES expira a
    # las 13:30 UTC, asi que ese viernes es un dia PARCIAL y la semana tambien.
    drop_last = not (last_ts.dayofweek == 4 and last_ts.hour >= 20)

    lo = tick_w.index[1] if (drop_first and len(tick_w) > 2) else tick_w.index[0]
    hi = tick_w.index[-2] if (drop_last and len(tick_w) > 2) else tick_w.index[-1]
    print(f"  Semana inicial {'descartada (parcial)' if drop_first else 'incluida'}; "
          f"semana final {'descartada (parcial)' if drop_last else 'incluida'}.")
    print(f"  Ventana evaluada: {lo.date()} -> {hi.date()}")

    st, colors = {}, {"tick": "#378ADD", "volume": "#D85A30", "dollar": "#7F77DD"}
    for bt, b in bars.items():
        w = b.resample("W-FRI").size()
        w = w[(w.index >= lo) & (w.index <= hi)]
        w = w[w > 0]
        st[bt] = {"s": w, "mean": w.mean(), "std": w.std(),
                  "cv": float(w.std() / w.mean()) if w.mean() else np.nan}

    best = min(st, key=lambda k: st[k]["cv"])
    print(f"\n  {'tipo':<8}{'semanas':>9}{'media':>9}{'sigma':>9}{'CV':>9}")
    print("  " + "-" * 44)
    for bt, s in st.items():
        mark = "  <- CV minimo" if bt == best else ""
        print(f"  {bt:<8}{len(s['s']):>9}{s['mean']:>9.1f}{s['std']:>9.1f}"
              f"{s['cv']:>9.4f}{mark}")
    print("\n  Con umbrales calibrados a igual recuento, el CV mide estabilidad")
    print(f"  temporal del muestreo y no el numero de barras: '{best}' es el mas estable.")

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("E-mini S&P 500 — Recuento semanal de barras (umbrales calibrados"
                 " a igual recuento)\nMuestra de evaluacion", fontsize=12, fontweight="bold")
    for bt, s in st.items():
        ax.plot(s["s"].index, s["s"].values, lw=1.5, marker="o", ms=3,
                color=colors[bt], label=f"{bt} bars (CV={s['cv']:.4f})")
    for i, rd in enumerate(roll_rep.roll_ts):
        if lo <= rd <= hi:
            ax.axvline(rd, color="red", lw=1.0, ls="--", alpha=0.7,
                       label="Roll" if i == 0 else "")
    m = st[best]
    ax.axhspan(m["mean"] - m["std"], m["mean"] + m["std"], color=colors[best],
               alpha=0.07, label=f"{best} ±1σ")
    ax.set_ylabel("Barras por semana")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.tight_layout()
    f = FIGS_PATH / "ES_2024_weekly_bar_counts.png"
    plt.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura: {f}")


def show_trajectory(rep: ac.DIBReport, n_blocks: int = 12) -> None:
    """Trayectoria de E[T], E[b·v] y umbral (apartado E del encargo)."""
    t = rep.traj
    if t.empty:
        return
    print("\n  Trayectoria de los estimadores (bloques de barras):")
    print(f"  {'bloque':<16}{'E[T]':>12}{'|E[b·v]|':>14}{'umbral':>14}"
          f"{'n_ticks med':>13}{'|θ| medio':>14}")
    print("  " + "-" * 83)
    step = max(1, len(t) // n_blocks)
    for i in range(0, len(t), step):
        c = t.iloc[i:i + step]
        print(f"  {f'{i}-{min(i+step, len(t))}':<16}{c['E_T_prior'].mean():>12.2f}"
              f"{c['E_bv_prior'].abs().mean():>14.4e}{c['threshold_usd'].mean():>14.4e}"
              f"{c['n_ticks'].mean():>13.2f}{c['imbalance_usd'].abs().mean():>14.4e}")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].semilogy(t["bar"], t["threshold_usd"], lw=0.8, color="#D85A30")
    axes[0].set_ylabel("umbral (USD)")
    axes[0].set_title("Trayectoria del umbral AFML  E[T]·|E[b·v]|  (escala log)")
    axes[1].semilogy(t["bar"], t["E_T_prior"], lw=0.8, color="#378ADD")
    axes[1].set_ylabel("E[T] (ticks)")
    axes[2].semilogy(t["bar"], t["n_ticks"].clip(lower=1), lw=0.8, color="#7F77DD")
    axes[2].set_ylabel("n_ticks por barra")
    axes[2].set_xlabel("indice de barra")
    for a in axes:
        a.grid(True, alpha=0.3)
    plt.tight_layout()
    f = FIGS_PATH / "ES_2024_dib_threshold_trajectory.png"
    plt.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura: {f}")


def sweep_E_T_init(df_all: pd.DataFrame, b_t: np.ndarray, max_bars: int) -> None:
    """
    ¿El colapso depende de la inicializacion elegida? Se barre E_0[T] sin tocar
    la regla de AFML. Si TODOS los valores degeneran, el fallo no es un artefacto
    de la inicializacion sino del estimador (§F).
    """
    bv = ac.dib_signed_flow(df_all, b_t)
    a_T, a_bv = 2.0 / (DIB_SPAN_T + 1), 2.0 / (DIB_SPAN_BV + 1)
    E_bv0 = ac.seed_E_bv(bv, a_bv, DIB_BURN)
    min_abs = float(np.abs(bv).min())

    print(f"  E_0[b·v] del burn-in = {E_bv0:+,.2f} USD/tick   "
          f"min|b·v| = {min_abs:,.0f} USD")
    usable = len(bv) - DIB_BURN
    print(f"\n  {'E_0[T]':>12}{'umbral inicial':>18}{'barras':>10}{'1-tick':>9}"
          f"{'cobertura':>12}{'T mediano':>13}  estado")
    print("  " + "-" * 100)
    n_degenerate = 0
    for E_T0 in DIB_E_T_INIT_SWEEP:
        st, en, _, _, _, _, status = ac._dib_kernel(
            bv, a_T, a_bv, float(E_T0), float(E_bv0), int(DIB_BURN),
            int(max_bars), int(DIB_ONE_TICK_RUN_LIMIT), min_abs,
            0.20 * usable, 20.0 * E_T0)
        nb = st.size
        nt = (en - st) if nb else np.array([0])
        cov = (en[-1] - DIB_BURN) / usable if nb else 0.0
        # Mismo criterio de residuo que aplica el wrapper: el kernel no puede
        # evaluarlo porque solo se conoce al agotar la muestra.
        resid_start = int(en[-1]) if nb else DIB_BURN
        if status == ac.DIB_OK and (len(bv) - resid_start) > 0.10 * usable:
            status = ac.DIB_EXPLOSIVE_RESIDUAL
        n_degenerate += int(status != ac.DIB_OK)
        print(f"  {E_T0:>12,.0f}{E_T0*abs(E_bv0):>18.4e}{nb:>10,}"
              f"{(nt == 1).mean():>9.1%}{cov:>12.4%}{np.median(nt):>13,.0f}"
              f"  {ac.DIB_STATUS_TEXT[int(status)]}")

    n_tot = len(DIB_E_T_INIT_SWEEP)
    print()
    if n_degenerate == n_tot:
        print(f"  CONCLUSION: las {n_tot}/{n_tot} inicializaciones degeneran. El fallo NO es")
        print("  un artefacto de E_0[T] ni un error de implementacion: es una propiedad")
        print("  del estimador E[T]·|E[b·v]| sobre esta muestra. Notese ademas que el")
        print("  regimen salta de COLAPSO (T=1) a EXPLOSION sin ningun rango intermedio")
        print("  utilizable, que es la firma de un punto fijo REPULSOR.")
    else:
        print(f"  {n_degenerate}/{n_tot} inicializaciones degeneran. Revisar las que no,")
        print("  comprobando si el numero de barras es utilizable en la practica.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ticks", type=float, default=None,
                    help="trunca la serie para pruebas rapidas")
    args = ap.parse_args()
    sys.exit(main(int(args.max_ticks) if args.max_ticks else None))
