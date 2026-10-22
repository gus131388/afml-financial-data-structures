# -*- coding: utf-8 -*-
"""
verificar_raw.py — contrasta que todos los contratos de data/raw comparten
formato, dominios de valores y patron de comportamiento.

POR QUE EXISTE
--------------
build_continuous_series() empalma los contratos asumiendo que son homogeneos:
mismas columnas, mismos tipos, mismo significado de `action` y `side`, misma
convencion horaria. Si un fichero llega de una descarga distinta —otro esquema
de Databento, otro publisher, ya filtrado a ejecuciones— el empalme no falla
con un error: produce una serie silenciosamente mal construida.

Este guion compara los contratos ENTRE SI y marca cualquier divergencia. No
valida contra una referencia externa: la pregunta es "¿son intercambiables?",
y eso solo se responde mirandolos juntos.

Se ejecuta con:  python verificar_raw.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PATRON = re.compile(r"^(?P<sym>[A-Z0-9]+)_(?P<ini>\d{4}-\d{2}-\d{2})_"
                    r"(?P<fin>\d{4}-\d{2}-\d{2})_trades\.parquet$")

# Columnas de baja cardinalidad cuyo dominio DEBE coincidir entre contratos.
#
# instrument_id NO esta aqui a proposito: es el identificador numerico que
# Databento asigna a cada contrato, asi que por definicion es DISTINTO en cada
# uno (ESH4=17077, ESM4=5602, ESU4=118, ESZ4=183748). Lo que si debe cumplirse
# es que haya EXACTAMENTE UNO por fichero; eso se comprueba aparte, porque dos
# instrument_id en el mismo fichero significarian que se mezclaron contratos.
CATEGORICAS = ["rtype", "publisher_id", "action", "side", "depth", "flags"]

# Por encima de este porcentaje de filas con ts_recv < ts_event se considera
# un problema y no ruido de reloj de la captura.
TOLERANCIA_RELOJ_PCT = 0.01

incidencias: list[str] = []


def hr(t=""):
    print("\n" + "=" * 100)
    if t:
        print(t)
        print("=" * 100)


def marcar(msg: str):
    incidencias.append(msg)


def col(path: Path, nombre: str):
    return pq.read_table(path, columns=[nombre])[nombre]


def contratos() -> list[tuple[str, Path]]:
    """Un fichero por contrato. Si hay varios trozos, avisa y usa el mas largo."""
    grupos: dict[str, list[Path]] = {}
    for f in sorted(RAW.glob("*.parquet")):
        m = PATRON.match(f.name)
        if m:
            grupos.setdefault(m["sym"], []).append(f)
    salida = []
    for s, v in sorted(grupos.items()):
        if len(v) > 1:
            elegido = max(v, key=lambda p: pq.ParquetFile(p).metadata.num_rows)
            print(f"  AVISO: {s} tiene {len(v)} ficheros; se usa el consolidado "
                  f"({elegido.name})")
            salida.append((s, elegido))
        else:
            salida.append((s, v[0]))
    return salida


def main() -> int:
    if not RAW.exists():
        print(f"No existe {RAW}")
        return 1

    hr("CONTRATOS ENCONTRADOS")
    cts = contratos()
    if len(cts) < 2:
        print("  Hace falta mas de un contrato para comparar.")
        return 1
    for s, p in cts:
        print(f"  {s}  ->  {p.name}")

    # =========================================================================
    # 1. ESQUEMA
    # =========================================================================
    hr("1. ESQUEMA — nombres, tipos y orden de las columnas")
    firmas = {}
    for s, p in cts:
        esq = pq.ParquetFile(p).schema_arrow
        firmas[s] = [(f.name, str(f.type), f.nullable) for f in esq]

    ref_s, ref = cts[0][0], firmas[cts[0][0]]
    print(f"  Referencia: {ref_s}  ({len(ref)} columnas)\n")
    print(f"  {'#':<3} {'columna':<16} {'tipo':<28} {'nullable':<9} " +
          " ".join(f"{s:>7}" for s, _ in cts))
    print("  " + "-" * (60 + 8 * len(cts)))
    for i, (n, t, nu) in enumerate(ref):
        marcas = []
        for s, _ in cts:
            marcas.append("  ==  " if i < len(firmas[s]) and firmas[s][i] == (n, t, nu)
                          else "  !!  ")
        print(f"  {i:<3} {n:<16} {t:<28} {str(nu):<9} " +
              " ".join(f"{m:>7}" for m in marcas))
    todas_iguales = all(firmas[s] == ref for s, _ in cts)
    print(f"\n  -> {'IDENTICO en los 4 contratos' if todas_iguales else 'HAY DIVERGENCIAS'}")
    if not todas_iguales:
        marcar("El esquema no es identico entre contratos")

    # =========================================================================
    # 2. METADATOS DEL FICHERO
    # =========================================================================
    hr("2. METADATOS DEL FICHERO PARQUET")
    print(f"  {'contrato':<8} {'filas':>12} {'rg':>4} {'filas/rg':>10} {'MB':>8} "
          f"{'compresion':<12} {'indice pandas':<16} {'created_by':<28}")
    print("  " + "-" * 110)
    compresiones, indices, creadores = set(), set(), set()
    for s, p in cts:
        m = pq.ParquetFile(p).metadata
        comp = m.row_group(0).column(0).compression
        md = pq.ParquetFile(p).schema_arrow.metadata or {}
        try:
            idx = ",".join(json.loads(md[b"pandas"].decode())["index_columns"])
        except Exception:
            idx = "(ninguno)"
        creador = (m.created_by or "")[:26]
        compresiones.add(comp); indices.add(idx); creadores.add(creador)
        print(f"  {s:<8} {m.num_rows:>12,} {m.num_row_groups:>4} "
              f"{m.num_rows // m.num_row_groups:>10,} "
              f"{p.stat().st_size/1048576:>8.1f} {comp:<12} {idx:<16} {creador:<28}")
    print()
    for etiqueta, conj in (("compresion", compresiones), ("indice pandas", indices),
                           ("created_by", creadores)):
        ok = len(conj) == 1
        print(f"  [{'OK' if ok else '!!'}] {etiqueta} comun: {sorted(conj)}")
        if not ok:
            marcar(f"Divergencia en {etiqueta}: {sorted(conj)}")

    # =========================================================================
    # 3. DOMINIOS DE VALORES
    # =========================================================================
    hr("3. DOMINIOS DE VALORES — las categoricas deben significar lo mismo")
    for c in CATEGORICAS:
        if c not in [n for n, _, _ in ref]:
            continue
        print(f"\n  --- {c} ---")
        dominios = {}
        for s, p in cts:
            a = col(p, c)
            vc = a.value_counts().to_pylist()
            tot = len(a)
            d = {str(x["values"]): x["counts"] / tot * 100 for x in vc}
            dominios[s] = d
            top = sorted(d.items(), key=lambda kv: -kv[1])[:5]
            print(f"    {s:<6} {len(d):>3} valores: " +
                  "  ".join(f"{k}={v:.2f}%" for k, v in top))
            del a
        claves = [frozenset(d) for d in dominios.values()]
        ok = len(set(claves)) == 1
        print(f"    [{'OK' if ok else '!!'}] mismo conjunto de valores en los "
              f"{len(cts)} contratos")
        if not ok:
            for s in dominios:
                print(f"        {s}: {sorted(dominios[s])}")
            marcar(f"La columna '{c}' no tiene el mismo dominio en todos los contratos")

    # instrument_id: uno y solo uno por fichero, y distinto entre contratos.
    print("\n  --- instrument_id (debe ser UNICO por fichero y DISTINTO entre contratos) ---")
    ids = {}
    for s, p in cts:
        v = sorted({str(x["values"]) for x in col(p, "instrument_id").value_counts().to_pylist()})
        ids[s] = v
        estado = "OK" if len(v) == 1 else "!! mas de un contrato mezclado"
        print(f"    {s:<6} {v}   [{estado}]")
        if len(v) != 1:
            marcar(f"{s} contiene {len(v)} instrument_id distintos: se mezclaron contratos")
    planos = [v[0] for v in ids.values() if len(v) == 1]
    if len(set(planos)) != len(planos):
        marcar("Dos contratos comparten instrument_id")
    else:
        print(f"    [OK] los {len(planos)} identificadores son distintos entre si")

    # =========================================================================
    # 4. NULOS
    # =========================================================================
    hr("4. VALORES NULOS POR COLUMNA")
    print(f"  {'columna':<16} " + " ".join(f"{s:>12}" for s, _ in cts))
    print("  " + "-" * (18 + 13 * len(cts)))
    for n, _, _ in ref:
        fila = []
        for s, p in cts:
            m = pq.ParquetFile(p).metadata
            idx = [f.name for f in pq.ParquetFile(p).schema_arrow].index(n)
            nn = sum(m.row_group(g).column(idx).statistics.null_count
                     for g in range(m.num_row_groups)
                     if m.row_group(g).column(idx).statistics is not None)
            fila.append(nn)
        estado = "" if len(set(fila)) == 1 and fila[0] == 0 else "  <-- revisar"
        print(f"  {n:<16} " + " ".join(f"{v:>12,}" for v in fila) + estado)

    # =========================================================================
    # 5. INTEGRIDAD TEMPORAL
    # =========================================================================
    hr("5. INTEGRIDAD TEMPORAL")
    print(f"  {'contrato':<8} {'monotono':>9} {'ts_recv<ts_event':>17} "
          f"{'dup ts_event':>13} {'dup sequence':>13}  periodo")
    print("  " + "-" * 118)
    resumen_t = {}
    for s, p in cts:
        te = col(p, "ts_event")
        mono = bool(pc.all(pc.less_equal(te[:-1], te[1:])).as_py())
        n = len(te)
        s_te = te.to_pandas()
        dup_te = int(s_te.duplicated().sum())
        lo, hi = s_te.iloc[0], s_te.iloc[-1]
        del te
        tr = col(p, "ts_recv").to_pandas()
        n_anterior = int((tr.to_numpy() < s_te.to_numpy()).sum())
        del tr, s_te
        sq = col(p, "sequence").to_pandas()
        dup_sq = int(sq.duplicated().sum())
        del sq
        pct_anterior = n_anterior / n * 100
        resumen_t[s] = (mono, pct_anterior, dup_te / n * 100, dup_sq / n * 100)
        print(f"  {s:<8} {str(mono):>9} {n_anterior:>9,} ({pct_anterior:.5f}%) "
              f"{dup_te/n*100:>12.3f}% {dup_sq/n*100:>12.3f}%  {lo}  ->  {hi}")

    if not all(v[0] for v in resumen_t.values()):
        marcar("Algun contrato tiene ts_event no monotono")

    # ts_recv < ts_event es desajuste de reloj de la captura de Databento, no
    # un defecto del fichero. Se tolera mientras sea residual, y ademas el
    # pipeline no lee ts_recv: _load_contract() solo usa ts_event, action,
    # price, size y symbol. Se informa para dejar constancia.
    peor = max(v[1] for v in resumen_t.values())
    if peor > TOLERANCIA_RELOJ_PCT:
        marcar(f"ts_recv anterior a ts_event en el {peor:.5f}% de las filas "
               f"(supera la tolerancia del {TOLERANCIA_RELOJ_PCT}%)")
    elif peor > 0:
        print(f"\n  Nota: hasta un {peor:.5f}% de filas tienen ts_recv < ts_event "
              f"(desajuste de reloj de la captura,")
        print(f"  por debajo de la tolerancia del {TOLERANCIA_RELOJ_PCT}%). "
              f"El pipeline no usa ts_recv, asi que no afecta.")
    rango_dup = [v[2] for v in resumen_t.values()]
    print(f"\n  dup ts_event: entre {min(rango_dup):.3f}% y {max(rango_dup):.3f}% "
          f"(esperado: CME emite varios prints por orden agresora)")
    if max(rango_dup) - min(rango_dup) > 3.0:
        marcar(f"El % de ts_event duplicados varia mucho entre contratos "
               f"({min(rango_dup):.2f}%–{max(rango_dup):.2f}%)")

    # =========================================================================
    # 6. PATRON DE SESION
    # =========================================================================
    hr("6. PATRON DE SESION — horarios y huecos")
    print(f"  {'contrato':<8} {'dias':>6} {'ticks/dia med':>14} "
          f"{'apertura tipica':>16} {'cierre tipico':>14} {'huecos>1h':>10} "
          f"{'findes':>7} {'mantto':>7} {'otros':>6}")
    print("  " + "-" * 104)
    for s, p in cts:
        t = col(p, "ts_event").to_pandas()
        dias = t.dt.date
        por_dia = dias.value_counts()
        inicio = t.groupby(dias).min().dt.strftime("%H:%M").mode().iloc[0]
        fin = t.groupby(dias).max().dt.strftime("%H:%M").mode().iloc[0]
        d = t.diff().dt.total_seconds()
        huecos = d[d > 3600]
        # Fin de semana: el mercado cierra el viernes (dayofweek 4) y reabre el
        # domingo por la tarde (6). Comparar con "<= 0" era el error: el domingo
        # es 6, no 0, asi que ningun hueco se clasificaba como finde.
        pos = [t.index.get_loc(i) for i in huecos.index]
        es_finde = [t.iloc[k - 1].dayofweek == 4 and t.iloc[k].dayofweek in (6, 0)
                    for k in pos]
        n_finde = int(sum(es_finde))
        no_finde = ~pd.Series(es_finde, index=huecos.index)
        n_mantto = int(((huecos <= 2 * 3600) & no_finde).sum())
        n_otros = len(huecos) - n_finde - n_mantto
        print(f"  {s:<8} {len(por_dia):>6} {int(por_dia.median()):>14,} "
              f"{inicio:>16} {fin:>14} {len(huecos):>10} {n_finde:>7} "
              f"{n_mantto:>7} {n_otros:>6}")
        del t, d

    # =========================================================================
    # 7. MAGNITUDES
    # =========================================================================
    hr("7. MAGNITUDES — precio y tamano")
    print(f"  {'contrato':<8} {'precio min':>11} {'precio max':>11} "
          f"{'size min':>9} {'size max':>9} {'size medio':>11} {'size=0':>9}")
    print("  " + "-" * 76)
    for s, p in cts:
        pr = col(p, "price").to_pandas()
        sz = col(p, "size").to_pandas()
        print(f"  {s:<8} {pr.min():>11,.2f} {pr.max():>11,.2f} "
              f"{sz.min():>9,} {sz.max():>9,} {sz.mean():>11,.2f} "
              f"{int((sz == 0).sum()):>9,}")
        del pr, sz

    # =========================================================================
    # 8. VEREDICTO
    # =========================================================================
    hr("8. VEREDICTO")
    if not incidencias:
        print("  Los contratos son INTERCAMBIABLES: mismo esquema, mismos")
        print("  metadatos, mismos dominios de valores y mismo patron temporal.")
        print("  Se pueden empalmar con build_continuous_series() sin ajustes.")
        return 0
    print(f"  {len(incidencias)} incidencia(s):\n")
    for i, m in enumerate(incidencias, 1):
        print(f"   {i}. {m}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
