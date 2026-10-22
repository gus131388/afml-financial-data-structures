# -*- coding: utf-8 -*-
"""
consolidar_raw.py — une los trozos de un contrato en un unico parquet

Databento entrega los rangos largos partidos en varios ficheros, y los trozos
CONSECUTIVOS SE SOLAPAN: el ultimo dia de un trozo aparece entero tambien al
principio del siguiente. En ESZ4 eso son 1,320,959 filas repetidas (5.49%).

POR QUE NO SE PUEDE USAR drop_duplicates()
------------------------------------------
Dentro de un MISMO fichero ya hay filas identicas legitimas: CME emite varios
prints por una sola orden agresora que barre niveles, y dos prints pueden
coincidir en todas las columnas. En ESZ4_2024-09-20 hay 26,468 de esas.
Un drop_duplicates() global las borraria y destruiria volumen real.

QUE SE HACE EN SU LUGAR
-----------------------
Recorte por frontera temporal: de cada trozo se conservan las filas con
    ts_event < (primer ts_event del trozo siguiente)
y el ultimo trozo se conserva entero. Es exacto porque se verifica antes que
la region solapada es identica fila a fila en ambos ficheros.

MEMORIA
-------
Se procesa por lotes y se escribe en streaming: el pico no depende del tamano
total del contrato. Un contrato de 24M de filas se consolida sin superar unos
cientos de MB de RAM.

Uso:
    python consolidar_raw.py ESZ4                # consolida ESZ4
    python consolidar_raw.py ESZ4 --dry-run      # solo informa, no escribe
    python consolidar_raw.py --todos --dry-run   # revisa todos los contratos
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# ESZ4_2024-09-20_2024-10-01_trades.parquet
PATRON = re.compile(r"^(?P<sym>[A-Z0-9]+)_(?P<ini>\d{4}-\d{2}-\d{2})_"
                    r"(?P<fin>\d{4}-\d{2}-\d{2})_trades\.parquet$")

LOTE = 500_000          # filas por lote leido
FILAS_POR_GRUPO = 1_000_000   # row group del fichero final (como los ya existentes)


def hr(t=""):
    print("\n" + "=" * 78)
    if t:
        print(t)
        print("=" * 78)


def trozos_por_simbolo() -> dict[str, list[Path]]:
    """Agrupa los ficheros de data/raw por contrato, ordenados por fecha inicial."""
    grupos: dict[str, list[tuple[str, Path]]] = {}
    for f in RAW.glob("*.parquet"):
        m = PATRON.match(f.name)
        if m:
            grupos.setdefault(m["sym"], []).append((m["ini"], f))
    return {s: [p for _, p in sorted(v)] for s, v in sorted(grupos.items())}


def primer_ts(path: Path) -> pa.Scalar:
    """
    Primer ts_event del fichero, como ESCALAR DE PYARROW.

    CUIDADO: aqui no se puede usar .as_py(). ts_event es timestamp[ns] y el
    datetime de Python solo llega a microsegundos, asi que .as_py() TRUNCA los
    nanosegundos. Comparar contra el valor truncado desplaza la frontera unos
    nanosegundos y mete o pierde filas sueltas en cada empalme.
    """
    return pq.read_table(path, columns=["ts_event"])["ts_event"][0]


def verificar_solape(a: Path, b: Path, corte: pa.Scalar) -> tuple[bool, int]:
    """
    Comprueba que la cola de `a` a partir de `corte` es identica a la cabeza
    de `b`. Si no lo fuera, recortar perderia informacion y hay que parar.

    Todas las comparaciones se hacen con escalares de pyarrow, por el motivo
    explicado en primer_ts().
    """
    ta = pq.read_table(a)
    cola = ta.filter(pc.greater_equal(ta["ts_event"], corte))
    if cola.num_rows == 0:
        return True, 0
    tb = pq.read_table(b)
    fin_a = pc.max(ta["ts_event"])
    cabeza = tb.filter(pc.less_equal(tb["ts_event"], fin_a))
    if cola.num_rows != cabeza.num_rows:
        return False, cola.num_rows
    return cola.equals(cabeza), cola.num_rows


def consolidar(simbolo: str, trozos: list[Path], dry_run: bool,
               verificar: bool) -> int:
    hr(f"{simbolo} — {len(trozos)} trozos")

    if len(trozos) == 1:
        print("  Ya esta en un unico fichero. Nada que hacer.")
        return 0

    # --- Esquemas: deben coincidir exactamente -------------------------------
    esquemas = {pq.ParquetFile(p).schema_arrow.remove_metadata() for p in trozos}
    if len(esquemas) != 1:
        print("  ERROR: los trozos no comparten esquema. Abortado.")
        return 1
    esquema = pq.ParquetFile(trozos[0]).schema_arrow   # con metadatos pandas
    print("  Esquema identico en los 6 trozos (se conserva el de pandas: "
          f"indice = {_indice(esquema)})")

    # --- Fronteras -----------------------------------------------------------
    cortes = [primer_ts(p) for p in trozos[1:]] + [None]

    print()
    print(f"  {'trozo':<24} {'filas':>12} {'se conservan':>13} {'recortadas':>12}")
    print("  " + "-" * 64)

    plan, bruto, neto = [], 0, 0
    for p, corte in zip(trozos, cortes):
        t = pq.read_table(p, columns=["ts_event"])["ts_event"]
        n = len(t)
        k = n if corte is None else int(pc.sum(pc.less(t, corte)).as_py())
        plan.append((p, corte, n, k))
        bruto += n
        neto += k
        print(f"  {p.name[len(simbolo)+1:len(simbolo)+23]:<24} {n:>12,} {k:>13,} {n-k:>12,}")

    print("  " + "-" * 64)
    print(f"  {'TOTAL':<24} {bruto:>12,} {neto:>13,} {bruto-neto:>12,}")
    print(f"\n  La concatenacion ingenua meteria {bruto-neto:,} filas duplicadas "
          f"({(bruto-neto)/bruto*100:.2f}%).")

    # --- Verificacion de que el solape es duplicacion exacta -----------------
    if verificar:
        print("\n  Verificando que las regiones solapadas son identicas...")
        for (a, corte, _, _), b in zip(plan[:-1], trozos[1:]):
            if corte is None:
                continue
            ok, n = verificar_solape(a, b, corte)
            estado = "identica" if ok else "DIFIERE  <-- NO RECORTAR"
            print(f"    {a.name[:28]:<30} {n:>9,} filas solapadas: {estado}")
            if not ok:
                print("\n  ABORTADO: el solape no es duplicacion exacta.")
                return 1

    if dry_run:
        print("\n  --dry-run: no se ha escrito nada.")
        return 0

    # --- Escritura en streaming ---------------------------------------------
    ini = min(PATRON.match(p.name)["ini"] for p in trozos)
    fin = max(PATRON.match(p.name)["fin"] for p in trozos)
    salida = RAW / f"{simbolo}_{ini}_{fin}_trades.parquet"
    tmp = salida.with_suffix(".parquet.tmp")

    print(f"\n  Escribiendo {salida.name} ...")
    escritas = 0
    buffer: list[pa.Table] = []
    buf_filas = 0
    writer = pq.ParquetWriter(tmp, esquema, compression="snappy")
    try:
        for p, corte, _, _ in plan:
            pf = pq.ParquetFile(p)
            for lote in pf.iter_batches(batch_size=LOTE):
                t = pa.Table.from_batches([lote], schema=pf.schema_arrow)
                if corte is not None:
                    t = t.filter(pc.less(t["ts_event"], corte))
                if t.num_rows == 0:
                    continue
                buffer.append(t)
                buf_filas += t.num_rows
                if buf_filas >= FILAS_POR_GRUPO:
                    writer.write_table(pa.concat_tables(buffer))
                    escritas += buf_filas
                    buffer, buf_filas = [], 0
                    print(f"    {escritas:>12,} / {neto:,} filas", end="\r")
        if buffer:
            writer.write_table(pa.concat_tables(buffer))
            escritas += buf_filas
    finally:
        writer.close()

    print(f"    {escritas:>12,} / {neto:,} filas")

    # --- Comprobaciones sobre el fichero escrito -----------------------------
    # El handle hay que CERRARLO antes de renombrar: en Windows un fichero con
    # un handle abierto no se puede mover, y os.replace falla con WinError 32.
    pf = pq.ParquetFile(tmp)
    try:
        filas_fichero = pf.metadata.num_rows
        grupos_fichero = pf.metadata.num_row_groups
        esquema_salida = pf.schema_arrow
    finally:
        pf.close()

    ts = pq.read_table(tmp, columns=["ts_event"])["ts_event"]
    monotona = bool(pc.all(pc.less_equal(ts[:-1], ts[1:])).as_py())
    del ts

    print()
    ok = True
    def check(c, msg):
        nonlocal ok
        print(f"  [{'OK' if c else 'ERROR'}] {msg}")
        ok &= bool(c)

    check(escritas == neto, f"filas escritas = previstas ({escritas:,})")
    check(filas_fichero == neto, f"filas en el fichero = {filas_fichero:,}")
    check(monotona, "ts_event no decrece en ningun punto")
    check(esquema_salida.equals(esquema), "esquema identico al de los trozos")

    if not ok:
        tmp.unlink(missing_ok=True)
        print("\n  ABORTADO: el fichero temporal se ha borrado.")
        return 1

    tmp.replace(salida)
    mb = salida.stat().st_size / 1048576
    print(f"\n  Guardado: {salida.name}  ({mb:,.1f} MB, "
          f"{grupos_fichero} row groups)")
    print("\n  Los trozos originales NO se han borrado. Comprueba el resultado")
    print("  y eliminalos tu si quieres:")
    for p, _, _, _ in plan:
        print(f"    del \"{p.name}\"")
    return 0


def _indice(esquema) -> str:
    import json
    md = esquema.metadata or {}
    try:
        return ", ".join(json.loads(md[b"pandas"].decode())["index_columns"])
    except Exception:
        return "(sin metadatos pandas)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("simbolo", nargs="?", help="contrato a consolidar, p.ej. ESZ4")
    ap.add_argument("--todos", action="store_true", help="procesa todos los contratos")
    ap.add_argument("--dry-run", action="store_true", help="informa sin escribir")
    ap.add_argument("--sin-verificar", action="store_true",
                    help="omite la comprobacion de que el solape es identico (mas rapido)")
    a = ap.parse_args()

    if not RAW.exists():
        print(f"No existe {RAW}")
        return 1

    grupos = trozos_por_simbolo()
    if not grupos:
        print(f"No hay parquet con el patron esperado en {RAW}")
        return 1

    if a.todos:
        objetivos = list(grupos)
    elif a.simbolo:
        if a.simbolo not in grupos:
            print(f"No hay ficheros de {a.simbolo}. Disponibles: {', '.join(grupos)}")
            return 1
        objetivos = [a.simbolo]
    else:
        hr("CONTRATOS EN data/raw")
        for s, v in grupos.items():
            estado = "1 fichero" if len(v) == 1 else f"{len(v)} trozos  <- consolidable"
            print(f"  {s}: {estado}")
        print("\nUsa:  python consolidar_raw.py <SIMBOLO>  [--dry-run]")
        return 0

    rc = 0
    for s in objetivos:
        rc |= consolidar(s, grupos[s], a.dry_run, not a.sin_verificar)
    return rc


if __name__ == "__main__":
    sys.exit(main())
