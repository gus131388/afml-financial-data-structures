import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
# Cambia esta ruta al archivo que quieras inspeccionar.
BASE_DIR = Path(__file__).parent.parent
ARCHIVO  = BASE_DIR / "data" / "etf_trick" / "ES_2024_etf_series.parquet"

N_HEAD = 10   # filas iniciales por contrato
N_TAIL = 10   # filas finales por contrato
# ──────────────────────────────────────────────────────────────────────────────

def separador(titulo: str = "", ancho: int = 80, char: str = "=") -> None:
    print("\n" + char * ancho)
    if titulo:
        print(titulo)
        print(char * ancho)


def mostrar_bloque(label: str, df_bloque: pd.DataFrame) -> None:
    print(f"\n  -- {label} --")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width",       None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:,.10f}".format)
    print(df_bloque.to_string())
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.max_colwidth")
    pd.reset_option("display.float_format")


# ── VERIFICAR EXISTENCIA ───────────────────────────────────────────────────────
if not ARCHIVO.exists():
    print(f"No se encontro el archivo: {ARCHIVO}")
    print(f"Archivos parquet disponibles en {ARCHIVO.parent}:")
    if ARCHIVO.parent.exists():
        found = sorted(ARCHIVO.parent.rglob("*.parquet"))
        if found:
            for f in found:
                print(f"  {f.relative_to(BASE_DIR)}")
        else:
            print("  (ninguno)")
    else:
        print(f"  La carpeta no existe: {ARCHIVO.parent}")
else:
    print(f"Archivo: {ARCHIVO}")

    # ── ESQUEMA SIN CARGAR TODO EN MEMORIA ────────────────────────────────────
    pf = pq.ParquetFile(ARCHIVO)
    schema = pf.schema_arrow

    separador("ESQUEMA DEL ARCHIVO (sin cargar datos)")
    print(f"  Filas totales : {pf.metadata.num_rows:,}")
    print(f"  Columnas      : {schema.names}")
    print(f"  Num columnas  : {len(schema.names)}")
    print()
    for i, field in enumerate(schema):
        print(f"  [{i:>2}]  {field.name:<30}  {field.type}")

    # ── CARGA COMPLETA ─────────────────────────────────────────────────────────
    separador("CARGANDO DATAFRAME COMPLETO")
    df = pd.read_parquet(ARCHIVO)
    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)

    print(f"  Shape  : {df.shape[0]:,} filas  x  {df.shape[1]} columnas")
    print(f"  Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    separador("TIPOS DE DATOS")
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        null_info = f"  ({n_null:,} nulos)" if n_null else ""
        print(f"  {col:<30}  {str(df[col].dtype):<15}{null_info}")

    # ── FILAS POR SIMBOLO ──────────────────────────────────────────────────────
    separador("PRIMERAS Y ULTIMAS FILAS POR CONTRATO")

    if "symbol" in df.columns:
        simbolos = df["symbol"].unique()
        print(f"  Simbolos encontrados: {list(simbolos)}\n")
        for sym in simbolos:
            grupo = df[df["symbol"] == sym].reset_index(drop=True)
            separador(f"CONTRATO: {sym}  ({len(grupo):,} filas)", char="-")
            print(f"  Columnas: {list(grupo.columns)}")
            mostrar_bloque(f"Primeras {N_HEAD} filas", grupo.head(N_HEAD))
            mostrar_bloque(f"Ultimas  {N_TAIL} filas", grupo.tail(N_TAIL))
    else:
        print("  Columna 'symbol' no encontrada — mostrando sobre el DataFrame completo.")
        print(f"  Columnas: {list(df.columns)}\n")
        mostrar_bloque(f"Primeras {N_HEAD} filas", df.head(N_HEAD))
        mostrar_bloque(f"Ultimas  {N_TAIL} filas", df.tail(N_TAIL))
