# AFML — Capítulo 2: Estructuras de datos financieros

> **Estado: en desarrollo.** Parte de lo implementado está validado y parte
> sigue abierta. Ver [Estado por componente](#estado-por-componente).

## Qué es AFML

**AFML** = *Advances in Financial Machine Learning*, de **Marcos López de
Prado** (Wiley, 2018). Es el libro de referencia sobre aplicación de machine
learning a datos financieros, conocido por insistir en que los métodos
estándar de ML fallan en finanzas si no se adaptan a la estructura particular
de los datos de mercado.

Su **Capítulo 2 — Financial Data Structures** es el que trabaja este
repositorio. La idea central: muestrear el mercado por **reloj de calendario**
(una barra cada 5 minutos, cada hora, cada día) es arbitrario, porque la
actividad no se reparte de forma uniforme en el tiempo. El capítulo propone
muestrear por **actividad**:

| Tipo de barra | Cierra cada vez que acumula |
|---|---|
| **Tick bars** | N operaciones |
| **Volume bars** | N contratos negociados |
| **Dollar bars** | N dólares intercambiados |
| **Imbalance bars** | un desequilibrio comprador/vendedor mayor del esperado |

Secciones implementadas: **§2.3.2** (tick rule e imbalance bars) y **§2.4.1**
(ETF trick, para empalmar contratos de futuros que vencen en una serie
continua).

## Datos

E-mini S&P 500 (ES) en CME Globex, año 2024 completo: los cuatro vencimientos
trimestrales (ESH4, ESM4, ESU4, ESZ4), **92.7 millones de ejecuciones**
tick-by-tick. Origen: [Databento](https://databento.com), dataset `GLBX.MDP3`,
esquema `trades`.

Los datos **no se versionan** (unos 4 GB y licencia de proveedor). Hay que
colocarlos en `data/raw/` con este patrón de nombre:

```
data/raw/ESZ4_2024-09-20_2024-12-20_trades.parquet
```

## Estado por componente

| Componente | Estado | Nota |
|---|---|---|
| Carga y empalme de contratos (rolls) | Funciona | Gap de carry excluido del retorno |
| ETF trick (§2.4.1) | Funciona | Pasa 5/5 validaciones independientes |
| Tick rule (§2.3.2.1) | Funciona | Contrastada con el agresor del exchange |
| Tick / Volume / Dollar bars | Funciona | Umbrales calibrados a igual recuento |
| **Dollar Imbalance Bars (§2.3.2.2)** | **No converge** | Ver abajo |
| Consolidación y verificación de datos | Funciona | `consolidar_raw.py`, `verificar_raw.py` |
| Diagnósticos (Hurst, variance ratio) | Pendiente de reincorporar | El guion no está en el repo |

### El problema abierto: las Imbalance Bars

La condición de cierre del libro es

```
T* = arg min_T { |θ_T| ≥ E_0[T] · |2v⁺ − E_0[v_t]| }
```

Sobre estos datos **el umbral no converge**: o deja pasar todos los ticks y la
barra cierra en T = 1, o explota y una sola barra absorbe gran parte de la
muestra. No hay rango intermedio utilizable.

Esto **no está resuelto**. Lo que hay hasta ahora son indicios de que el
problema está en el estimador y no en la implementación —la relación
señal/ruido de `E[b·v]` es de varios miles, y el flujo firmado tiene memoria
larga— pero falta trabajo para afirmarlo con solidez. La figura
`data/figures/ES_2024_dib_threshold_trajectory.png` muestra la trayectoria del
umbral.

**Decisión de diseño:** las ecuaciones de AFML no se modifican. No se añaden
suelos, techos ni tamaños objetivo para forzar barras de tamaño razonable.
Cuando el estimador degenera, el código lo detecta y lo reporta en lugar de
reajustar el umbral.

## Estructura

```
├── programs/
│   ├── afml_core.py          define los objetos de AFML; no ejecuta experimentos
│   ├── Exercise_2_1_a.py     el experimento: ETF trick, barras, DIB
│   ├── consolidar_raw.py     une trozos de descarga en un parquet por contrato
│   ├── verificar_raw.py      comprueba que los contratos son homogéneos
│   └── Lector_Parquet.py     inspector de parquet
└── data/
    ├── raw/                  ticks (no versionados)
    ├── bars/  returns/       salidas del pipeline
    ├── figures/              gráficas
    └── reports/              logs de ejecución
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt

cd programs
python verificar_raw.py                   # comprobar que los datos son homogéneos
python Exercise_2_1_a.py --max-ticks 3e6  # prueba rápida
python Exercise_2_1_a.py                  # muestra completa
```

> **Memoria:** la muestra completa llega a usar unos 10 GB de RAM. Con menos
> de 16 GB, usar `--max-ticks`.

Los núcleos de cálculo están compilados con **Numba**; sin eso, 92 millones de
ticks en Python puro no serían viables.

## Pendiente

- Determinar si las Imbalance Bars son recuperables sobre estos datos o si la
  degeneración es inevitable con este estimador.
- Reincorporar el guion de diagnósticos (`diagnostics_2_1_a.py`).
- Pruebas unitarias de los núcleos de cálculo.
- Fijar una licencia.

## Referencia

López de Prado, M. (2018). *Advances in Financial Machine Learning*, cap. 2.
Wiley.
