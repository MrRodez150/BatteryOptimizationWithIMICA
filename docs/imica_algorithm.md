# IMICA y el Island Riez-Based Optimizer (IRBO)

## IMICA con indicadores por isla

`IMICA` conserva la estructura original basada en indicadores y archipiélagos:

- cada isla mantiene una población, una referencia NDR y un indicador de
  calidad asignado;
- variables reales e enteras con SBX y mutación polinomial;
- dominancia con factibilidad primero y violación agregada para soluciones no
  factibles;
- selección de supervivientes por frentes de Pareto, usando el indicador propio
  de la isla para truncar el frente que excede la capacidad;
- referencias NDR no dominadas independientes por isla, acumuladas entre
  generaciones y podadas por energía de Riesz;
- migración periódica entre islas, seguida de selección con el indicador de
  cada isla;
- archivo externo con las soluciones no dominadas observadas;
- presupuesto de evaluaciones aplicado exactamente, incluida la población
  inicial.

El conjunto por defecto asigna, en orden, hipervolumen (`SMS`), `R2`, `IGD+`,
epsilon+ y `DeltaP`; si hay más islas, la secuencia se repite. También se puede
pasar exactamente un indicador por isla. `IGD+`, epsilon+ y `DeltaP` usan la
referencia NDR de esa isla como aproximación al frente. Los objetivos y la
referencia se normalizan conjuntamente antes de calcular el indicador. Los
valores por isla y su historial de generaciones se devuelven en
`indicator_values` e `indicator_history`.

La referencia se acumula con los candidatos nuevos. Si hay puntos factibles,
solo conserva puntos factibles no dominados; si todavía no los hay, aplica
dominancia constraint-aware. La referencia orienta los indicadores que la
requieren, pero no se inyecta como población histórica.

Ejemplo:

```python
from src.optimization import (
    EVLCOProblem, IMICA, SMS, R2, IGDPlus, EpsilonPlus, DeltaP
)

optimizer = IMICA(
    EVLCOProblem(),
    population_size=40,
    indicators=[SMS(), R2(), IGDPlus(), EpsilonPlus(), DeltaP()],
    migration_interval=10,
    migrants=2,
    seed=42,
)
result = optimizer.run(max_evaluations=2000)
```

La selección de padres admite `select_mode="Rand"` y `select_mode="Roulette"`.
Roulette asigna mayor probabilidad a frentes de mejor rango de dominancia.

## Island Riez-Based Optimizer (IRBO)

La implementación evolutiva de benchmark basada en dominancia
constraint-aware, selección por energía de Riesz y migración en anillo ahora
es un optimizador independiente llamado `IRBO` (Island Riez-Based Optimizer).
Está separado de IMICA en `src/optimization/irbo.py`; la implementación
indicadora de IMICA permanece en `src/optimization/imica.py`. IRBO no calcula
indicadores distintos por isla. El runner `run_irbo.py` y la comparación con
NSGA-III usan IRBO para preservar el benchmark existente. Los runners
`run_imica.py` y `run_irbo.py` ejecutan cada método por separado.

## Alcance y limitaciones

Se implementan los indicadores HV, R2, IGD+, epsilon+ y DeltaP. NDR es el modo
de referencia soportado actualmente; no se ha portado el modo CSR, el guardado
y recuperación de experimentos ni la ejecución paralela de V2. La referencia
NDR es una aproximación dependiente de cada isla, no el frente de Pareto
verdadero; los valores IGD+, epsilon+ y DeltaP no son métricas contra una
solución exacta.

`EVLCOProblem` adapta la interfaz actual del evaluador: convierte energía
específica a minimización y escala las violaciones de voltaje por el voltaje
del paquete. `EVLCOBattery` todavía no está conectado al simulador P2D y usa
perfiles de voltaje y temperatura provisionales. Los resultados sirven para
probar el flujo de optimización, no para afirmar predicciones validadas de una
celda real.

## Comparación con NSGA-III

`src/optimization/nsga3.py` implementa NSGA-III con direcciones de referencia
de Das-Dennis, normalización por puntos extremos/interceptos y niching en el
frente de corte. Reutiliza el evaluador EV/LCO y los operadores de variables
mixtas. El script `experiments/EV_LCO/scripts/compare_irbo_nsga3.py` compara
NSGA-III con `IRBO`, con presupuesto exacto y semillas emparejadas.

Al no conocerse el frente de Pareto verdadero, el reporte calcula hipervolumen
con punto de referencia 1.1 en objetivos normalizados y `IGD+` contra la unión
no dominada factible de los archivos de ambos algoritmos para esa semilla.
También reporta ocupación de direcciones y variación de distancias entre
vecinos. Esta referencia es una aproximación dependiente de los algoritmos,
no una solución exacta: las métricas permiten una comparación empírica, pero
no prueban optimalidad global ni superioridad universal. Se requieren varias
semillas y análisis pareado. Las conclusiones deberán limitarse al desempeño
observado sobre el problema de referencia; la validación electroquímica frente
a la celda experimental es un trabajo futuro independiente.

## API breve

```python
problem = EVLCOProblem()
optimizer = IRBO(
    problem,
    population_size=8,
    islands=2,
    migration_interval=5,
    migrants=1,
    seed=42,
)
result = optimizer.run(max_evaluations=128)
```

Ambos optimizadores devuelven el archivo no dominado, la población de cada
isla, el número exacto de evaluaciones y el número de generaciones. `IMICA`
también devuelve nombres, valores finales e historial de los indicadores por
isla.
