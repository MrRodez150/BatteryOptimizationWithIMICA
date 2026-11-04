# Implementación inicial de IMICA

## Alcance implementado

La implementación en `src/optimization/` es una primera versión ejecutable del
núcleo evolutivo para el caso EV + LCO:

- muestreo de una población distribuida entre islas;
- variables reales e enteras con SBX y mutación polinomial;
- dominancia con factibilidad primero y violación agregada para soluciones no
  factibles;
- selección de supervivientes por frentes de Pareto y energía de Riesz
  normalizada;
- referencia NDR no dominada independiente por isla, acumulada entre
  generaciones y podada por Riesz cuando supera el tamaño de población;
- migración periódica unidireccional entre islas vecinas;
- archivo externo con las soluciones no dominadas observadas;
- presupuesto de evaluaciones aplicado exactamente, incluida la población
  inicial.

La referencia se actualiza con la unión de su estado anterior y los nuevos
candidatos. Cuando aparece al menos una solución factible, conserva solamente
soluciones factibles y filtra la dominancia en el espacio de objetivos. Si aún
no hay soluciones factibles, filtra en el espacio aumentado por las
violaciones positivas normalizadas. Los puntos duplicados se eliminan antes de
calcular Riesz. La referencia entra a la selección ambiental de su isla, para
que sus soluciones históricas puedan sobrevivir y no sea solo un registro.

`EVLCOProblem` adapta la interfaz actual del simulador: convierte energía
específica a minimización, mantiene los otros tres objetivos en minimización y
escala las violaciones de voltaje por el voltaje del paquete.

## Limitaciones deliberadas

Esta primera etapa no pretende reproducir todavía todas las variantes y
configuraciones experimentales de V2. En particular, aún no incorpora los modos
CSR, los indicadores alternativos para supervivencia, guardado/reanudación de
experimentos ni ejecución paralela. La referencia NDR implementada es
constraint-aware y debe validarse contra el protocolo final antes de usarla en
resultados publicables.

El evaluador `EVLCOBattery` todavía no es P2D: utiliza perfiles de voltaje y
temperatura provisionales. Por ello, el runner actual sirve para probar el
flujo y el presupuesto del algoritmo, no para afirmar que ya se optimiza el
modelo P2D. El benchmark del artículo usará el P2D Ai2020 de referencia una vez
integrado como evaluador; sus resultados sustentarán conclusiones sobre el
desempeño relativo de los algoritmos en ese problema definido, no sobre la
exactitud física o predictiva de una celda real. La validación experimental
de la celda queda como trabajo futuro.

## Comparación con NSGA-III

`src/optimization/nsga3.py` implementa NSGA-III con direcciones de referencia
de Das-Dennis, normalización por puntos extremos/interceptos y niching en el
frente de corte. Reutiliza el evaluador EV/LCO y los operadores de variables
mixtas de IMICA, para que la comparación se centre en la selección ambiental.
El script `experiments/EV_LCO/scripts/compare_imica_nsga3.py` ejecuta ambos con
el mismo presupuesto exacto y semillas emparejadas.

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
optimizer = IMICA(
    problem,
    population_size=8,
    islands=2,
    migration_interval=5,
    migrants=1,
    seed=42,
)
result = optimizer.run(max_evaluations=128)
```

El resultado incluye el archivo no dominado, la población de cada isla, el
número exacto de evaluaciones y el número de generaciones.
