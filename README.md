# IMICA V3.0 — EV + LCO

Esta carpeta corresponde a la tercera y última versión del proyecto, enfocada en un caso experimental más limpio y defendible para la publicación final.

## Alcance

- Aplicación: EV
- Química: LCO
- Enfoque: evaluación de la capacidad de optimización de IMICA en un problema
  reproducible EV/LCO y comparación justa con NSGA-III
- Objetivo: dejar una versión final reproducible y sustentada por resultados consistentes con la revisión

## Estructura

- `src/`: implementación principal del algoritmo y del pipeline de optimización
  - `core/`: lógica central del problema y flujo de ejecución
  - `optimization/`: IMICA, IRBO, NSGA-III, indicadores y selección
  - `surrogate/`: entrenamiento y evaluación del surrogate
  - `utils/`: utilidades generales
- `experiments/EV_LCO/`: experimentos específicos para EV + LCO
  - `configs/`: archivos de configuración
  - `scripts/`: runners para cada experimento
  - `results/`: salidas y métricas
- `validation/`: verificaciones físicas y de modelo
  - `p2d/`: chequeos con P2D
  - `surrogate/`: validación del surrogate
- `docs/`: documentación de alcance, procedimiento y revisión
- `paper/`: figuras y tablas para el manuscrito final
- `data/`: datos de entrada y artefactos generados

## Enfoque de la versión

La V3.0 no intenta generalizar todas las aplicaciones ni todas las químicas. Su objetivo es consolidar una versión final sólida y enfocada en el caso EV/LCO, que es el escenario más relevante y fácil de validar científicamente.

## Estado actual del algoritmo

IMICA mantiene la estructura original de islas con indicadores de calidad por
isla; su implementación está en `src/optimization/imica.py`. El optimizador
de benchmark basado en energía de Riesz se denomina Island Riez-Based Optimizer
(IRBO) y está separado en `src/optimization/irbo.py`. Ambos comparten
operadores mixtos y el adaptador EV/LCO traduce el objetivo de energía a
minimización y normaliza las restricciones de voltaje.

Prueba corta:

```powershell
python -m unittest discover -s tests -v
python experiments\EV_LCO\scripts\run_imica.py
python experiments\EV_LCO\scripts\run_irbo.py
```

**Importante:** el evaluador de batería de `src/core/battery.py` sigue siendo
un baseline provisional y no está conectado al núcleo `src/p2d/`. El P2D
incluye un set de referencia Ai2020 con OCP LCO/grafito rastreables para
verificación numérica, pero tampoco está parametrizado ni validado contra la
celda experimental objetivo. Por tanto, los resultados del artículo se
limitarán a comparar el desempeño de optimización sobre el problema de
referencia definido; no demostrarán la precisión del modelo de batería ni la
validez de sus predicciones para una celda real. La validación experimental se
plantea como trabajo futuro.

Pendiente: conectar los evaluadores de IMICA e IRBO al P2D de referencia,
restringirlo a un dominio numéricamente válido, completar la
selección/referencia del algoritmo frente a la especificación de V2 y definir
el protocolo de comparación.

## Base del simulador P2D

La base numérica de malla axial, difusión radial, transporte de electrolito,
cinética y descarga galvanostática está en `src/p2d/`. El avance de especies
usa división de operadores; el set Ai2020 aporta parámetros y OCP de referencia
para verificación numérica. El módulo P2D aún no está conectado al evaluador
que utiliza IMICA, y su respuesta no ha sido validada frente a la celda
experimental; el alcance y las limitaciones están en
`docs/p2d_implementation.md`.
