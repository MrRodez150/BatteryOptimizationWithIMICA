# IMICA V3.0 — EV + LCO

Esta carpeta corresponde a la tercera y última versión del proyecto, enfocada en un caso experimental más limpio y defendible para la publicación final.

## Alcance

- Aplicación: EV
- Química: LCO
- Enfoque: formulación física coherente, validación con P2D y comparación justa con NSGA-III
- Objetivo: dejar una versión final reproducible y sustentada por resultados consistentes con la revisión

## Estructura

- `src/`: implementación principal del algoritmo y del pipeline de optimización
  - `core/`: lógica central del problema y flujo de ejecución
  - `optimization/`: IMICA, NSGA-III, indicadores y selección
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

## Siguientes pasos

1. Definir variables de diseño relevantes para EV/LCO
2. Reescribir la formulación de objetivos y restricciones
3. Implementar el pipeline de surrogate
4. Ejecutar experimentos reproducibles
5. Validar frente a P2D
6. Generar resultados y figuras finales
