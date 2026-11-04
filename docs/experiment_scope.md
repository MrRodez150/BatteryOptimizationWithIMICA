# Alcance experimental de la V3.0

## Caso de estudio

- Aplicación: EV
- Química: LCO

## Objetivo

Evaluar la capacidad de optimización de IMICA para un problema reproducible
EV/LCO y compararla con NSGA-III bajo un protocolo común. La evaluación del
algoritmo no equivale a validar las predicciones electroquímicas contra una
celda física.

## Aspectos a definir

- Variables de diseño del problema EV/LCO
- Objetivos relevantes para la aplicación
- Restricciones esenciales de operación y manufactura
- Definición reproducible de la función objetivo EV/LCO y sus restricciones
- Configuración del problema, presupuesto y protocolo de optimización
- Comparación de IMICA y NSGA-III con múltiples semillas y métricas
  multiobjetivo apropiadas
- Usar el P2D Ai2020 de referencia como evaluador del problema de optimización,
  una vez conectado al algoritmo, documentando el set y acotando el dominio a
  condiciones donde la simulación sea numéricamente válida

## Fuera del alcance del artículo

No se validarán las predicciones del P2D contra la celda experimental objetivo
ni se afirmará que el set de referencia representa esa celda. La
caracterización de la celda y la validación experimental se reportarán como
trabajo futuro. Las conclusiones del artículo se limitarán al desempeño
observado de los algoritmos en el problema computacional especificado; no
demostrarán optimalidad global ni superioridad universal de IMICA.

## Reglas de diseño

- Mantener un único caso de aplicación
- Evitar dispersión en materiales o escenarios
- Usar definiciones físicas consistentes de energía, corriente y térmica
- Decidir un único protocolo experimental para todos los resultados
