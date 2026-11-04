# Experimentos EV + LCO

Esta carpeta concentra los experimentos computacionales de la versión final:
evaluar la capacidad de optimización de IMICA en el problema EV/LCO y
compararla con NSGA-III con el mismo protocolo y presupuesto.

## Subcarpetas

- `configs/`: archivos de configuración del problema y del algoritmo
- `scripts/`: scripts para correr cada experiencia
- `results/`: salidas, métricas y logs

## Nota

Los resultados caracterizan el desempeño de los algoritmos en la función
objetivo computacional definida, no validan predicciones contra una celda real.
La validación de la celda experimental queda fuera del artículo y se plantea
como trabajo futuro. Si se usa el P2D de referencia como evaluador, deben
documentarse sus parámetros, el dominio numéricamente válido y los casos de
fallo. El objetivo es usar el P2D Ai2020 de referencia para este benchmark,
pero todavía no está conectado al evaluador que usa IMICA.
