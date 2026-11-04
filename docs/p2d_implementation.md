# Base del simulador P2D de V3

## Decisión de implementación

La V3 reutilizará la formulación propia de la tesis V1, pero trasladada a un
paquete independiente y con correcciones verificables. El solver V1 original
depende de JAX y de un Jacobiano bandado ensamblado manualmente; JAX no está
instalado en el entorno Python activo. La primera base de V3 usa NumPy y
discretización de volúmenes finitos para poder probar conservación y
condiciones de frontera sin copiar todavía el Jacobiano legacy.

## Componentes implementados en esta etapa

- `ElectrochemicalGrid1D`: malla axial por volúmenes finitos para cátodo,
  separador y ánodo, con interfaces explícitas.
- `SphericalParticleDiffusion`: difusión radial esférica de concentración en
  el sólido con Euler implícito.
- El flujo molar superficial se define positivo hacia afuera de la partícula.
  Su balance global es
  `N(t+dt)-N(t) = -4*pi*R**2*J_out*dt`.
- La interfaz de conversión de corriente usa `J_out = j/F`; el signo del
  `j` electroquímico se asignará explícitamente al integrar las reacciones de
  cátodo y ánodo.
- La dependencia térmica de `D_s` usa
  `D_s(T)=D_s,ref*exp(E_D/R*(1/T_ref-1/T))`, que aumenta con T para `E_D > 0`.
- `ElectrolyteTransport`: balance conservativo implícito para `c_e`, con
  coeficientes efectivos por porosidad, promedio armónico en las interfaces,
  fuentes `(1-t_+) a_s j/F` y fronteras externas sin flujo.
- El potencial del electrolito se obtiene de la conservación de carga y de
  `i_e=-kappa_eff grad(phi_e)+2(1-t_+)RT/F*kappa_eff grad(ln(c_e))`.
  La solución fija `phi_e[0]=0` como referencia de gauge y verifica que las
  corrientes de reacción concuerden con las corrientes iónicas de frontera.
- `ButlerVolmerKinetics`: cinética simétrica con la forma de corriente de
  intercambio heredada de V1. `k*sqrt(c_e*c_s*(c_s,max-c_s))` se interpreta
  como flujo molar (mol m⁻² s⁻¹); la densidad de corriente interfacial se calcula como
  `i_F=F*J`. El corriente volumétrica que entra en los balances es `a_s*i_F`.
- `solve_solid_potential`: integración conservativa de la corriente electrónica
  para un electrodo, `i_s=-sigma_eff grad(phi_s)` y
  `i_s,right-i_s,left=-integral(a_s*i_F dx)`, con conductividad efectiva por
  celda, continuidad de corriente y referencia de potencial explícita.
- `TabulatedOpenCircuitPotential`: interpolación de OCP con fuente obligatoria,
  corrección entrópica opcional y sin extrapolación fuera del rango tabulado.
  La configuración de referencia usa OCP LCO de Rieger (2016), distribuida en
  PyBaMM Ai2020, y OCP de grafito Enertech; sus fuentes y licencia están en
  `data/p2d_parameters/`.
- `GalvanostaticElectrochemistry`: cierre algebraico no lineal a corriente
  impuesta para la distribución de reacción, ambos potenciales sólidos,
  potencial de electrolito, Butler–Volmer y voltaje entre colectores. Comprueba
  balances de corriente en ambos electrodos y corriente iónica neta. El sistema
  se resuelve con mínimos cuadrados Levenberg–Marquardt y se acepta solo si el
  residual escalado cumple tolerancia.
- `advance_species`: paso secuencial que actualiza electrolito y perfiles
  radiales de partículas usando la reacción del estado electroquímico. Es un
  operador dividido explícito respecto al estado de carga, no un paso temporal
  no lineal monolítico.
- `simulate_constant_current`: integrador de descarga acotado por número de
  pasos y cutoff de voltaje; devuelve `V(t)`, tiempo y estado final.
- Las correlaciones heredadas para `D_e(c_e,T)` y `kappa(c_e,T)` quedan
  separadas de la porosidad: `D_eff=D_e*psi^b` y
  `kappa_eff=kappa*psi^b`. La conductividad usa el polinomio completo al
  cuadrado una sola vez, conforme a la corrección documentada en P3.

## Parámetros y alcance físico

`build_ai2020_reference_model` proporciona un conjunto reproducible de
parámetros Ai2020 para pruebas y análisis numérico, no una parametrización
validada de la celda experimental de esta investigación. Las fuentes,
versiones y límites de las tablas OCP se describen en
`data/p2d_parameters/README.md`. La simulación actual es isotérmica a 298.15 K;
el ajuste entrópico de grafito no está incluido. No deben presentarse estos
datos como validación experimental ni como predicciones validadas para la
celda objetivo. El artículo puede usar el modelo de referencia como problema
computacional para estudiar el desempeño de optimización, dejando explícito
que sus conclusiones se restringen a esa parametrización y no a una celda real.
La caracterización de la celda y su validación experimental quedan como
trabajo futuro.

## Verificación numérica actual

En la malla de referencia de 2/1/2 celdas axiales y 8 celdas radiales por
partícula, una descarga de 100 A/m² durante 10 s produjo voltajes finales de
3.6781348993, 3.6779491664 y 3.6778547270 V con pasos de 1, 0.5 y 0.25 s,
respectivamente. La diferencia se reduce al refinar el paso, pero esta prueba
corta no demuestra convergencia global ni exactitud experimental.

El cierre galvanostático converge a 300 A/m² en el estado inicial de referencia.
Sin embargo, una simulación de 500 A/m² durante 10 s puede detenerse porque la
concentración calculada del electrolito deja de ser positiva; esta corrida no
se considera válida y muestra que ese régimen todavía no está verificado.

## Siguientes etapas de física

1. Validar las tablas OCP, el ajuste entrópico y parámetros frente a la celda
   objetivo. La expresión
   racional heredada de V1 no se copia: al evaluarla, su denominador cruza cero
   dentro del dominio físico de estequiometría (alrededor de `theta=0.277`).
   La parametrización atribuida a Ramadass et al. (2004), expuesta en
   [PyBaMM](https://github.com/pybamm-team/PyBaMM/blob/main/packages/pybamm/src/pybamm/input/parameters/lithium_ion/Ramadass2004.py),
   usa un factor de estiramiento de 1.13; al evaluar el denominador se detectan
   polos en el intervalo de estequiometría 0.1–0.9. Por ello no la usamos sin
   validar/reconstruir un intervalo de operación y ajuste seguro. Referencia
   reportada por PyBaMM: P. Ramadass et al., “Development of First Principles
   Capacity Fade Model for Li-Ion Cells” (2004).
2. Reemplazar `advance_species` por un paso temporal monolítico si estudios
   adicionales demuestran que la división de operadores es insuficiente.
3. Ampliar los estudios de convergencia de residual, malla, tiempo y
   conservación de litio; la prueba temporal corta actual solo confirma
   refinamiento para el set de referencia.
4. Añadir el balance térmico solo después de corregir las fuentes de calor y
   escoger un modelo de disipación a escala de celda.
5. Validar malla y tiempo, y contrastar el voltaje contra una fuente
   independiente antes de generar datos de surrogate.

Esta etapa añade OCPs y parámetros de referencia rastreables, un cierre
galvanostático no lineal de potenciales y cinética, y un avance secuencial de
especies. No incluye un paso temporal monolítico ni validación externa de la
celda de investigación. Las pruebas unitarias y de convergencia verifican
comportamiento numérico del código; no validan la electroquímica de la celda.
El evaluador `EVLCOBattery` que usa IMICA todavía no está conectado a este
módulo P2D. Antes de usar P2D en la comparación de optimizadores, se debe
integrar explícitamente y restringir el dominio experimental a estados donde
el solver entregue resultados numéricamente válidos.

La rutina de descarga admite OCPs tabuladas. Las pruebas del solver genérico
usan curvas sintéticas; las pruebas de la factoría Ai2020 usan el set de
referencia y no representan por sí solas la celda objetivo. Las dependencias
NumPy/SciPy están declaradas en `requirements.txt`.
