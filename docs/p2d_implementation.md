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

## Evaluador de diseño P2D

`src/core/p2d_evaluator.py` añade `P2DEvaluator.evaluate(design)`, que construye
una malla con `lp`, `lo` y `ln`, configura el set Ai2020, inicializa el estado
de carga y ejecuta una descarga galvanostática determinista. `Rp` y `Rn`
definen radios de partícula; `efp` y `efn` se interpretan como fracciones
volumétricas de material activo, y `efo` como fracción sólida del separador.
El resto del volumen en cada región se trata como porosidad de electrolito.
Esta interpretación excluye explícitamente el volumen de aglutinante y
aditivos conductores. Los colectores `la` y `lz` contribuyen a la masa areal.
Variables como `Ns`, `Np`, `Lh` y `Rcell` no alteran todavía el cálculo 1D.

La tasa `C` se convierte a densidad de corriente con la menor capacidad areal
teórica de los dos electrodos, calculada a partir de sus intervalos OCP. La
estequiometría inicial se coloca dentro del rango tabulado, a una distancia de
`1e-4` del intervalo normalizado. La energía específica reportada es positiva
y se calcula integrando `V(t) * I` y dividiendo entre la masa areal estimada
de electrodos activos, separador, electrolito y colectores. El resultado
incluye objetivo `specific_energy`, restricciones firmadas (factibles si son
`<= 0`), métricas y arrays de tiempo/voltaje.

El objetivo `ohmic_energy_loss_j_m2` integra en el tiempo la disipación Joule
areal `sum(i^2 / sigma_eff * dx)` en las fases sólidas positiva y negativa y
en el electrolito (incluido el separador). Usa las corrientes por cara
promediadas a cada volumen de control, conductividades sólidas efectivas del
modelo Ai2020 y conductividad efectiva del electrolito en el estado local. La
métrica `ohmic_energy_loss_fraction` es la pérdida Joule dividida por la suma
de esa pérdida y la energía eléctrica terminal entregada; no incluye pérdidas
de activación, difusión, colectores, contactos ni calor reversible. La
trayectoria expone también `ohmic_power_loss_w_m2` en cada tiempo de voltaje.
Esta fracción es una medida normalizada de las pérdidas óhmicas modeladas, no
una eficiencia energética total de celda.

El objetivo `price` estima el costo total de materiales del pack en USD. Para
cada celda se calcula la masa de LCO, grafito, separador, electrolito y los
colectores de aluminio/cobre a partir de sus espesores, fracciones volumétricas,
densidades y área enrollada; cada masa se multiplica por el precio unitario
configurado en `src/core/materials.py`. El costo por celda se escala por
`Ns * Np`. `metrics.material_costs_usd` desglosa los costos por material para
todo el pack, y `metrics.material_masses_kg` las cantidades totales utilizadas.
El grafito usa los valores heredados de la línea base (2160 kg/m³ y 60 USD/kg);
los precios son supuestos de entrada, no cotizaciones de mercado ni un costo
completo de fabricación. No se incluyen aglutinantes, aditivos conductores,
carcasa, conexiones, refrigeración, ensamblaje ni mermas.

Las restricciones incluyen límites de estequiometría OCP, concentración no
negativa del electrolito, voltaje máximo de 4.3 V y terminación de descarga.
El corte por defecto es 2.5 V. El límite de 4.3 V es configurable y constituye
un límite de cribado, no una calibración de la celda experimental. El solver
detiene la descarga en el último estado válido si el siguiente paso cruzaría
el intervalo tabulado de estequiometría. Por defecto se permiten hasta `1/C`
horas de simulación, con paso de 10 s; se puede limitar `maximum_steps` para
pruebas rápidas, caso en el cual una descarga truncada queda marcada como
restricción positiva. Los errores de dominio y convergencia del solver se
propagan explícitamente: una corrida de estrés a 4C en la malla 2/1/2 presentó
concentración de electrolito no positiva incluso al extenderla, por lo que ese
régimen no se considera resuelto ni válido; hay que estudiar el paso temporal,
la malla y el dominio antes de usarlo.

Este evaluador aún no reemplaza el `EVLCOBattery` del adaptador
`EVLCOProblem`: el acoplamiento con IMICA sigue pendiente. La masa es una
estimación con las hipótesis anteriores. Las simulaciones usan los parámetros
Ai2020 de referencia, no una parametrización validada de la celda experimental.

## Ciclos CC y crecimiento de SEI

`GalvanostaticElectrochemistry.solve` admite corriente con signo: positiva
para descarga y negativa para carga. `simulate_constant_current` permite fijar
los límites de ambas fases e interpola linealmente el último intervalo para
estimar el instante de cruce. El estado de especies y el espesor de SEI se
conservan entre medias fases.

Por defecto, `P2DEvaluator` usa una malla de 8/3/8 celdas axiales
(positivo/separador/negativo) y 12 celdas radiales por partícula, y calcula
además 20 ciclos completos a 1C, 25 °C, con carga CC hasta 4.0 V y descarga
CC hasta 3.4 V, sin fase CV. Esta
configuración fue elegida para permanecer dentro del dominio OCP Ai2020: una
descarga de 1C con corte de 2.5 V alcanzó el límite tabulado de estequiometría
a 3.04 V. Los cortes de durabilidad son configurables por separado del corte
de 2.5 V usado por el cálculo de energía. Cada ciclo empieza desde el estado
descargado, ejecuta carga y luego descarga; la capacidad se mide durante esta
última. El objetivo `capacity_fade_fraction` compara la capacidad entregada en
la primera descarga con la de la descarga 20. `metrics` también informa el
crecimiento medio de SEI, el espesor final y la pérdida integrada de litio.
`aging_cycles=None` desactiva esta evaluación adicional. Si una fase no
alcanza su límite de voltaje o sale del dominio de estequiometría, la evaluación
falla explícitamente en vez de reportar una pérdida de capacidad incompleta.

El submodelo usa la cinética limitada por reacción de SEI,
`j_SEI = -j0 exp(-alpha F eta_SEI / (R T))`, con `eta_SEI` incluyendo el
potencial de equilibrio y la caída resistiva de la película. Los valores
`j0=1.5e-7 A/m²`, `R_SEI=2e5 Ohm.m`, `Vbar=9.585e-5 m³/mol`, razón
Li/SEI `2`, `U_SEI=0.4 V` y espesor inicial `5 nm` proceden del conjunto
PyBaMM Ai2020, [`Ai2020.py` en el commit b4b6089](https://github.com/pybamm-team/PyBaMM/blob/b4b6089a65f736fe2444ecf4536a297cbfbd3420/packages/pybamm/src/pybamm/input/parameters/lithium_ion/Ai2020.py).
Son parámetros de referencia para simulación, no una calibración de esta celda.
La corriente parásita se incluye en el balance de carga del ánodo y el
electrolito; el avance temporal sigue siendo secuencial (operator-split), no
un sistema P2D monolítico. No se modelan otros mecanismos de degradación ni
variación térmica. El resultado de durabilidad es exploratorio y requiere
estudio de convergencia temporal/malla y validación experimental antes de
interpretarse como vida útil real.

Una prueba de dos ciclos a 60 s por paso, malla 2/1/2 y 4 celdas radiales
completó ambos límites: la capacidad simulada pasó de 24.1445 a 24.1365 Ah/m²
(0.0333 % de pérdida). Esta corrida gruesa solo comprueba el flujo y la
finalización numérica del ciclo, no la convergencia temporal ni la validez
física del pronóstico. Una evaluación de un ciclo a 10 s por paso tardó 212 s
en una malla reducida 2/1/2 con 4 celdas radiales. No se ha medido el runtime
de los nuevos valores por defecto; por su mayor resolución y 20 ciclos, cada
evaluación será sustancialmente más costosa y puede tardar muchas horas según
el equipo. Se recomienda cronometrar una corrida representativa antes de un
estudio amplio del MOP.

Para medir runtime y tasa de fallas sobre 30 diseños aleatorios reproducibles
muestreados de los límites de `EV_LCO_VARIABLES`, ejecutar desde la raíz de
`V3_0`:

```powershell
python experiments\EV_LCO\scripts\benchmark_p2d_random.py
```

El benchmark usa por defecto semilla 2026, 20 ciclos y la malla 8/3/8 con 12
celdas radiales del evaluador. Registra el tiempo individual, éxito/falla,
excepción y variables en un CSV con nombre timestamped, y guarda estadísticas
agregadas y configuración en un archivo `.summary.json` adyacente. El proceso
continúa aunque falle una evaluación. Para una prueba corta se pueden cambiar
`--count`, `--aging-cycles`, `--maximum-steps`, `--time-step-s`, los cuatro
parámetros de malla, la semilla y la ruta con `--output`; cualquier reducción
de resolución debe indicarse al comparar el porcentaje de fallas.

La rutina de descarga admite OCPs tabuladas. Las pruebas del solver genérico
usan curvas sintéticas; las pruebas de la factoría Ai2020 usan el set de
referencia y no representan por sí solas la celda objetivo. Las dependencias
NumPy/SciPy están declaradas en `requirements.txt`.
