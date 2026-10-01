# Red de Hopfield

Implementación en Python (NumPy) de una red de Hopfield discreta como memoria asociativa: se entrena con
cuatro letras de 7×7 (A, T, X, L) y recupera cada una a partir de una versión con ruido.

**Autor:** Daniel Alfredo Barreras Meraz (A01254805) · TC3002B.570, Desarrollo de aplicaciones avanzadas de ciencias computacionales · Tec de Monterrey

## Cómo funciona

| Concepto | Fórmula | Dónde |
|---|---|---|
| Estados | neuronas bipolares, `s_i ∈ {-1, +1}` | `hopfield/network.py` |
| Aprendizaje (regla de Hebb) | `W = (1/N) Σ_p x_p x_pᵀ`, con `W_ii = 0` | `HopfieldNetwork.train` |
| Actualización | `s_i ← sign(Σ_j W_ij s_j)` | `HopfieldNetwork.recall` |
| Energía | `E(s) = -½ sᵀ W s` | `HopfieldNetwork.energy` |
| Capacidad | ≈ 0.138 N patrones | `HopfieldNetwork.capacity` |

- **Pesos simétricos y sin autoconexiones.** Con esas dos condiciones, cada actualización asíncrona
  (una neurona a la vez) no aumenta la energía, así que la red siempre llega a un punto fijo.
- **Patrones como mínimos.** Los patrones entrenados quedan como mínimos locales de la energía. Un patrón
  con ruido "cae" al mínimo más cercano, que normalmente es el patrón original.
- **Modo síncrono.** `recall(mode="sync")` actualiza todas las neuronas a la vez. Es más rápido, pero
  puede oscilar entre dos estados; por eso el modo por defecto es el asíncrono.
- **Capacidad.** Con 49 neuronas, la capacidad teórica es de unos 6.8 patrones; se usan 4. Si se guardan
  más patrones de los que soporta, o patrones muy parecidos entre sí, aparecen estados espurios
  (mezclas que no corresponden a ningún patrón).

## Uso

```bash
pip install -r requirements.txt
python demo.py                 # 20 % de ruido, semilla 7
python demo.py --noise 0.3     # otro nivel de ruido
python -m unittest discover -s tests -t . -v
```

```python
from hopfield import HopfieldNetwork, add_noise, letter_patterns

names, patterns = letter_patterns()
net = HopfieldNetwork(patterns.shape[1]).train(patterns)
recovered, energies = net.recall(add_noise(patterns[0], 0.2))
```

## Resultados (`python demo.py`)

Cada letra con 10 de sus 49 bits invertidos (20 %) se recupera exactamente en 2 barridos asíncronos.

![Recuperación](results/recuperacion.png)

La energía baja en cada iteración hasta estabilizarse en el mínimo del patrón:

![Energía](results/energia.png)

Porcentaje de recuperación exacta según el nivel de ruido (50 pruebas por nivel):

| Ruido | 0–15 % | 20 % | 25 % | 30 % | 35 % | 40 % | 45 % | 50 % |
|---|---|---|---|---|---|---|---|---|
| Recuperación exacta | 100 % | 90 % | 88 % | 74 % | 74 % | 32 % | 14 % | 8 % |

![Ruido vs. éxito](results/ruido_vs_exito.png)

Hasta 15 % de ruido la recuperación es perfecta. A partir del 40 % el patrón con ruido ya se parece más a
otra letra, o al inverso de un patrón (que también es un mínimo de energía), y la red converge ahí.

## Pruebas

`tests/test_hopfield.py` (unittest) verifica:
- que la matriz de pesos sea simétrica y tenga la diagonal en cero;
- que los patrones entrenados sean puntos fijos;
- que se recuperen con 10 % de ruido;
- que la energía nunca aumente en modo asíncrono;
- que se rechacen entradas inválidas.

## Estructura

```
hopfield/network.py    red de Hopfield y función de ruido
hopfield/patterns.py   letras de 7×7
demo.py                entrenamiento, recuperación y figuras
tests/                 pruebas unitarias
results/               figuras generadas por demo.py
```

## Referencia

Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational
abilities. *Proceedings of the National Academy of Sciences, 79*(8), 2554–2558.
https://doi.org/10.1073/pnas.79.8.2554
