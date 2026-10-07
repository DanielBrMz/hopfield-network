#!/usr/bin/env python3
"""Reconocimiento de figuras (círculo, cuadrado, triángulo, cruz, rombo) con una red de Hopfield de 144 neuronas.

Compara la regla de Hebb con la regla de proyección (pseudoinversa) y prueba la red con ruido y con figuras
a las que les falta la mitad.

usage: python shapes_demo.py [--noise 0.25] [--seed 3]
Genera results/figuras_reconocimiento.png, results/figuras_hebb_vs_pinv.png y results/figuras_energia.png.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hopfield import HopfieldNetwork, add_noise, occlude, recognize, shape_patterns  # noqa: E402
from hopfield.shapes import SIZE  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def label(result):
    if result.name is None:
        return f"espurio (m={result.overlap:.2f})"
    return f"{result.name}{' (invertido)' if result.inverted else ''}"


def recognition_figure(names, patterns, rows, path):
    """rows: lista de (título, entradas, resultados)."""
    fig, axes = plt.subplots(len(rows) * 2, len(names), figsize=(2.3 * len(names), 2.5 * len(rows) * 2))
    for r, (title, inputs, results) in enumerate(rows):
        for col in range(len(names)):
            top, bottom = axes[2 * r, col], axes[2 * r + 1, col]
            top.imshow(inputs[col].reshape(SIZE, SIZE), cmap="binary", vmin=-1, vmax=1)
            bottom.imshow(results[col].state.reshape(SIZE, SIZE), cmap="binary", vmin=-1, vmax=1)
            ok = results[col].name == names[col]
            bottom.set_title(label(results[col]), fontsize=9, color="green" if ok else "red")
            for ax in (top, bottom):
                ax.set_xticks([]), ax.set_yticks([])
            if col == 0:
                top.set_ylabel(title, fontsize=9)
                bottom.set_ylabel("Reconocido", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def sweep(net, names, patterns, rng, trials=60):
    levels = np.linspace(0, 0.45, 10)
    rates = []
    for level in levels:
        ok = 0
        for _ in range(trials):
            k = rng.integers(len(patterns))
            ok += recognize(net, names, patterns, add_noise(patterns[k], level, rng), rng=rng).name == names[k]
        rates.append(ok / trials)
    return levels, np.array(rates)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(OUT, exist_ok=True)

    names, patterns = shape_patterns()
    n = patterns.shape[1]
    overlaps = patterns @ patterns.T / n
    print(f"{len(names)} figuras de {SIZE}x{SIZE} = {n} neuronas. Traslape entre figuras (s·p/N):")
    print("            " + " ".join(f"{x[:6]:>7}" for x in names))
    for name, row in zip(names, overlaps):
        print(f"{name:>11} " + " ".join(f"{v:7.2f}" for v in row))

    hebb = HopfieldNetwork(n).train(patterns, rule="hebb")
    pinv = HopfieldNetwork(n).train(patterns, rule="pseudoinverse")
    print("\n¿Cada figura es un punto fijo de la red?")
    for title, net in (("Hebb", hebb), ("Pseudoinversa", pinv)):
        fixed = [recognize(net, names, patterns, p, rng=rng) for p in patterns]
        print(f"  {title:>13}: " + ", ".join(f"{nm} -> {label(r)}" for nm, r in zip(names, fixed)))

    noisy = [add_noise(p, args.noise, rng) for p in patterns]
    halves = [occlude(p, side) for p, side in zip(patterns, ("right", "left", "top", "bottom", "right"))]
    rows = []
    for title, inputs in ((f"Ruido {args.noise:.0%}", noisy), ("Media figura", halves)):
        results = [recognize(pinv, names, patterns, x, rng=rng) for x in inputs]
        rows.append((title, inputs, results))
        print(f"\n{title} (pseudoinversa):")
        for name, r in zip(names, results):
            print(f"  {name:>10} -> {label(r):<22} traslape {r.overlap:.2f}, {r.sweeps} barridos")
    recognition_figure(names, patterns, rows, os.path.join(OUT, "figuras_reconocimiento.png"))

    fig, ax = plt.subplots(figsize=(6.4, 4))
    print("\nRuido vs. reconocimiento correcto (60 pruebas por nivel):")
    curves = {t: sweep(net, names, patterns, rng) for t, net in (("Hebb", hebb), ("Pseudoinversa", pinv))}
    for title, (levels, rates) in curves.items():
        ax.plot(levels * 100, rates * 100, marker="o", label=title)
    for i, level in enumerate(curves["Hebb"][0]):
        print(f"  {level * 100:4.0f} %   Hebb {curves['Hebb'][1][i] * 100:5.1f} %   "
              f"Pseudoinversa {curves['Pseudoinversa'][1][i] * 100:5.1f} %")
    ax.set_xlabel("Pixeles invertidos (%)")
    ax.set_ylabel("Figura reconocida (%)")
    ax.set_title("Regla de Hebb vs. regla de proyección")
    ax.grid(alpha=0.3), ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figuras_hebb_vs_pinv.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, x in zip(names, noisy):
        _, energies = pinv.recall(x, rng=rng)
        ax.plot(range(len(energies)), energies, marker="o", label=name)
    ax.set_xlabel("Iteración (barrido asíncrono)")
    ax.set_ylabel("Energía E(s)")
    ax.set_title("Energía durante el reconocimiento")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figuras_energia.png"), dpi=150)
    plt.close(fig)
    print(f"\nFiguras en {OUT}/")


if __name__ == "__main__":
    main()
