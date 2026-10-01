#!/usr/bin/env python3
"""Entrena la red con las letras A, T, X y L, y recupera cada una a partir de una versión con 20 % de ruido.

usage: python demo.py [--noise 0.2] [--seed 7]
Genera results/recuperacion.png, results/energia.png y results/ruido_vs_exito.png.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hopfield import HopfieldNetwork, add_noise, letter_patterns, to_text  # noqa: E402
from hopfield.patterns import SIZE  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def recovery_figure(names, patterns, noisy, recovered):
    fig, axes = plt.subplots(3, len(names), figsize=(2.2 * len(names), 6.6))
    for col, name in enumerate(names):
        for row, (img, label) in enumerate(((patterns[col], "Original"), (noisy[col], "Con ruido"),
                                            (recovered[col], "Recuperado"))):
            ax = axes[row, col]
            ax.imshow(img.reshape(SIZE, SIZE), cmap="binary", vmin=-1, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "recuperacion.png"), dpi=150)


def energy_figure(names, histories):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, h in zip(names, histories):
        ax.plot(range(len(h)), h, marker="o", label=name)
    ax.set_xlabel("Iteración (barrido asíncrono)")
    ax.set_ylabel("Energía E(s)")
    ax.set_title("La energía nunca aumenta durante la recuperación")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "energia.png"), dpi=150)


def noise_sweep(net, patterns, rng, trials=50):
    levels = np.linspace(0, 0.5, 11)
    success = []
    for level in levels:
        ok = 0
        for _ in range(trials):
            k = rng.integers(len(patterns))
            out, _ = net.recall(add_noise(patterns[k], level, rng), rng=rng)
            ok += np.array_equal(out, patterns[k])
        success.append(ok / trials)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(levels * 100, np.array(success) * 100, marker="o")
    ax.set_xlabel("Bits invertidos (%)")
    ax.set_ylabel("Recuperación exacta (%)")
    ax.set_title(f"Robustez al ruido ({trials} pruebas por nivel)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ruido_vs_exito.png"), dpi=150)
    return levels, success


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(OUT, exist_ok=True)

    names, patterns = letter_patterns()
    net = HopfieldNetwork(patterns.shape[1]).train(patterns)
    print(f"Neuronas: {net.n}  Patrones: {net.n_patterns}  Capacidad teórica ~{net.capacity():.1f}\n")

    noisy, recovered, histories = [], [], []
    for name, p in zip(names, patterns):
        x = add_noise(p, args.noise, rng)
        out, energies = net.recall(x, rng=rng)
        noisy.append(x), recovered.append(out), histories.append(energies)
        print(f"Letra {name}: {int(np.sum(x != p))} bits invertidos -> "
              f"{'recuperada' if np.array_equal(out, p) else 'NO recuperada'} "
              f"en {len(energies) - 1} barridos, energía {energies[0]:.2f} -> {energies[-1]:.2f}")
        for a, b, c in zip(to_text(x).split("\n"), to_text(out).split("\n"), to_text(p).split("\n")):
            print(f"  {a}   ->   {b}   (original {c})")
        print()

    recovery_figure(names, patterns, noisy, recovered)
    energy_figure(names, histories)
    levels, success = noise_sweep(net, patterns, rng)
    print("Ruido vs. recuperación exacta:")
    for level, s in zip(levels, success):
        print(f"  {level * 100:4.0f} %  ->  {s * 100:5.1f} %")
    print(f"\nFiguras en {OUT}/")


if __name__ == "__main__":
    main()
