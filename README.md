# 🧬 Evolutionary Algorithm for the Travelling-Salesperson Problem (EA-TSP)

This project implements a Numba-accelerated evolutionary algorithm that finds near-optimal tours for symmetric TSP instances of up to **1 000 cities** within a strict 5-minute wall-clock budget.  The design blends classic GA operators with lightweight local search and diversity control to balance exploration and speed.

---

## ✨ Key Features

| Component | Key idea | Notes |
|------------|----------|-------|
| **Representation** | Permutation of city indices, last index duplicates the first | Fast to manipulate and naturally feasible |
| **Hybrid initialisation** | 80 % Nearest-Neighbour tours + 2-Opt; 20 % random permutations | Combines strong seeds with diversity |
| **Selection** | Fitness-sharing penalty → quadratic ranking | Prevents early convergence while favouring good tours  |
| **Crossover** | Partially-Mapped Crossover (PMX) | Chosen over Edge CX for efficiency |
| **Mutation** | Inversion + post-mutation 2-Opt sweep | Low-randomness tweak plus local refinement |
| **Elimination** | 3-tournament with elitism | Keeps the best tour and maintains pressure |
| **Diversity boost** | Fitness sharing (σ = 15) + population restart every 200 gens | Sustains exploration on large graphs |
| **Acceleration** | Numba‐JIT on objective, 2-Opt, and loops | Allows 600-member populations without GPU |

---

## 🔧 Default parameters

| Parameter | Value | Rationale |
|-----------|------:|-----------|
| Population size | **600** | Good quality / runtime trade-off |
| Offspring per gen | 1 200 | Two parents per individual |
| Mutation probability | 0.10 | Balances diversity and speed |
| σ-share | 15 | Empirically keeps niche spread |
| Tournament *k* | 3 | Moderate selection pressure |
| Max generations | 1 000 or 5 min | Meets run-time cap |

---


## 📊 Benchmarks (Apple M1 Pro, 5-min limit)

| Dataset | Cities | Best tour length | Generations completed |
|---------|------:|-----------------:|----------------------:|
| `tour50.csv`   | 50   | **26 502** | 1 000 |
| `tour100.csv`  | 100  | **78 281** | 1 000 |
| `tour500.csv`  | 500  | **152 325** |   ~800 |
| `tour1000.csv` | 1 000| **194 739** |    92 |

Early iterations cut tour length sharply; afterwards diversity mechanisms slow stagnation but cannot fully prevent it on the largest graph .

---

## 🧠 Future improvements
- Adaptive mutation and σ-share schedules

- Edge-preserving crossover with tabu repair

- CUDA kernel for > 10 000-city instance

---

## 📁 Project layout

```text
ea-tsp/
├── data/                     # Distance-matrix CSVs (50 – 1 000 cities)
├── src/                      # Source code
│   ├── main.py           # Main EA implementation
│   └── Reporter.py           # Logging helper (course-supplied)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick-start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run on a 100-city instance
python src/r1024617.py data/tour100.csv
```

---

© License
Released under the MIT License – see LICENSE for details.
