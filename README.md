# Evolutionary Algorithm for the Travelling Salesman Problem (EA-TSP)

An **evolutionary algorithm (EA)** that tackles symmetric Travelling-Salesman instances up to **1 000 cities** in under five minutes.  
---

## ✨ Key Features

| Component | Implementation |
|-----------|----------------|
| **Representation** | Permutation encoding (cycle closed by repeating start city) |
| **Initialisation** | Hybrid: 80 % Nearest-Neighbour tours + 20 % random, each refined by *2-Opt* |
| **Selection** | Fitness-sharing ➜ quadratic ranking |
| **Crossover** | Partially-Mapped Crossover (*PMX*) with feasibility repair |
| **Mutation** | Inversion + *2-Opt* local search |
| **Elimination** | k-tournament with elitism |
| **Speed-ups** | Critical loops `@njit`-compiled with **Numba** |

---

## 📁 Project layout

```text
ea-tsp/
├── data/                     # Distance-matrix CSVs (50 – 1 000 cities)
├── src/                      # Source code
│   ├── r1024617.py           # Main EA implementation
│   └── Reporter.py           # Logging helper (course-supplied)
├── docs/
│   ├── EA-TSP_Report.pdf     # Final report
│   └── EA-TSP_Intermediate.pdf  # Intermediate report
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

