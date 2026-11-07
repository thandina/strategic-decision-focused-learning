# Strategic Decision Focused Learning

Code for *Strategic Decision Focused Learning* (Handina, Diao, Wierman, Mazumdar; NeurIPS 2026).
Paper: https://arxiv.org/abs/2609.14907

## Layout

| Folder | Contents |
|---|---|
| `conservation_game/` | Conservation Security Game (Etosha): Table 1, Figure 2(a) |
| `infrastructure_game/` | Infrastructure Security Game (NYC subway): Table 1, Figure 2(b) |
| `synthetic_game/` | Synthetic grid game (Appendix B.1) |
| `figures/` | Scripts for the figures |
| `data/` | Data sources and build scripts (no data is included) |
| `results/paper/` | Result files behind the paper's tables and figures |

Each folder has its own README with commands and a file map.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Not included. `data/README.md` gives the sources and the scripts that build
`data/data_ETOSHA_CELL_9.csv` and `data/data_MTA_8H_Manhattan.csv`.

## Table 1

```bash
cd conservation_game   && python train.py              # 4 methods, 5 seeds, 500 epochs
cd infrastructure_game && bash launch_all_seeds.sh     # 4 methods, 5 seeds, 500 epochs
```

| Method | Conservation utility | Conservation MAE | Infrastructure utility | Infrastructure MAE |
|---|---|---|---|---|
| PTO | -0.831 ± 0.005 | 0.110 ± 0.005 | -13.611 ± 0.210 | 0.409 ± 0.009 |
| ZOC | -0.785 ± 0.009 | 0.118 ± 0.006 | -13.160 ± 0.355 | 0.559 ± 0.062 |
| KKT | -0.404 ± 0.483 | 0.433 ± 0.159 | -12.829 ± 0.465 | 1.104 ± 0.467 |
| Hybrid | -0.173 ± 0.013 | 0.458 ± 0.107 | -12.578 ± 0.383 | 0.736 ± 0.028 |

Mean ± standard deviation over 5 seeds. `results/paper/` holds the per-seed files.

## Figures

```bash
cd figures
python plot_utility_over_time_gsg.py     # Figure 2(a)
python plot_utility_over_time_mta.py     # Figure 2(b)
python plot_instance_panels.py           # Figure 3(a)
python plot_strategies_mta.py            # Figure 3(b), 3(c)
cd ../synthetic_game && python plot_regime_panels.py   # Appendix B.1
```

## Citation

```bibtex
@article{handina2026strategic,
  title   = {Strategic Decision Focused Learning},
  author  = {Handina, Tinashe and Diao, Yuehan and Wierman, Adam and Mazumdar, Eric},
  journal = {arXiv preprint arXiv:2609.14907},
  year    = {2026}
}
```
