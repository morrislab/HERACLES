# Tutorial data

Four simulated lineage-tracing datasets of 250 cells each.

| Folder | Recorder | Sites | Missing entries | Source |
|---|---|---|---|---|
| `cas9_mixed_dropout/` | Cas9 | 30 | 22%, heritable and stochastic | LAML, condition s50d50 |
| `cas9_stochastic_dropout/` | Cas9 | 30 | 26%, stochastic only | LAML, condition s0d100 |
| `cas9_heritable_dropout/` | Cas9 | 30 | 26%, heritable only | LAML, condition s100d0 |
| `typewriter/` | DNA Typewriter, 12 cassettes of 5 sites | 60 | 12% (heritable rate 0.01, stochastic rate 0.05) | Cassiopeia 2.1.0 |

The three Cas9 datasets are replicate 0 of the simulations from the LAML
paper (Chu et al., *Genome Biology* 2025, doi:10.1186/s13059-025-03649-9):
1024-leaf trees subsampled to 250 cells. The `typewriter` dataset was
simulated with Cassiopeia's sequential lineage-tracing simulator
(initiation rate 1.0, continuation rate 0.5, 30 states, seed 0).

Each folder holds:

- `characters.csv`: one row per cell, one column per site. The first column is
  the cell name. 0 = unedited, a positive integer = an edit, -1 = missing.
  In `typewriter`, column `c3_2` is the second site of cassette 3.
- `true_tree.nwk`: the simulated tree with its branch lengths.

The simulators number cells in tree order, which can leak the answer to
methods that break ties by row order. The rows here are shuffled and the
cells renamed `cell001` to `cell250` in the shuffled order.
