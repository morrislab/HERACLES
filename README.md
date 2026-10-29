# HERACLES

**H**yperbolic **E**mbedding for **R**econstructing **A**ccurate **C**ell **L**ineag**ES**

HERACLES reconstructs cell lineage trees from CRISPR lineage-tracing data.
It places every cell as a point in hyperbolic space and moves the points so
that their distances explain the observed edits under an explicit model of
CRISPR editing. The objective is a composite likelihood over quartets of
cells, so it never has to build or score a whole tree during the fit. A tree
is read off the final distances with neighbour joining and can then be
polished with likelihood-scored NNI moves.

Because the fit is a continuous optimisation, HERACLES scales to very large
lineages: in the paper it reconstructs 150,000 cells in about six hours on a
single GPU. The editing model is replaceable, so recorders with cassette-level
dropout (PETracer) or ordered editing (DNA Typewriter) use the same
framework.

## Installation

HERACLES needs Python 3.10 or newer and a C++ compiler, which builds the
bundled [RapidNJ](https://github.com/johnlees/rapidnj) during installation.
The conda route below installs the compiler for you.

With conda (or mamba):

```bash
conda create -n heracles -c conda-forge python=3.11 cxx-compiler
conda activate heracles
pip install git+https://github.com/morrislab/HERACLES.git
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 heracles-env
source heracles-env/bin/activate
uv pip install git+https://github.com/morrislab/HERACLES.git
```

Either route takes one to five minutes, mostly the PyTorch download. For GPU
fits, install the PyTorch build for your CUDA version
([pytorch.org](https://pytorch.org/get-started/locally/)) *before* the
`pip install` line; otherwise pip picks the default build.

### Check the installation

```bash
heracles fit --help
python -c "from heracles.trees.rapidnj import rapidnj_executable; print(rapidnj_executable())"
```

The first prints the options of the `fit` command; the second prints the path
of the RapidNJ binary. If either fails, see below.

### Common problems

- **`heracles: command not found`.** The environment is not activated
  (`conda activate heracles` or `source heracles-env/bin/activate`), or pip
  installed into another Python. `python -m heracles fit --help` works from
  any Python that has the package.
- **`FileNotFoundError: no RapidNJ binary`.** No C++ compiler was found when
  the package was installed. Install one, then reinstall without pip's cache
  (a cached wheel would skip the build):
  - conda: `conda install -c conda-forge cxx-compiler`
  - Debian/Ubuntu: `sudo apt install build-essential`
  - Fedora/RHEL: `sudo dnf install gcc-c++`
  - macOS: `xcode-select --install`
  - then `pip install --no-cache-dir --force-reinstall --no-deps git+https://github.com/morrislab/HERACLES.git`
    (uv: `uv pip install --no-cache --reinstall --no-deps git+https://github.com/morrislab/HERACLES.git`)

  Alternatively, HERACLES builds the binary on first use into
  `~/.cache/heracles/` once a compiler is on `PATH`, or uses your own build
  if you set `HERACLES_RAPIDNJ=/path/to/rapidnj`. Without any binary the fit
  still runs: it starts from an MDS embedding instead of the RapidNJ tree and
  reads the final tree out with a slower pure-Python neighbour joining, and
  the log says so.
- **The PyTorch download is several GB or fails.** The default Linux `torch`
  wheel includes CUDA libraries. On a CPU-only machine install the CPU wheel
  first, then HERACLES:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- **Fits are slow on a many-core machine.** PyTorch uses every core and
  oversubscribes when several fits share a node. Set `OMP_NUM_THREADS` to
  the number of cores each fit should use.

## Quick start

The input is a CSV with one row per cell and one column per target site. The
first column holds the cell names. Values are integer allele ids: `0` is
unedited, a positive id is an edit, and `-1` is a missing entry.

```python
from heracles import Heracles

model = Heracles(outdir="runs/example")   # the published method
model.fit("characters.csv")               # a CSV path or a DataFrame

model.newick("nj_collapsed")              # the tree, as Newick
model.polish("characters.csv")            # optional likelihood NNI polish
model.newick("nj_polished_collapsed")
```

The same from the command line:

```bash
heracles fit characters.csv --outdir runs/example
heracles polish characters.csv --model runs/example/model.npz
heracles inspect runs/example/model.npz
```

`heracles fit --help` lists every option with its default. On a GPU, add
`device="cuda"` (or `--device cuda`). For tens of thousands of cells, add
`low_memory=True` (`--low-memory`), which avoids storing an n x n distance
matrix during the optimisation. Building the final tree is still O(n^2) in
memory: in the paper (UPGMA read-out), 100,000 cells needed 78 GB of host memory and 150,000
cells needed 173 GB.

Trees are rooted at a synthetic unedited founder cell. The `_collapsed` trees
contract every edge that carries no inferred edit (Camin-Sokal collapse);
these are the trees to compare against other methods or a known truth.

## Tutorials

The [`tutorials/`](tutorials) folder has four notebooks that run on simulated
datasets of 250 cells, each in five to twenty minutes on a laptop CPU. Clone the
repository and install the notebook dependencies into the same environment:

```bash
git clone https://github.com/morrislab/HERACLES.git
pip install -r HERACLES/requirements.txt
```

1. [Quick start](tutorials/01_quickstart.ipynb): fit, read the tree, compare
   with the true tree and with neighbour joining, save and load.
2. [Missing data](tutorials/02_missing_data.ipynb): heritable versus
   stochastic dropout, and choosing an edit model.
3. [Polishing and ancestral states](tutorials/03_polish_and_ancestral_states.ipynb):
   branch support, the likelihood NNI polish, and placing edits on branches.
4. [Sequential recorders](tutorials/04_sequential_recorders.ipynb): DNA
   Typewriter data with the Cas9 and the sequential edit models.

A `ModuleNotFoundError` inside a notebook means its kernel is not the Python
you installed HERACLES into: start Jupyter from the activated environment.

## Edit models

The edit model describes how the recorder writes and loses information. Pick
one with `edit_model=` (Python) or `--edit-model` (command line).

| Model | Recorder |
|---|---|
| `Cas9Model()` (default) | independent sites, each edited at most once; sites drop out on their own |
| `PETracerModel(cassette_size=3)` | Cas9 editing; a cassette of consecutive sites drops out as a unit |
| `SequentialModel(cassette_size=K)` | the sites of a cassette are edited in order (DNA Typewriter) |

```python
from heracles import Heracles, PETracerModel

model = Heracles(outdir="runs/pet", edit_model=PETracerModel(cassette_size=3))
```

Missing data comes in two kinds, both learned by default: heritable dropout
(`heritable_eps`, a site lost in an ancestor and in all its descendants) and
stochastic dropout (`stochastic_delta`, a site lost independently in each
cell). To support a new recorder, write an `EditProcess` and pair it with a
missing-data model through `heracles.models.CustomModel`; the template is
documented in `heracles.models.base`.

## Outputs

`fit()` writes to `outdir`:

| File | Contents |
|---|---|
| `model.npz` | everything needed to reload the model with `Heracles.load()` |
| `tree_nj.nwk` | the neighbour-joining tree, rooted at the founder |
| `tree_nj_collapsed.nwk` | the same tree with edit-free edges contracted |
| `config.json` | every setting, the edit model and the fitted rates |
| `heracles.log` | the run log |

`polish()` adds `tree_nj_polished.nwk` and its collapsed form, and
`write_report()` adds diagnostic plots and an HTML report. Fitted values are
attributes ending in `_`, for example `trees_`, `leaf_dists_` (distances
between cells in the embedding), `heritable_eps_`, `stochastic_delta_`,
`history_` (the loss and rates at every step) and `converged_`.

## Options

Every option is a field of `HeraclesConfig` with the published default and can
be passed as a keyword, for example `Heracles(outdir=..., dimension=64,
max_iterations=10000, seed=1)`. The ones you are most likely to change:

| Option | Default | Meaning |
|---|---|---|
| `dimension` | `ceil(5 * sqrt(n))` | dimension of the hyperbolic space |
| `max_iterations` | 5000 | optimisation steps |
| `heritable_eps`, `stochastic_delta` | 0.001, learned | dropout rates; pass a value with `learn_heritable_eps=False` (or `learn_stochastic_delta=False`) to fix one |
| `allele_prior` | `"per_site_uniform"` | prior over the alleles at each site |
| `readouts` | `["nj"]` | trees built from the embedding (`"nj"`, `"upgma"`) |
| `device` | `"cpu"` | `"cpu"`, `"cuda"` or `"auto"` |
| `low_memory` | `False` | no n x n matrix during the optimisation, for very large inputs |

The method internals (quartet sampler, penalties, learning rate) are in
`HeraclesConfig.advanced` and listed under "Advanced" in `heracles fit
--help`. The paper describes the method and the reasoning behind these
settings.

## Other tools in the package

- `heracles.polish.branch_support`: quartet support for every branch of a tree.
- `heracles.polish.ancestral.infer_ancestral_states`: posterior character states of
  every ancestor under the same edit model; `optimize_branch_lengths` refits
  branch lengths.
- `heracles.sokal_collapse`: Camin-Sokal collapse of any tree or Newick string.
- `heracles.robinson_foulds`: Robinson-Foulds distance between two trees or
  Newick strings (`normalized=True` divides by the number of internal edges).
- `Heracles(init_tree="start.nwk")`: start the embedding from your own tree.

## Development

```bash
git clone https://github.com/morrislab/HERACLES.git
cd HERACLES
pip install -e ".[dev]"
pytest                 # fast tests, about a minute
pytest -m slow         # command-line and RapidNJ build tests
```

## Citation

If you use HERACLES, please cite:

> Sitara Persad and Quaid Morris. HERACLES for fast and accurate
> likelihood-based reconstruction of massive lineages. 2026.

## License

GPL-2.0-or-later. The bundled RapidNJ source is GPL-2.0; see
[`heracles/_vendor/rapidnj/`](heracles/_vendor/rapidnj) for its licence, the
upstream commit and the local portability patch.
