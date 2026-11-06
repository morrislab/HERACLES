"""HTML run report and diagnostic plots for a fitted ``Heracles`` model.

Users normally call :meth:`heracles.Heracles.write_report`, which runs
:func:`save_plots` and :func:`generate_report`.
"""

import base64
import logging
import os
from datetime import datetime
from io import BytesIO

import numpy as np

logger = logging.getLogger("heracles")


_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>HERACLES Run Report</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
     max-width: 820px; margin: 2em auto; padding: 0 1em;
     color: #1a1a1a; line-height: 1.5; }}
  h1 {{ border-bottom: 2px solid #5B2D8E; padding-bottom: 0.3em; }}
  h2 {{ color: #5B2D8E; margin-top: 1.5em; }}
  table {{ border-collapse: collapse; width: 100%%; margin: 0.5em 0; }}
  th, td {{ text-align: left; padding: 4px 10px; border-bottom: 1px solid #ddd; }}
  th {{ background: #f6f2fa; }}
  .box {{ background: #f6f2fa; border-left: 4px solid #5B2D8E;
      padding: 0.8em 1em; margin: 1em 0; }}
  .warn {{ background: #fff8e1; border-left: 4px solid #e6a700;
       padding: 0.8em 1em; margin: 1em 0; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }}
  .meta {{ color: #666; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>HERACLES Run Report</h1>
<p class="meta">Generated {timestamp} &middot; heracles {version}</p>

<h2>Run summary</h2>
<div class="box">
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Wall-clock time</td><td>{elapsed}</td></tr>
<tr><td>&emsp;Initialisation</td><td>{init_time}</td></tr>
<tr><td>&emsp;Optimiser</td><td>{optimizer_time}</td></tr>
{readout_time_rows}
<tr><td>Peak RSS</td><td>{peak_rss}</td></tr>
<tr><td>Input size</td><td>{n_cells} cells &times; {n_sites} sites</td></tr>
<tr><td>Platform</td><td>{platform}</td></tr>
<tr><td>Processor</td><td>{processor}</td></tr>
<tr><td>CPU cores</td><td>{cpu_count}</td></tr>
<tr><td>Torch device</td><td>{torch_device}</td></tr>
<tr><td>Python</td><td>{python_version}</td></tr>
<tr><td>torch</td><td>{torch_version}</td></tr>
<tr><td>geoopt</td><td>{geoopt_version}</td></tr>
<tr><td>heracles</td><td>{version}</td></tr>
</table>
</div>

<h2>Parameters</h2>
<table>
<tr><th>Section</th><th>Parameter</th><th>Value</th></tr>
{param_rows}
</table>

<h2>Fitted values</h2>
<table>
<tr><th></th><th>Parameter</th><th>Value</th></tr>
{fitted_rows}
</table>

<h2>Warnings</h2>
<div class="warn">
{warnings}
</div>

<h2>Loss curve</h2>
{loss_curve}

</body>
</html>
"""


__all__ = [
    "generate_report",
    "loss_curve_figure",
    "save_plots",
]


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;")


def _row(cells):
    return "<tr>%s</tr>" % "".join("<td>%s</td>" % _esc(c) for c in cells)


def _loss_curve_b64(model):
    """The loss curve as a base64 PNG, or None without matplotlib."""
    result = loss_curve_figure(model)
    if result is None:
        return None
    import matplotlib.pyplot as plt
    fig, _ax = result
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _param_rows(model):
    """One table row per setting, grouped by config section."""
    rows = [("Run", "edit model", model.edit_model.describe()),
            ("Run", "readouts", model.readouts)]
    for f in model.config.option_fields():
        v = getattr(model.config, f.name)
        if isinstance(v, np.ndarray):
            v = "array(%s)" % "x".join(str(s) for s in v.shape)
        elif f.name == "dimension" and v is None:
            v = "auto, ceil(5*sqrt(n))"
        rows.append((f.metadata["section"], f.name, v))
    out, last = [], None
    for section, name, value in rows:
        out.append(_row((section if section != last else "", name, value)))
        last = section
    return out


def _fitted_rows(model):
    """Table rows of the fitted values and the two-stage bookkeeping."""
    cfg, diag = model.config, model.diagnostics_ or {}

    def num(v):
        return "%.6g" % v if v is not None else "n/a"

    def how(learned):
        return "fitted" if learned else "fixed"

    rows = [
        ("mutation_rate (%s)" % how(cfg.learn_mutation_rate), num(model.mutation_rate_)),
        ("dimension", model.dimension_ if model.dimension_ is not None else "n/a"),
        ("heritable_eps (%s)" % how(cfg.learn_heritable_eps), num(model.heritable_eps_)),
        ("stochastic_delta (%s)" % how(cfg.learn_stochastic_delta), num(model.stochastic_delta_)),
        ("stochastic_delta per site (min..max)",
         "%.6g..%.6g (S=%d)" % (min(model.stochastic_delta_sites_), max(model.stochastic_delta_sites_),
                                len(model.stochastic_delta_sites_))
         if model.stochastic_delta_sites_ else "n/a"),
        ("stochastic_delta per cassette",
         ", ".join("%.4g" % v for v in model.stochastic_delta_cassettes_)
         if model.stochastic_delta_cassettes_ else "n/a"),
        ("n_iter", model.n_iter_),
        ("converged", model.converged_),
        ("final loss", num(model.history_["loss"][-1]) if (model.history_ or {}).get("loss") else "n/a"),
    ]
    if diag.get("stage_transition_step") is not None:
        rows += [(k, diag.get(k)) for k in
                 ("stage1_iters", "stage1_converged", "stage_transition_step", "stage2_iters")]
    if model.learned_allele_prior_ is not None:
        shape = model.learned_allele_prior_.shape
        rows.append(("learned_allele_prior",
                     "(%d states) shared; saved in model.npz" % shape[0] if len(shape) == 1
                     else "(%d sites, %d states); saved in model.npz" % shape))
    return [_row(("",) + r) for r in rows]


def generate_report(model, path=None):
    """Writes an HTML run report for a fitted model.

    The report lists hardware, timings, memory, every setting, any warnings
    from the fit, and the loss curve.

    Args:
        model: A fitted :class:`~heracles.Heracles`.
        path: Output ``.html`` file (default ``<outdir>/report.html``).

    Returns:
        The path written.
    """
    from heracles import __version__
    model._check_fitted()
    if path is None:
        path = os.path.join(model.outdir, "report.html")
    diag = model.diagnostics_ or {}
    tel = diag.get("telemetry") or {}

    def seconds(t):
        return "%.1f s" % t if t is not None else "n/a"

    img_b64 = _loss_curve_b64(model)
    loss_html = ('<img src="data:image/png;base64,%s" style="max-width:100%%;">' % img_b64
                 if img_b64 else "<p><em>matplotlib not installed; loss curve omitted.</em></p>")
    warn_list = tel.get("warnings", [])
    warn_html = ("<ul>%s</ul>" % "".join("<li>%s</li>" % _esc(w) for w in warn_list)
                 if warn_list else "<p>None.</p>")
    readout_time_rows = "\n".join(
        _row(("\u2003Readout (%s)" % rname, seconds(rt)))
        for rname, rt in (diag.get("readout_times") or {}).items())
    html = _TEMPLATE.format(
        timestamp=_esc(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        version=_esc(__version__),
        elapsed=_esc(seconds(tel.get("elapsed_s"))),
        init_time=_esc(seconds(diag.get("init_time"))),
        optimizer_time=_esc(seconds(diag.get("optimizer_time"))),
        readout_time_rows=readout_time_rows,
        peak_rss=_esc("%.0f MB" % tel["peak_rss_mb"] if tel.get("peak_rss_mb") else "n/a"),
        n_cells=_esc(tel.get("n_cells", "?")),
        n_sites=_esc(tel.get("n_sites", "?")),
        platform=_esc(tel.get("platform", "unknown")),
        processor=_esc(tel.get("processor", "unknown")),
        cpu_count=_esc(tel.get("cpu_count", "?")),
        torch_device=_esc(tel.get("torch_device", "unknown")),
        python_version=_esc(tel.get("python_version", "unknown")),
        torch_version=_esc(tel.get("torch_version", "unknown")),
        geoopt_version=_esc(tel.get("geoopt_version", "unknown")),
        param_rows="\n".join(_param_rows(model)),
        fitted_rows="\n".join(_fitted_rows(model)),
        warnings=warn_html,
        loss_curve=loss_html,
    )
    with open(path, "w") as f:
        f.write(html)
    return path


def loss_curve_figure(model):
    """Builds the loss-curve figure; returns ``(fig, ax)``, or None without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot((model.history_ or {}).get("loss", []), lw=1, color="#5B2D8E")
    step2 = (model.diagnostics_ or {}).get("stage_transition_step")
    if step2 is not None:
        ax.axvline(step2, color="orange", ls=":", lw=1.2, label="stage 2 @ %d" % step2)
    if model.converged_:
        ax.axvline(model.n_iter_, color="green", ls="--", lw=1,
                   label="converged @ %d" % model.n_iter_)
    if step2 is not None or model.converged_:
        ax.legend(fontsize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("Loss curve")
    fig.tight_layout()
    return fig, ax


def _dual_axis_plot(plt, series_a, series_b, label_a, label_b, title, colors, path):
    """Two series against the iteration, each on its own y axis, saved to *path*."""
    fig, ax1 = plt.subplots(figsize=(6, 2.8))
    ax1.plot(series_a, lw=1, color=colors[0], label=label_a)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(label_a, color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])
    ax2 = ax1.twinx()
    ax2.plot(series_b, lw=1, color=colors[1], alpha=0.7, label=label_b)
    ax2.set_ylabel(label_b, color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_plots(model):
    """Saves diagnostic plots of the fit into the model's ``outdir``.

    Writes those of ``loss_curve.png``, ``dropout_params.png``,
    ``ultrametric.png`` and ``treelike.png`` whose series are in ``history_``.
    Does nothing if matplotlib is not installed.
    """
    model._check_fitted()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    hist = model.history_ or {}
    saved = []

    # 1. Loss curve
    result = loss_curve_figure(model)
    if result is not None:
        fig, _ax = result
        p = os.path.join(model.outdir, "loss_curve.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # 2. Dropout parameters (eps / delta)
    eps = hist.get("heritable_eps")
    delta = hist.get("stochastic_delta")
    if eps or delta:
        n_panels = (1 if eps else 0) + (1 if delta else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 2.8),
                                 squeeze=False)
        idx = 0
        if eps:
            axes[0, idx].plot(eps, lw=1, color="#2D6B8E")
            axes[0, idx].set_xlabel("Iteration")
            axes[0, idx].set_ylabel("heritable_eps")
            axes[0, idx].set_title("Heritable dropout (eps)")
            idx += 1
        if delta:
            axes[0, idx].plot(delta, lw=1, color="#8E5B2D")
            axes[0, idx].set_xlabel("Iteration")
            axes[0, idx].set_ylabel("stochastic_delta")
            axes[0, idx].set_title("Stochastic dropout (delta)")
        fig.tight_layout()
        p = os.path.join(model.outdir, "dropout_params.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # 3, 4. penalty weight and the quantity it controls, on two y axes
    for key_w, key_v, name, label_w, label_v, title, colors in (
            ("ultrametric_weight", "depth_cv", "ultrametric", "UP weight", "Depth CV",
             "Ultrametric penalty", ("#2D8E5B", "#8E2D5B")),
            ("treelike_weight", "gromov_delta", "treelike", "TP weight", "Gromov delta",
             "Treelike penalty", ("#8E2D2D", "#2D5B8E"))):
        w, v = hist.get(key_w), hist.get(key_v)
        if w and v:
            p = os.path.join(model.outdir, name + ".png")
            _dual_axis_plot(plt, w, v, label_w, label_v, title, colors, p)
            saved.append(p)

    if saved:
        logger.info("saved plots: %s", ", ".join(saved))
