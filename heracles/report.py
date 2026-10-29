"""HTML run report and diagnostic plots for a fitted ``Heracles`` model.

Users normally call :meth:`heracles.Heracles.write_report`, which runs
:func:`save_plots` and :func:`generate_report`.
"""

import logging
import os

import numpy as np

logger = logging.getLogger("heracles")


__all__ = [
    'generate_report',
    'loss_curve_figure',
    'save_plots',
]


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
    model._check_fitted()
    if path is None:
        path = os.path.join(model.outdir, "report.html")
    from heracles import __version__
    import base64
    from io import BytesIO
    from datetime import datetime

    diag = model.diagnostics_ or {}
    tel = diag.get("telemetry") or {}

    def _esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;")

    def _loss_curve_b64():
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None          # report renders without the plot
        result = loss_curve_figure(model)
        if result is None:
            return None
        fig, _ax = result
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    # parameter table: one row per config field, grouped by section
    param_rows = []
    rows = [("Run", "edit model", model.edit_model.describe()),
            ("Run", "readouts", model.readouts)]
    for f in model.config.option_fields():
        v = getattr(model.config, f.name)
        if isinstance(v, np.ndarray):
            v = "array(%s)" % "x".join(str(s) for s in v.shape)
        elif f.name == "dimension" and v is None:
            v = "auto, ceil(5*sqrt(n))"
        rows.append((f.metadata["section"], f.name, v))
    last = None
    for section, name, value in rows:
        param_rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (_esc(section if section != last else ""), _esc(name), _esc(value)))
        last = section

    # fitted values
    fitted_rows = []
    for name, value in [
        ("mutation_rate (%s)" % ("fitted" if model.config.learn_mutation_rate else "fixed"),
         "%.6g" % model.mutation_rate_ if model.mutation_rate_ is not None else "n/a"),
        ("dimension", model.dimension_ if model.dimension_ is not None else "n/a"),
        ("heritable_eps (fitted)", "%.6g" % model.heritable_eps_ if model.heritable_eps_ is not None else "n/a"),
        ("stochastic_delta (fitted)", "%.6g" % model.stochastic_delta_ if model.stochastic_delta_ is not None else "n/a"),
        ("stochastic_delta per site (min..max)",
         "%.6g..%.6g (S=%d)" % (min(model.stochastic_delta_sites_), max(model.stochastic_delta_sites_),
                                len(model.stochastic_delta_sites_))
         if model.stochastic_delta_sites_ else "n/a"),
        ("stochastic_delta per cassette",
         ", ".join("%.4g" % v for v in model.stochastic_delta_cassettes_)
         if getattr(model, "stochastic_delta_cassettes_", None) else "n/a"),
        ("n_iter", model.n_iter_),
        ("converged", model.converged_),
        ("final loss", "%.6g" % model.history_["loss"][-1] if (model.history_ or {}).get("loss") else "n/a"),
    ]:
        fitted_rows.append(
            "<tr><td></td><td>%s</td><td>%s</td></tr>"
            % (_esc(name), _esc(value)))
    if diag.get("stage_transition_step") is not None:
        for name, value in [
            ("stage1_iters", diag.get("stage1_iters")),
            ("stage1_converged", diag.get("stage1_converged")),
            ("stage_transition_step", diag.get("stage_transition_step")),
            ("stage2_iters", diag.get("stage2_iters")),
        ]:
            fitted_rows.append(
                "<tr><td></td><td>%s</td><td>%s</td></tr>"
                % (_esc(name), _esc(value)))
    if model.learned_allele_prior_ is not None:
        shape = model.learned_allele_prior_.shape
        if len(shape) == 1:
            desc = "(%d states) shared; saved in model.npz" % shape[0]
        else:
            desc = "(%d sites, %d states); saved in model.npz" % (shape[0], shape[1])
        fitted_rows.append(
            "<tr><td></td><td>learned_allele_prior</td><td>%s</td></tr>" % desc)

    # warnings
    warn_list = tel.get("warnings", [])
    if warn_list:
        warn_html = "<ul>%s</ul>" % "".join(
            "<li>%s</li>" % _esc(w) for w in warn_list)
    else:
        warn_html = "<p>None.</p>"

    # loss curve
    img_b64 = _loss_curve_b64()
    if img_b64:
        loss_html = '<img src="data:image/png;base64,%s" style="max-width:100%%;">' % img_b64
    else:
        loss_html = "<p><em>matplotlib not installed; loss curve omitted.</em></p>"

    elapsed = tel.get("elapsed_s")
    elapsed_str = "%.1f s" % elapsed if elapsed is not None else "n/a"
    peak_str = "%.0f MB" % tel.get("peak_rss_mb") if tel.get("peak_rss_mb") else "n/a"

    def _fmt_time(t):
        return "%.1f s" % t if t is not None else "n/a"

    init_time_str = _fmt_time(diag.get("init_time"))
    optimizer_time_str = _fmt_time(diag.get("optimizer_time"))
    readout_time_rows = ""
    if diag.get("readout_times"):
        for rname, rt in diag["readout_times"].items():
            readout_time_rows += (
                "<tr><td>&emsp;Readout (%s)</td><td>%s</td></tr>\n"
                % (_esc(rname), _esc(_fmt_time(rt)))
            )

    html = """\
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

<h2>Replicability</h2>
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
""".format(
        timestamp=_esc(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        version=_esc(__version__),
        elapsed=_esc(elapsed_str),
        init_time=_esc(init_time_str),
        optimizer_time=_esc(optimizer_time_str),
        readout_time_rows=readout_time_rows,
        peak_rss=_esc(peak_str),
        n_cells=_esc(tel.get("n_cells", "?")),
        n_sites=_esc(tel.get("n_sites", "?")),
        platform=_esc(tel.get("platform", "unknown")),
        processor=_esc(tel.get("processor", "unknown")),
        cpu_count=_esc(tel.get("cpu_count", "?")),
        torch_device=_esc(tel.get("torch_device", "unknown")),
        python_version=_esc(tel.get("python_version", "unknown")),
        torch_version=_esc(tel.get("torch_version", "unknown")),
        geoopt_version=_esc(tel.get("geoopt_version", "unknown")),
        param_rows="\n".join(param_rows),
        fitted_rows="\n".join(fitted_rows),
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

    # 3. Ultrametric weight + depth CV (dual y-axis)
    up_w = hist.get("ultrametric_weight")
    cv = hist.get("depth_cv")
    if up_w and cv:
        fig, ax1 = plt.subplots(figsize=(6, 2.8))
        ax1.plot(up_w, lw=1, color="#2D8E5B", label="UP weight")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Ultrametric weight", color="#2D8E5B")
        ax1.tick_params(axis="y", labelcolor="#2D8E5B")
        ax2 = ax1.twinx()
        ax2.plot(cv, lw=1, color="#8E2D5B", alpha=0.7, label="Depth CV")
        ax2.set_ylabel("Depth CV", color="#8E2D5B")
        ax2.tick_params(axis="y", labelcolor="#8E2D5B")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax1.set_title("Ultrametric penalty")
        fig.tight_layout()
        p = os.path.join(model.outdir, "ultrametric.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # 4. Treelike weight + Gromov delta (dual y-axis)
    tp_w = hist.get("treelike_weight")
    gd = hist.get("gromov_delta")
    if tp_w and gd:
        fig, ax1 = plt.subplots(figsize=(6, 2.8))
        ax1.plot(tp_w, lw=1, color="#8E2D2D", label="TP weight")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Treelike weight", color="#8E2D2D")
        ax1.tick_params(axis="y", labelcolor="#8E2D2D")
        ax2 = ax1.twinx()
        ax2.plot(gd, lw=1, color="#2D5B8E", alpha=0.7,
                 label="Gromov delta")
        ax2.set_ylabel("Gromov delta", color="#2D5B8E")
        ax2.tick_params(axis="y", labelcolor="#2D5B8E")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax1.set_title("Treelike penalty")
        fig.tight_layout()
        p = os.path.join(model.outdir, "treelike.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    if saved:
        logger.info("saved plots: %s", ", ".join(saved))
