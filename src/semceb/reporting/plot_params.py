import matplotlib as mpl

def apply_plot_params(
    fig_height: float,
    scale: float = 1.0,
    double_column: bool = False,
) -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (
                (7.00697 if double_column else 3.3374) * scale,
                fig_height * scale,
            ),
            "figure.dpi": 300,
            "font.size": 11.0,
            "font.family": "serif",
            "axes.titlesize": "medium",
            "axes.labelsize": "medium",
            "figure.titlesize": "medium",
            "xtick.labelsize": "medium",
            "ytick.labelsize": "medium",
            "legend.fontsize": "medium",
            "legend.title_fontsize": "medium",
            "text.usetex": True,
            "text.latex.preamble": (
                r"\usepackage{amsmath}\usepackage{amssymb}"
                r"\usepackage{siunitx}[=v2]"
            ),
            "pgf.rcfonts": False,
            "pgf.texsystem": "pdflatex",
        }
    )
