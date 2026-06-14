import matplotlib as mpl
import seaborn as sns

def apply_plot_params(
    fig_height: float,
    scale: float = 1.0,
    double_column: bool = False,
) -> None:
    sns.set_theme(
            context="paper",
            style="whitegrid",
            font_scale=1.0,
            rc={
                "axes.facecolor": "white",
                "figure.facecolor": "white",
                "grid.color": "#d0d0d0",
                "grid.linewidth": 0.8,
                "axes.edgecolor": "#666666",
                "axes.linewidth": 0.8,
                "axes.titleweight": "bold",
                "axes.labelcolor": "#222222",
                "xtick.color": "#222222",
                "ytick.color": "#222222",
            },
        )
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
