import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

out_folder = "scaling_plots"


def main():
    parser = argparse.ArgumentParser(
        description="Process IQAE scaling data and save separate PDF plots with distinct markers for each line."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the input CSV file containing the experiment records.",
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Obligatory experiment name used as a suffix for the output PDF filenames.",
    )
    parser.add_argument(
        "--log_scale",
        type=bool,
        default=False,
        required=False,
        help="Showing both axis in logarithmic scale",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)

    # 1. Compute runtime per iteration/session (depth * shots)
    df["runtime_iteration"] = df["depth"] * df["Shots"]

    # 2. Compute runtime and number of inner iterations per configuration
    df_config = df.groupby(["Epsilon", "Delta"], as_index=False).agg(
        total_runtime=("runtime_iteration", "sum"),
        num_iterations=("runtime_iteration", "size"),
    )

    # 3. Compute theoretical scaling axes: 1/eps and ln(2/delta)
    df_config["inv_epsilon"] = 1.0 / df_config["Epsilon"]
    df_config["log_inv_delta"] = np.log(2.0 / df_config["Delta"])

    print("\nAggregated Runtimes and Iterations with Theoretical Scaling Variables:")
    print(df_config.head())

    # Set general plotting style
    sns.set_theme(style="whitegrid")

    # Fit proportional reference lines based on mean ratios
    slope_delta = np.mean(
        df_config["total_runtime"]
        / (df_config["inv_epsilon"] * df_config["log_inv_delta"])
    )
    slope_iter = np.mean(df_config["num_iterations"] / df_config["inv_epsilon"])

    # Set professional paper visual style
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 16,
            "axes.titlesize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    figsize = (6, 5)

    # ==========================================
    # Plot 1: Scaling vs log(2/delta)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=df_config,
        x="log_inv_delta",
        y="total_runtime",
        hue="Epsilon",
        style="Epsilon",  # Ensures different line styles/markers per group
        markers=True,  # Enables distinct markers (circles, squares, diamonds, etc.)
        dashes=False,  # Keep lines solid or let seaborn handle styles safely
        ax=ax1,
        palette="tab10",
    )

    # Add reference line
    x_vals_delta = np.linspace(
        df_config["log_inv_delta"].min(), df_config["log_inv_delta"].max(), 100
    )
    med_inv_eps = df_config["inv_epsilon"].median()

    c_delta = slope_delta * med_inv_eps
    exp_delta = int(np.floor(np.log10(abs(c_delta))))
    mant_delta = c_delta / 10**exp_delta

    ax1.plot(
        x_vals_delta,
        slope_delta * med_inv_eps * x_vals_delta,
        color="blue",
        linestyle="--",
        label=(
            rf"$\mathcal{{O}}\left("
            rf"{mant_delta:.1f}\times10^{{{exp_delta}}}"
            rf"\,\ln(2/\delta)"
            rf"\right)$"
        ),
    )
    if args.log_scale:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    ax1.set_xlabel(r"$\ln(2/\delta)$")
    ax1.set_ylabel(r"Total Runtime (Depth $\times$ Shots)")
    ax1.legend(title=r"$\epsilon$", loc="upper left", ncols=2)
    plt.tight_layout()

    pdf_delta_name = f"{out_folder}/iqae_scaling_delta_{args.experiment_name}.pdf"
    fig1.savefig(pdf_delta_name, format="pdf", bbox_inches="tight")
    print(f"Saved Delta scaling plot to {pdf_delta_name}")
    plt.close(fig1)

    # ==========================================
    # Plot 2: Scaling vs 1/epsilon (Total Runtime)
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=df_config,
        x="inv_epsilon",
        y="total_runtime",
        hue="Delta",
        style="Delta",  # Ensures different line styles/markers per group
        markers=True,  # Enables distinct markers
        dashes=False,
        ax=ax2,
        palette="tab10",
    )

    # Add reference line
    x_vals_eps = np.linspace(
        df_config["inv_epsilon"].min(), df_config["inv_epsilon"].max(), 100
    )
    med_log_delta = df_config["log_inv_delta"].median()

    c_eps = slope_delta * med_log_delta
    exp_eps = int(np.floor(np.log10(abs(c_eps))))
    mant_eps = c_eps / 10**exp_eps

    ax2.plot(
        x_vals_eps,
        slope_delta * x_vals_eps * med_log_delta,
        color="blue",
        linestyle="--",
        label=(
            rf"$\mathcal{{O}}\left("
            rf"{mant_eps:.1f}\times10^{{{exp_eps}}}"
            rf"/\epsilon"
            rf"\right)$"
        ),
    )
    if args.log_scale:
        ax2.set_xscale("log")
        ax2.set_yscale("log")
    ax2.set_xlabel(r"$1/\epsilon$")
    ax2.set_ylabel(r"Total Runtime (Depth $\times$ Shots)")
    ax2.legend(title=r"$\delta$", loc="upper left", ncols=2)
    plt.tight_layout()

    pdf_eps_name = f"{out_folder}/iqae_scaling_epsilon_{args.experiment_name}.pdf"
    fig2.savefig(pdf_eps_name, format="pdf", bbox_inches="tight")
    print(f"Saved Epsilon scaling plot to {pdf_eps_name}")
    plt.close(fig2)

    # ==========================================
    # Plot 3: Number of Iterations vs 1/epsilon
    # ==========================================
    fig3, ax3 = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=df_config,
        x="inv_epsilon",
        y="num_iterations",
        hue="Delta",
        style="Delta",
        markers=True,
        dashes=False,
        ax=ax3,
        palette="tab10",
    )

    # Add reference line
    # ax3.plot(
    #     x_vals_eps,
    #     slope_iter * x_vals_eps,
    #     color="blue",
    #     linestyle="--",
    #     label=r"Expected $\mathcal{O}(1/\epsilon)$ Trend",
    # )
    if args.log_scale:
        ax3.set_xscale("log")
        # ax3.set_yscale("log")
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.set_xlabel(r"$1/\epsilon$")
    ax3.set_ylabel(r"Number of IQAE Inner Iterations")
    ax3.legend(title=r"$\delta$", loc="upper left", ncols=2)
    plt.tight_layout()

    pdf_iter_name = f"{out_folder}/iqae_iterations_epsilon_{args.experiment_name}.pdf"
    fig3.savefig(pdf_iter_name, format="pdf", bbox_inches="tight")
    print(f"Saved Iterations vs Epsilon plot to {pdf_iter_name}")
    plt.close(fig3)

    print("\nAll plots successfully generated and saved as PDFs with distinct markers!")


if __name__ == "__main__":
    main()
