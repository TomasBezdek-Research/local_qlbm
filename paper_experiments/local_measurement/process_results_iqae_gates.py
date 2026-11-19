import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

figsize = (7, 4)

out_folder = "scaling_plots"


def setup_publication_style():
    """Configure matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "font.size": 15,
            "axes.labelsize": 16,
            "axes.titlesize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 15,
            "figure.titlesize": 12,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "axes.edgecolor": "#333333",
        }
    )
    sns.set_theme(style="whitegrid", palette="deep")


def main():
    parser = argparse.ArgumentParser(
        description="Process IQAE data and generate publication-ready plots."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the input CSV file containing the experiment records.",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)

    # Apply publication style
    setup_publication_style()
    sns.set_theme(style="whitegrid")

    # Compute shared color and marker mappings for consistency across plots
    oracles = df["Oracle"].unique()
    colors = sns.color_palette("colorblind", len(oracles))
    oracle_colors = {oracle: colors[i] for i, oracle in enumerate(oracles)}

    marker_pool = ["o", "s", "^", "D", "v", "<", ">", "P", "X"]
    oracle_markers = {
        oracle: marker_pool[i % len(marker_pool)] for i, oracle in enumerate(oracles)
    }

    # ==========================================
    # Plot 1: CNOT Ratio vs Session Index
    # ==========================================
    df["cnot_ratio"] = df["cnot_total"] / df["gate_count"]

    fig1, ax1 = plt.subplots(figsize=figsize)

    for oracle in oracles:
        sub_df = df[df["Oracle"] == oracle].copy()
        if sub_df.empty:
            continue
        ax1.plot(
            sub_df["Session_Index"],
            sub_df["cnot_ratio"],
            color=oracle_colors[oracle],
            marker=oracle_markers[oracle],
            linestyle="-",
            linewidth=1.5,
            markersize=6,
            label=oracle,
        )

    ax1.set_xlabel("IQAE Inner Iteration (Session Index)")
    ax1.set_ylabel(
        r"CNOT Ratio ($\text{CNOT}_{\text{total}} / \text{Gates}_{\text{total}}$)"
    )
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        ncol=1,
    )
    plt.tight_layout()

    pdf_ratio_name = f"{out_folder}/iqae_cnot_ratio.pdf"
    fig1.savefig(pdf_ratio_name, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved publication-ready CNOT ratio plot to {pdf_ratio_name}")
    plt.close(fig1)

    # ==========================================
    # Plot 2: Estimated Execution Time vs Session Index
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=figsize)
    gate_times = [20.0, 40.0]
    gate_styles = {20.0: "-", 40.0: "--"}

    for oracle in oracles:
        for gt in gate_times:
            sub_df = df[df["Oracle"] == oracle].copy()
            if sub_df.empty:
                continue
            sub_df["execution_time"] = sub_df["depth"] * gt

            color = oracle_colors[oracle]
            marker = oracle_markers[oracle]
            linestyle = gate_styles[gt]
            label = f"{oracle} ({int(gt)} ns)"

            ax2.plot(
                sub_df["Session_Index"],
                sub_df["execution_time"] / 1000,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=6,
                label=label,
            )

    ax2.axhline(
        y=872,
        color="blue",
        linestyle="--",
        linewidth=1.4,
        label=r"Coherence Limit (872 $\mu$s)",
    )
    ax2.axhline(
        y=967,
        color="red",
        linestyle="-.",
        linewidth=1.4,
        label=r"Coherence Limit (967 $\mu$s)",
    )

    ax2.set_xlabel("IQAE Inner Iteration (Session Index)")
    ax2.set_ylabel(
        r"Estimated Run Time ($\text{Depth} \times \text{ET}(\text{Quantum Gate})$ [$\mu$s])"
    )
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.legend(
        loc="upper left",
        # bbox_to_anchor=(1.02, 1),
        frameon=True,
        ncol=2,
    )
    plt.tight_layout()

    pdf_time_name = f"{out_folder}/iqae_execution_time.pdf"
    fig2.savefig(pdf_time_name, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved publication-ready execution time plot to {pdf_time_name}")
    plt.close(fig2)

    # ==========================================
    # Plot 3: CNOT Count and Circuit Depth vs Session Index
    # ==========================================
    fig3, ax3 = plt.subplots(figsize=figsize)
    metric_styles = {"cnot_total": "-", "depth": "--"}

    for oracle in oracles:
        sub_df = df[df["Oracle"] == oracle].copy()
        if sub_df.empty:
            continue
        color = oracle_colors[oracle]
        marker = oracle_markers[oracle]

        # Plot Total CNOT Count (Solid line)
        ax3.plot(
            sub_df["Session_Index"],
            sub_df["cnot_total"],
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.5,
            markersize=6,
            label=f"{oracle} (CNOT Count)",
        )

        # Plot Circuit Depth (Dashed line with same oracle color/marker)
        ax3.plot(
            sub_df["Session_Index"],
            sub_df["depth"],
            color=color,
            marker=marker,
            linestyle="--",
            linewidth=1.5,
            markersize=6,
            label=f"{oracle} (Depth)",
        )

    ax3.set_xlabel("IQAE Inner Iteration (Session Index)")
    ax3.set_ylabel(r"Count / Circuit Depth")
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        ncol=1,
    )
    plt.tight_layout()

    pdf_comparison_name = f"{out_folder}/iqae_gate_comparison.pdf"
    fig3.savefig(pdf_comparison_name, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved publication-ready gate comparison plot to {pdf_comparison_name}")
    plt.close(fig3)

    print("\nAll publication-ready plots successfully generated and saved as PDFs!")


if __name__ == "__main__":
    main()
