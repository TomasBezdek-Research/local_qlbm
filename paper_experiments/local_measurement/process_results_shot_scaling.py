import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator


def main():
    parser = argparse.ArgumentParser(
        description="Post-process shot scaling data to plot Total Runtime vs empirical 1/eps^2 and log(2/delta) with theoretical scaling lines."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the output expanded CSV file from your experiment.",
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

    parser.add_argument(
        "--ground_true_value",
        type=float,
        required=True,
        help="The known ground truth probability for error calculation.",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)

    # Verify columns
    if "Estimated_Probability" not in df.columns or "Shots_Config" not in df.columns:
        raise ValueError(
            "CSV must contain 'Estimated_Probability' and 'Shots_Config' columns."
        )

    # 1. Compute absolute error for every run/row
    df["absolute_error"] = np.abs(df["Estimated_Probability"] - args.ground_true_value)

    # 2. Compute runtime per iteration/session
    effective_shots = (
        df["Actual_Shots"].fillna(df["Shots_Config"])
        if "Actual_Shots" in df.columns
        else df["Shots_Config"]
    )
    df["runtime_iteration"] = df["depth"] * effective_shots

    shots_configs = sorted(df["Shots_Config"].unique())

    # ==========================================
    # Slice 1: Runtimes vs. Empirical 1/epsilon^2 (for fixed target Delta values)
    # ==========================================
    fixed_deltas = [0.01, 0.05, 0.10, 0.20]
    delta_vs_runtime_data = []

    for delta in fixed_deltas:
        percentile = (1.0 - delta) * 100
        for shots in shots_configs:
            subset_df = df[df["Shots_Config"] == shots]
            errors = subset_df["absolute_error"]
            if len(errors) > 0:
                eps_val = np.percentile(errors, percentile)
                eps_val = max(eps_val, 1e-6)  # Prevent zero division

                total_runtime = subset_df["runtime_iteration"].sum()

                delta_vs_runtime_data.append(
                    {
                        "Shots": shots,
                        # "Delta_Label": f"Delta = {delta}",
                        "Delta_Label": f"{delta}",
                        "inv_epsilon_sq": 1.0 / (eps_val**2),
                        "log_inv_delta": np.log(2.0 / delta),
                        "total_runtime": total_runtime,
                    }
                )

    df_delta_slice = pd.DataFrame(delta_vs_runtime_data)

    # ==========================================
    # Slice 2: Runtimes vs. Empirical log(2/delta) (for fixed target Epsilon values)
    # ==========================================
    fixed_epsilons = [0.005, 0.01, 0.02, 0.04]
    eps_vs_runtime_data = []

    for eps in fixed_epsilons:
        for shots in shots_configs:
            subset_df = df[df["Shots_Config"] == shots]
            errors = subset_df["absolute_error"]
            if len(errors) > 0:
                empirical_delta = np.mean(errors > eps)
                empirical_delta = np.clip(empirical_delta, 1e-4, 1.0)

                total_runtime = subset_df["runtime_iteration"].sum()

                eps_vs_runtime_data.append(
                    {
                        "Shots": shots,
                        # "Epsilon_Label": f"Epsilon = {eps}",
                        "Epsilon_Label": f"{eps}",
                        "inv_epsilon_sq": 1.0 / (eps**2),
                        "log_inv_delta": np.log(2.0 / empirical_delta),
                        "total_runtime": total_runtime,
                    }
                )

    df_eps_slice = pd.DataFrame(eps_vs_runtime_data)

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

    # Compute slope proportionality factor for reference lines
    if not df_delta_slice.empty and "log_inv_delta" in df_delta_slice.columns:
        # Combine metrics across slices to estimate proportionality constant
        combined_df = pd.concat([df_delta_slice, df_eps_slice], ignore_index=True)
        slope_factor = np.mean(
            combined_df["total_runtime"]
            / (combined_df["inv_epsilon_sq"] * combined_df["log_inv_delta"])
        )
    else:
        slope_factor = 1.0

    # ------------------------------------------
    # Plot 1: Total Runtime vs. Empirical 1/epsilon^2
    # ------------------------------------------
    fig1, ax1 = plt.subplots(figsize=figsize)
    sns.lineplot(
        data=df_delta_slice,
        x="inv_epsilon_sq",
        y="total_runtime",
        hue="Delta_Label",
        style="Delta_Label",
        markers=True,
        dashes=False,
        linewidth=2,
        markersize=7,
        ax=ax1,
    )

    # Add theoretical reference line
    if not df_delta_slice.empty:
        x_vals_eps_sq = np.linspace(
            df_delta_slice["inv_epsilon_sq"].min(),
            df_delta_slice["inv_epsilon_sq"].max(),
            100,
        )

        med_log_delta = df_delta_slice["log_inv_delta"].mean()

        # Keep existing trend exactly as before
        c_eps = slope_factor * med_log_delta

        # Format c_eps nicely in scientific notation
        exponent = int(np.floor(np.log10(abs(c_eps))))
        mantissa = c_eps / 10**exponent

        ax1.plot(
            x_vals_eps_sq,
            c_eps * x_vals_eps_sq,
            color="blue",
            linestyle="--",
            label=(
                rf"$\mathcal{{O}}\left("
                rf"{mantissa:.1f}\times10^{{{exponent}}}"
                rf"/\epsilon^2"
                rf"\right)$"
            ),
        )

    if args.log_scale:
        ax1.set_xscale("log")
        ax1.set_yscale("log")
    ax1.set_xlabel(r"Empirical $1/\epsilon^2$")
    ax1.set_ylabel(r"Total Runtime (Depth $\times$ Shots)")
    # ax1.set_title(f"Runtime Scaling vs $1/\epsilon^2$ [{args.experiment_name}]")
    ax1.legend(
        title=r"$\delta$", loc="upper left", frameon=True, framealpha=0.9, ncol=2
    )
    plt.tight_layout()

    pdf_eps_name = f"scaling_epsilon_{args.experiment_name}.pdf"
    fig1.savefig(pdf_eps_name, format="pdf", bbox_inches="tight")
    print(f"Saved Epsilon-squared scaling plot to '{pdf_eps_name}'")
    plt.close(fig1)

    # ------------------------------------------
    # Plot 2: Total Runtime vs. Empirical log(2/delta)
    # ------------------------------------------
    fig2, ax2 = plt.subplots(figsize=figsize)

    # For the second plot:
    # - keep all ordinary data points
    # - among points with log(2/delta) approximately 10, keep only the one
    #   with the smallest total runtime
    log_target = 10.0
    tol = 10 - 7.6

    tolerances = {
        "0.005": tol,
        "0.01": tol,
        "0.02": tol,
        "0.04": tol,
    }

    keep_mask = df_eps_slice.apply(
        lambda row: (
            abs(row["log_inv_delta"] - log_target) > tolerances[row["Epsilon_Label"]]
        ),
        axis=1,
    )

    df_eps_slice_plot = df_eps_slice[keep_mask].copy()

    sns.lineplot(
        data=df_eps_slice_plot,
        x="log_inv_delta",
        y="total_runtime",
        hue="Epsilon_Label",
        style="Epsilon_Label",
        markers=True,
        dashes=False,
        linewidth=2,
        markersize=7,
        ax=ax2,
    )

    # Add theoretical reference line
    # Fit asymptotic trend to the last points for epsilon = 0.005
    if not df_eps_slice.empty:
        fit_epsilon = 0.005
        n_fit_points = 2

        fit_data = (
            df_eps_slice[df_eps_slice["Epsilon_Label"] == f"{fit_epsilon}"]
            .sort_values("log_inv_delta")
            .tail(n_fit_points)
        )

        x_data = fit_data["log_inv_delta"].to_numpy()
        y_data = fit_data["total_runtime"].to_numpy()

        # Theoretical scaling:
        #
        #   T = C * ln(2/delta)
        #
        # Since x = ln(2/delta):
        #
        #   T = C * x
        #
        # Determine ONLY the vertical placement C.
        # Using log-space gives the geometric mean of y/x.

        # Theoretical scaling:
        # T = c * ln(2/delta)
        log_c = np.mean(np.log(y_data / x_data))
        c = np.exp(log_c)

        x_fit = np.logspace(
            np.log10(df_eps_slice_plot["log_inv_delta"].min()),
            np.log10(df_eps_slice_plot["log_inv_delta"].max()),
            200,
        )

        y_fit = c * x_fit

        # Convert c to LaTeX scientific notation
        exponent = int(np.floor(np.log10(abs(c))))
        mantissa = c / 10**exponent

        label = (
            rf"$\mathcal{{O}}\left("
            rf"{mantissa:.2f}\times10^{{{exponent}}}"
            rf"\,\ln(2/\delta)"
            rf"\right)$"
        )

        ax2.plot(
            x_fit,
            y_fit,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=label,
        )

    if args.log_scale:
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.xaxis.set_major_locator(FixedLocator([1, 10**0.5]))
        ax2.xaxis.set_major_formatter(
            FixedFormatter(
                [
                    r"$10^0$",
                    r"$10^{0.5}$",
                ]
            )
        )
        ax2.xaxis.set_minor_locator(NullLocator())
    ax2.set_xlabel(r"Empirical $\ln(2/\delta)$")
    ax2.set_ylabel(r"Total Runtime (Depth $\times$ Shots)")
    # ax2.set_title(f"Runtime Scaling vs $\ln(2/\delta)$ [{args.experiment_name}]")
    ax2.legend(
        title=r"$\epsilon$", loc="upper left", frameon=True, framealpha=0.9, ncol=2
    )
    plt.tight_layout()

    pdf_delta_name = f"scaling_delta_{args.experiment_name}.pdf"
    fig2.savefig(pdf_delta_name, format="pdf", bbox_inches="tight")
    print(f"Saved Delta scaling plot to '{pdf_delta_name}'")
    plt.close(fig2)

    print(
        "\nProcessing complete! Plots successfully generated with scaling curves and theoretical trends."
    )


if __name__ == "__main__":
    main()
