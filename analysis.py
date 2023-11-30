#!/usr/bin/env python3
import os
from collections import namedtuple
import itertools
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

eLevel = "TPM"
Bootstrap = namedtuple("Bootstrap", ["mean", "replicates"])
GREEN = "#8FB03E"
RED = "#EB6231"
YELLOW = "#E29D26"
BLUE = "#5D80B4"
LIGHTGREEN = "#6ABD9B"


def filter_df(input_df: pd.DataFrame, x: str, cutoff: float = 0.0) -> pd.DataFrame:
    before = len(input_df)
    df_output = input_df[(input_df[x] > cutoff) & np.isfinite(input_df[x])]
    after = len(df_output)
    print(f"Filtering {x}: before={before}, after={after}. Removed {before - after} rows.")
    assert id(input_df) != id(df_output), "Input and output are the same object"
    return df_output


def bootstrap(input_df: pd.DataFrame) -> pd.DataFrame:
    df_output = input_df.sample(n=len(input_df), replace=True)
    return df_output


def open_data(input_path: str, theta) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep='\t')
    muN_muS = df["lds"] / df["ldn"]
    df[eLevel] = (df[f"k{eLevel}"] + df[f"e{eLevel}"]) / 2
    for sp in ['k', 'e']:
        df[f"{sp}dN/dS"] = muN_muS * df[f"{sp}MisFix"] / df[f"{sp}SynFix"]
        pNpS = df[f"{sp}Mis#"] / df[f"{sp}Syn#"]
        finite = np.isfinite(pNpS)
        assert np.allclose(pNpS[finite], df[f"{sp}pN/pS"][finite]), "pN/pS is not equal to Mis#/Syn#"
        df[f"{sp}pN/pS"] = muN_muS * pNpS
        piNpiS = df[f"{sp}Mis{theta}"] / df[f"{sp}Syn{theta}"]
        df[f"{sp}piN/piS"] = muN_muS * piNpiS
    return df


def format_label(label: str) -> str:
    if label == "piN/piS":
        return "$\\pi_{\\mathrm{N}} / \\pi_{\\mathrm{S}}$"
    elif label == "pN/pS":
        return "$p_{\\mathrm{N}} / p_{\\mathrm{S}}$"
    elif label == "dN/dS":
        return "$d_{\\mathrm{N}} / d_{\\mathrm{S}}$"
    else:
        return label


def format_chi_label(label: str) -> str:
    if label == "king" or label == "k":
        return "$\\chi^{\\mathrm{K}}$"
    elif label == "emperor" or label == "e":
        return "$\\chi^{\\mathrm{E}}$"
    elif label == "exome":
        return "$\\frac{\\Delta \\omega}{\\Delta \\log(N_\\mathrm{e})}$"
    else:
        return label


def average_group(gr_sp, x_col: str) -> np.array:
    return np.array([np.average(df_group[x_col]) if len(df_group) > 0 else np.nan for qcut, df_group in gr_sp])


def slope_asfct_expression(df_sp: pd.DataFrame, q: int, sp: str, x_col: str, rate: str, name: str, ax: plt.Axes = None):
    if 0 < q < len(df_sp) // 2:
        df_sp['qcut'] = pd.qcut(df_sp[x_col], q=q, duplicates='drop')
        gr_sp = df_sp.groupby("qcut")
        # np.average(x, weights=w) if you want to weight the average
        expression = np.log(average_group(gr_sp, x_col))
        pNpS = average_group(gr_sp, f"{sp}{rate}")
    else:
        expression = np.log(df_sp[x_col])
        pNpS = np.array(df_sp[f"{sp}{rate}"])
    # Linear regression between log({eLevel}) and pN/pS
    finite = np.isfinite(expression) & np.isfinite(pNpS)
    slope, intercept = np.polyfit(expression[finite], pNpS[finite], 1)
    rsquared = np.corrcoef(expression[finite], pNpS[finite])[0, 1] ** 2
    if ax is not None:
        label = f"{format_chi_label(sp)} = {slope:.3f}, $R^2$={rsquared:.3f}"
        print(f"{rate} for {name.title()} penguins: {label}")
        color = BLUE if name == "king" else YELLOW
        if 0 < q < len(df_sp) // 2:
            ax.scatter(expression, pNpS, alpha=0.5, color=color)
        else:
            keep = (pNpS < 1) & (pNpS > 0)
            ax.scatter(expression[keep], pNpS[keep], alpha=0.1, color=color)
        ax.plot(expression, slope * np.array(expression) + intercept, label=label, color=RED)
        ax.legend()
    return slope, rsquared


def rate_asfct_expression(input_df, output_path: str, cat_filter: str, rate: str, q: int, rep: int, cutoff: float):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    slopes = dict()
    # Plot pN/pS ratio as a function of {eLevel}, both for king and emperor penguins
    for sp, name in [('k', "king"), ('e', 'emperor')]:
        x_col = f"{sp}{eLevel}"
        # Remove rows with {eLevel} = 0 rows with pS = 0
        df_sp = filter_df(filter_df(input_df, x_col, cutoff=0.0), f"{sp}{cat_filter}", cutoff=cutoff)
        if len(df_sp) < 2:
            print(f"Skipping {name} penguins because of insufficient data")
            continue
        # Create bins of equal size
        # Linear regression between log({eLevel}) and pN/pS
        ax = axs[0] if sp == 'k' else axs[1]
        slope, rsquared = slope_asfct_expression(df_sp, q, sp, x_col, rate, name, ax=ax)
        ax.set_xlabel(f"log({eLevel})")
        ax.set_ylabel(format_label(rate))
        ax.set_title(f"{name.title()} penguins ({len(df_sp)} genes)")
        slopes[name] = Bootstrap(mean=slope, replicates=[])
        for i in range(rep):
            df_bootstrap = bootstrap(df_sp)
            slope, rsquared = slope_asfct_expression(df_bootstrap, q, sp, x_col, rate, "", ax=None)
            slopes[name].replicates.append(slope)

    if len(slopes) == 0:
        print("No data to plot")
        return slopes
    plt.suptitle(f"{format_label(rate)} ratio as a function of {eLevel} for {q} bins")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path + ".pdf", format="pdf")
    print(f"Saving figure to {output_path}")
    plt.close()
    plt.clf()
    return slopes


def exome_slope_asfct_diversity(df: pd.DataFrame, mut: str, delta_log_pi: float):
    dico = dict()
    muN_muS = np.sum(df["lds"]) / np.sum(df["ldn"])
    for sp in ['k', 'e']:
        sum_n = np.sum(df[f"{sp}Mis{mut}"])
        sum_s = np.sum(df[f"{sp}Syn{mut}"])
        dico[sp] = muN_muS * sum_n / sum_s
    delta_rate = dico['e'] - dico['k']
    slope = delta_rate / delta_log_pi
    return slope


def plot_rates(input_df: pd.DataFrame, delta_log_pi: float, mut: str, rate: str, ax: plt.Axes,
               slopes_expression: dict, rep: int):
    boxplot_data = {k: v.replicates for k, v in slopes_expression.items()}
    boxplot_data["exome"] = list()

    mean_data = {k: v.mean for k, v in slopes_expression.items()}
    mean_data["exome"] = exome_slope_asfct_diversity(input_df, mut, delta_log_pi)

    for i in range(rep):
        exome_bootstrap = bootstrap(input_df)
        boxplot_data["exome"].append(exome_slope_asfct_diversity(exome_bootstrap, mut, delta_log_pi))

    assert len(boxplot_data) == len(mean_data)
    medianprops = dict(linestyle='-', linewidth=0)
    ax.boxplot(boxplot_data.values(), labels=[format_chi_label(k) for k in boxplot_data.keys()],
               medianprops=medianprops)
    for i, (name, b) in enumerate(mean_data.items()):
        color = BLUE if name == "king" else (YELLOW if name == "emperor" else GREEN)
        ax.scatter(i + 1, b, label=f"{format_chi_label(name)}={b:.3f}", color=color)
    ax.set_ylabel("$\\chi$")
    ax.set_xlabel("")
    ax.legend()
    ax.set_title(f"$\\omega$={format_label(rate)}")


def rate_asfct_diversity(input_df, slopes, output_path, poly: str, rate_poly: str, rep: int, delta_log_pi: float):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Filter rows with no synonymous polymorphisms
    plot_rates(input_df, delta_log_pi, 'Fix', 'dN/dS', axs[0], slopes["div"], rep)
    plot_rates(input_df, delta_log_pi, poly, rate_poly, axs[1], slopes["poly"], rep)
    plt.suptitle(f"Estimators of $\\chi$ ({rep} bootstrap replicates)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path + ".pdf", format="pdf")
    print(f"Saving figure to {output_path}")
    plt.close()
    plt.clf()


def compute_delta_log_pi(input_path: str, region: str = "Intergenic", theta: str = "tajima"):
    df = pd.read_csv(input_path, sep='\t')
    df_e = df[(df["pop"] == "e") & (df["region"] == region)]
    df_k = df[(df["pop"] == "k") & (df["region"] == region)]
    assert len(df_e) == len(df_k) == 1, f"Expected 1 row for {region} in each population"
    pi_e = df_e[theta].values[0]
    pi_k = df_k[theta].values[0]
    dlp = np.log(pi_e / pi_k)
    print(f"Delta log pi = {dlp}")
    return dlp


def run(file_path: str, q=100, rep=1000, cutoff=5):
    # '#' for polymorphism, 'Tajima' for diversity
    poly = "Tajima"  # or "Watterson", "Fay_wu" or # for polymorphism
    rate_poly = 'piN/piS'  # or 'pN/pS' for polymorphism
    df_data = open_data(file_path, poly)
    slopes_poly = rate_asfct_expression(df_data, output_path=f"results/poly_eLevel/{q}bins_{cutoff}cutoff",
                                        cat_filter='Syn#', rate=rate_poly, q=q, rep=rep, cutoff=cutoff)
    slopes_div = rate_asfct_expression(df_data, output_path=f"results/div_eLevel/{q}bins_{cutoff}cutoff",
                                       cat_filter='SynFix', rate='dN/dS', q=q, rep=rep, cutoff=cutoff)
    slopes = {"poly": slopes_poly, "div": slopes_div}
    delta_log_pi = compute_delta_log_pi("results/theta_intergenic_allVariants.tsv")
    rate_asfct_diversity(df_data, slopes, output_path=f"results/rate_diversity/{q}bins_{cutoff}cutoff",
                         poly=poly, rate_poly=rate_poly, rep=rep, delta_log_pi=delta_log_pi)


def main():
    qcuts = [0, 25, 50, 100]
    cutoffs = [0]
    for qcut, cutoff in itertools.product(qcuts, cutoffs):
        run('results/geneStatsTheta.tsv', q=qcut, rep=1000, cutoff=cutoff)


if __name__ == "__main__":
    main()
