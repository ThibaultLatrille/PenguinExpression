#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

eLevel = "TPM"


def filter_df(input_df: pd.DataFrame, x: str, cutoff: float = 0.0) -> pd.DataFrame:
    before = len(input_df)
    df_output = input_df[(input_df[x] > cutoff) & np.isfinite(input_df[x])]
    after = len(df_output)
    print(f"Filtering {x}: before={before}, after={after}. Removed {before - after} rows.")
    assert id(input_df) != id(df_output), "Input and output are the same object"
    return df_output


def open_data(input_path: str, opportunity_ratio: float) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep='\t')
    df[eLevel] = (df[f"k{eLevel}"] + df[f"e{eLevel}"]) / 2
    for sp in ['k', 'e']:
        df[f"{sp}dN/dS"] = opportunity_ratio * df[f"{sp}MisFix"] / df[f"{sp}SynFix"]
        pNpS = df[f"{sp}Mis#"] / df[f"{sp}Syn#"]
        finite = np.isfinite(pNpS)
        assert np.allclose(pNpS[finite], df[f"{sp}pN/pS"][finite]), "pN/pS is not equal to Mis#/Syn#"
        df[f"{sp}pN/pS"] = opportunity_ratio * pNpS
    return df


def rate_asfct_expression(input_df, q, cat: str = 'Syn#', rate: str = 'pN/pS'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    slopes = {}
    # Plot pN/pS ratio as a function of {eLevel}, both for king and emperor penguins
    for sp, name in [('k', "king"), ('e', 'emperor')]:
        x_col = f"{sp}{eLevel}"
        # Remove rows with {eLevel} = 0 rows with pS = 0
        df_sp = filter_df(filter_df(input_df, x_col, cutoff=0.0), f"{sp}{cat}", cutoff=0.0)
        # Create bins of equal size
        df_sp['qcut'] = pd.qcut(df_sp[x_col], q=q)
        gr_sp = df_sp.groupby("qcut")
        # np.average(x, weights=w) if you want to weight the average
        expression = [np.average(np.log(df_group[x_col])) for qcut, df_group in gr_sp]
        pNpS = [np.average(df_group[f"{sp}{rate}"]) for qcut, df_group in gr_sp]
        # Linear regression between log({eLevel}) and pN/pS
        slope, intercept = np.polyfit(expression, pNpS, 1)
        rsquared = np.corrcoef(expression, pNpS)[0, 1] ** 2
        label = f"slope={slope:.3f}, $R^2$={rsquared:.3f}"
        print(f"{rate} for {name.title()} penguins: {label}")
        ax = axs[0] if sp == 'k' else axs[1]
        ax.scatter(expression, pNpS, alpha=0.5)
        ax.plot(expression, slope * np.array(expression) + intercept, label=label)
        ax.set_xlabel(f"log({eLevel})")
        ax.set_ylabel(rate)
        ax.legend()
        ax.set_title(f"{name.title()} penguins")
        slopes[name] = slope
    plt.suptitle(f"{rate} ratio as a function of {eLevel} for {q} bins")
    plt.tight_layout()
    plt.savefig(f"{rate.replace('/', '')}vs{eLevel}_{q}bins.pdf")
    plt.close()
    plt.clf()
    return slopes


def plot_rates(input_df: pd.DataFrame, delta_log_pi: float, cat: str, rate: str, ax: plt.Axes, slopes: dict):
    # Filter rows with no synonymous substitutions
    df_e = filter_df(filter_df(input_df, f"e{cat}"), f"e{eLevel}")
    df_k = filter_df(filter_df(input_df, f"k{cat}"), f"k{eLevel}")
    mean_e, mean_k = np.average(df_e[f"e{rate}"]), np.average(df_k[f"k{rate}"])
    print(f"{rate}={mean_k:3f} for king penguins")
    print(f"{rate}={mean_e:3f} for emperor penguins")
    mean_delta_rate = mean_e - mean_k
    mean_slope = mean_delta_rate / delta_log_pi
    print(f"Average slope for {rate} is {mean_slope:.3f}")
    # Create bins of equal size
    boxplot_data = {}
    for q in [1, 10, 20, 50, 100]:
        df_e[f'cut{q}'] = pd.qcut(df_e[f"e{eLevel}"], q=q)
        df_k[f'cut{q}'] = pd.qcut(df_k[f"k{eLevel}"], q=q)
        list_delta_rate = []
        for (qcut_e, df_group_e), (qcut_k, df_group_k) in zip(df_e.groupby(f'cut{q}'), df_k.groupby(f'cut{q}')):
            e_rate = np.average(df_group_e[f"e{rate}"])
            k_rate = np.average(df_group_k[f"k{rate}"])
            delta_rate = e_rate - k_rate
            list_delta_rate.append(delta_rate)
        slope_list = [dy / delta_log_pi for dy in list_delta_rate]
        boxplot_data[q] = slope_list
    ax.boxplot(boxplot_data.values(), labels=boxplot_data.keys())
    for name, s in slopes.items():
        ax.axhline(s, color='green', label=f"{name.title()}={s:.3f}")
    ax.set_ylabel(f"Δω / Δlog(π)")
    ax.set_xlabel(f"Number of bins")
    ax.legend()
    ax.set_title(rate)


def rate_asfct_diversity(input_df, slopes):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    sum_len = input_df["exonLen"].sum()
    dico_pi = {sp: input_df[f"{sp}Syn#"].sum() / sum_len for sp in ['k', 'e']}
    print(f"pi={dico_pi}")
    delta_log_pi = np.log(dico_pi['e'] / dico_pi['k'])
    print(f"Delta_logNe={delta_log_pi:.3f}")

    # Filter rows with no synonymous polymorphisms
    plot_rates(input_df, delta_log_pi, 'SynFix', 'dN/dS', axs[0], slopes["div"])
    plot_rates(input_df, delta_log_pi, 'Syn#', 'pN/pS', axs[1], slopes["poly"])
    plt.suptitle(f"Susceptibility as function of effective population size")
    plt.tight_layout()
    plt.savefig(f"rate_diversity.pdf")
    plt.close()
    plt.clf()


def main(file_path: str):
    df_data = open_data(file_path, opportunity_ratio=0.3)
    slopes_poly = rate_asfct_expression(df_data, cat='Syn#', rate='pN/pS', q=100)
    slopes_div = rate_asfct_expression(df_data, cat='SynFix', rate='dN/dS', q=100)
    slopes = {"poly": slopes_poly, "div": slopes_div}
    rate_asfct_diversity(df_data, slopes)


if __name__ == "__main__":
    main('geneStatsExp.tsv')
