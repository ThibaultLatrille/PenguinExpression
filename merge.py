from collections import defaultdict
import pandas as pd


def main():
    input_opp = "results/mutational_opportunities.tsv"
    input_theta = "results/theta.tsv"
    input_stats = "data/geneStatsExp.tsv"
    output_path = "results/geneStatsTheta.tsv"
    df_opp = pd.read_csv(input_opp, sep='\t')
    df_theta = pd.read_csv(input_theta, sep='\t')
    df_stats = pd.read_csv(input_stats, sep='\t')

    dico_opp = {row["gene"]: row for _, row in df_opp.iterrows()}
    dico_theta = defaultdict(dict)
    for _, row in df_theta.iterrows():
        dico_theta[row["gene"]][(row["pop"], row["type"])] = row

    output_dict = defaultdict(list)
    for _, row in df_stats.iterrows():
        for col in df_stats.columns:
            output_dict[col].append(row[col])
        gene = row["gene"]
        lds = dico_opp[gene]["mu_syn"]
        ldn = dico_opp[gene]["mu_nonsyn"]
        output_dict["lds"].append(lds)
        output_dict["ldn"].append(ldn)
        for k, v in dico_theta[gene].items():
            pop, type_mut = k
            prefix = f"{pop[0]}{'Syn' if type_mut == 'synonymous' else 'Mis'}"
            for theta in ["Watterson", "Tajima", "Fay_wu"]:
                output_dict[f"{prefix}{theta}"].append(v[theta.lower()])

    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(output_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
