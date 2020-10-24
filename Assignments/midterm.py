import sys
from itertools import combinations, combinations_with_replacement, product

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from cat_correlation import cat_cont_correlation_ratio, cat_correlation
from pandas.api.types import is_string_dtype
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import confusion_matrix

pd.options.mode.chained_assignment = None


def load():

    filepath = "ENTER DATASET FILEPATH HERE"

    df = pd.read_csv(filepath)

    response = "ENTER RESPONSE VARIABLE HERE"
    predictors = "ENTER PREDICTORS HERE"

    return df, response, predictors


def response_processing(df, response):

    rcol = df[response]

    rcol.fillna(0, inplace=True)

    rcheck = is_string_dtype(rcol)
    rratio = len(np.unique(rcol.values)) / len(rcol.values)

    if rcheck or rratio < 0.05:
        rtype = "Categorical"

        rplot = px.histogram(rcol)
        rplot.write_html(file=f"./plots/response.html", include_plotlyjs="cdn")

        rcol = pd.Categorical(rcol, categories=rcol.unique())
        rcol, resp_labels = pd.factorize(rcol)

        rcol = pd.DataFrame(rcol, columns=[response])
        rcolnc = df[response]

    else:
        rtype = "Continuous"
        rcolnc = []

        rplot = px.histogram(rcol)
        rplot.write_html(file=f"./plots/response.html", include_plotlyjs="cdn")

    rmean = rcol.mean()

    return rcol, rtype, rmean, rcolnc


def predictor_processing(df, predictors, response, rcol, rtype, rmean, rcolnc):

    pcol = df[df.columns.intersection(predictors)]

    resultscols = [
        "Response",
        "Predictor Type",
        "Correlation",
        "t Score",
        "p Value",
        "Regression Plot",
        "Diff Mean of Response (Unweighted)",
        "Diff Mean of Response (Weighted)",
        "Diff Mean Plot",
    ]
    results = pd.DataFrame(columns=resultscols, index=predictors)

    bins = "ENTER NUMBER OF BINS HERE"

    for pname, pdata in pcol.iteritems():

        pdata.fillna(0, inplace=True)

        pcheck = is_string_dtype(pdata)
        pratio = len(pdata.unique()) / len(pdata)
        if pcheck or pratio < 0.05:
            ptype = "Categorical"

            pdata = pd.Categorical(pdata, categories=pdata.unique())
            pdata, pred_labels = pd.factorize(pdata)

            pdata = pd.DataFrame(pdata, columns=[pname])
            pdatanc = df[pname].fillna(0, inplace=True)

        else:
            ptype = "Continuous"

        df_c = pd.concat([rcol, pdata], axis=1)
        df_c.columns = [response, pname]

        if rtype == "Categorical" and ptype == "Categorical":
            resultsmatrix = confusion_matrix(pdata, rcol)
            relation = go.Figure(
                data=go.Heatmap(z=resultsmatrix, zmin=0, zmax=resultsmatrix.max())
            )
            relation.update_layout(
                title=f"Relationship Between {response} and {pname}",
                xaxis_title=pname,
                yaxis_title=response,
            )

            corr = cat_correlation(df_c[pname], df_c[response])

        elif rtype == "Categorical" and ptype == "Continuous":

            relation = px.histogram(df_c, x=pname, color=rcolnc)
            relation.update_layout(
                title=f"Relationship Between {response} and {pname}",
                xaxis_title=pname,
                yaxis_title="count",
            )

            corr = stats.pointbiserialr(df_c[response], df_c[pname])[0]

        elif rtype == "Continuous" and ptype == "Categorical":

            relation = px.histogram(df_c, x=response, color=pdatanc)
            relation.update_layout(
                title=f"Relationship Between {response} and {pname}",
                xaxis_title=response,
                yaxis_title="count",
            )

            corr = cat_cont_correlation_ratio(df_c[pname], df_c[response])

        elif rtype == "Continuous" and ptype == "Continuous":

            relation = px.scatter(y=rcol, x=pdata, trendline="ols")
            relation.update_layout(
                title=f"Relationship Between {response} and {pname}",
                xaxis_title=pname,
                yaxis_title=response,
            )

            corr = df_c[response].corr(df_c[pname])

        response_html = response.replace(" ", "")
        pname_html = pname.replace(" ", "")

        relationsave = f"./plots/{response_html}_{pname_html}_relate.html"
        relationopen = f"./{response_html}_{pname_html}_relate.html"
        relation.write_html(file=relationsave, include_plotlyjs="cdn")
        relationlink = (
            "<a target='blank' href=" + relationopen + "><div>" + ptype + "</div></a>"
        )

        if rtype == "Categorical":
            regression = sm.Logit(rcol, pdata, missing="drop")

        else:
            regression = sm.OLS(rcol, pdata, missing="drop")

        regfitted = regression.fit()

        t_score = round(regfitted.tvalues[0], 6)
        p_value = "{:.6e}".format(regfitted.pvalues[0])

        regfig = px.scatter(y=df_c[response], x=df_c[pname], trendline="ols")
        regfig.write_html(
            file=f"./plots/{pname}_regression.html", include_plotlyjs="cdn"
        )
        regfig.update_layout(
            title=f"Regression: {response} on {pname}",
            xaxis_title=pname,
            yaxis_title=response,
        )

        regsave = f"./plots/{response_html}_{pname_html}_reg.html"
        regopen = f"./{response_html}_{pname_html}_reg.html"
        regfig.write_html(file=regsave, include_plotlyjs="cdn")
        reg_link = "<a target='blank' href=" + regopen + "><div>Plot</div></a>"

        if ptype == "Continuous":
            df_c["bin_labels"] = pd.cut(df_c[pname], bins=bins, labels=False)
            binned_means = df_c.groupby("bin_labels").agg(
                {response: ["mean", "count"], pname: "mean"}
            )
            bin_f = bins

        else:
            df_c.columns = [f"{response}", f"{pname}"]
            binned_means = df_c.groupby(pdata.iloc[:, 0]).agg(
                {response: ["mean", "count"], pname: "mean"}
            )
            bin_f = len(np.unique(pdata.iloc[:, 0].values))

        binned_means.columns = [f"{response} mean", "count", f"{pname} mean"]

        binned_means["weight"] = binned_means["count"] / binned_means["count"].sum()
        binned_means["mean_sq_diff"] = (
            binned_means[f"{response} mean"].subtract(rmean, fill_value=0) ** 2
        )
        binned_means["mean_sq_diff_w"] = (
            binned_means["weight"] * binned_means["mean_sq_diff"]
        )

        msd_uw = binned_means["mean_sq_diff"].sum() * (1 / bin_f)
        msd_w = binned_means["mean_sq_diff_w"].sum()

        figdiff = make_subplots(specs=[[{"secondary_y": True}]])
        figdiff.add_trace(
            go.Bar(
                x=binned_means[f"{pname} mean"],
                y=binned_means["count"],
                name="Observations",
            )
        )
        figdiff.add_trace(
            go.Scatter(
                x=binned_means[f"{pname} mean"],
                y=binned_means[f"{response} mean"],
                line=dict(color="red"),
                name=f"Relationship with {response}",
            ),
            secondary_y=True,
        )
        figdiff.update_layout(
            title_text=f"Difference in Mean Response: {response} and {pname}",
        )
        figdiff.update_xaxes(title_text=f"{pname} (binned)")
        figdiff.update_yaxes(title_text="count", secondary_y=False)
        figdiff.update_yaxes(title_text=f"{response}", secondary_y=True)

        figdiffsave = f"./plots/{response_html}_{pname_html}_diff.html"
        figdiffopen = f"./{response_html}_{pname_html}_diff.html"
        figdiff.write_html(file=figdiffsave, include_plotlyjs="cdn")
        diff_link = "<a target='blank' href=" + figdiffopen + "><div>Plot</div></a>"

        if pname == pcol.columns[0]:
            pred_proc = pd.concat([rcol, pdata], axis=1)
        else:
            pred_proc = pd.concat([pred_proc, pdata], axis=1)

        results.loc[pname] = pd.Series(
            {
                "Response": response,
                "Predictor Type": relationlink,
                "Correlation": corr,
                "t Score": t_score,
                "p Value": p_value,
                "Regression Plot": reg_link,
                "Diff Mean of Response (Unweighted)": msd_uw,
                "Diff Mean of Response (Weighted)": msd_w,
                "Diff Mean Plot": diff_link,
            }
        )

    results = results.sort_values(["Correlation"], ascending=False)

    return pred_proc, results, pcol, bins


def predictor_processing_two_way(response, pcol, bins, rcol, rmean):
    combo = list(combinations(pcol.columns, 2))

    combo_len = range(1, len(combo))

    bruteforce = [
        "Response",
        "Predictor 1",
        "Predictor 2",
        "Predictor 1 Type",
        "Predictor 2 Type",
        "DMR Unweighted",
        "DMR Weighted",
        "DMR Weighted Plot",
    ]
    bfresults = pd.DataFrame(columns=bruteforce, index=combo_len)

    predictorcorr = [
        "Response",
        "Predictor 1",
        "Predictor 2",
        "Predictor 1 Type",
        "Resp/Pred 1 Plot",
        "Predictor 2 Type",
        "Resp/Pred 2 Plot",
        "Correlation",
    ]
    pcresults = pd.DataFrame(columns=predictorcorr, index=combo_len)

    comb_pos = 1

    for comb in combo:

        pname_1 = comb[0]
        pname_2 = comb[1]

        pdata_1 = pcol[comb[0]]
        pdata_2 = pcol[comb[1]]

        pcheck = is_string_dtype(pdata_1)
        pratio = len(pdata_1.unique()) / len(pdata_1)
        if pcheck or pratio < 0.05:
            ptype_1 = "Categorical"

            pdata_1 = pd.Categorical(pdata_1, categories=pdata_1.unique())
            pdata_1, pred_labels_1 = pd.factorize(pdata_1)

            pdata_1 = pd.DataFrame(pdata_1, columns=[pname_1])

        else:
            ptype_1 = "Continuous"

        pcheck = is_string_dtype(pdata_2)
        pratio = len(pdata_2.unique()) / len(pdata_2)
        if pcheck or pratio < 0.05:
            ptype_2 = "Categorical"

            pdata_2 = pd.Categorical(pdata_2, categories=pdata_2.unique())
            pdata_2, pred_labels_2 = pd.factorize(pdata_2)

            pdata_2 = pd.DataFrame(pdata_2, columns=[pname_2])

        else:
            ptype_2 = "Continuous"

        df_p = pd.concat([rcol, pdata_1, pdata_2], axis=1)

        if ptype_1 == "Categorical" and ptype_2 == "Categorical":
            corr = cat_correlation(df_p[pname_2], df_p[pname_1])

        elif (
            ptype_1 == "Categorical"
            and ptype_2 == "Continuous"
            or ptype_1 == "Continuous"
            and ptype_2 == "Categorical"
        ):

            if ptype_1 == "Categorical":

                corr = cat_cont_correlation_ratio(df_p[pname_1], df_p[pname_2])

            elif ptype_2 == "Categorical":

                corr = cat_cont_correlation_ratio(df_p[pname_2], df_p[pname_1])

        elif ptype_1 == "Continuous" and ptype_2 == "Continuous":

            corr = df_p[pname_1].corr(df_p[pname_2])

        if ptype_1 == "Continuous":
            df_p["bin_labels_1"] = pd.cut(df_p[pname_1], bins=bins, labels=False)
            bin_1_f = bins

        else:
            df_p["bin_labels_1"] = df_p[pname_1]
            bin_1_f = len(np.unique(pdata_1.iloc[:, 0].values))

        if ptype_2 == "Continuous":
            df_p["bin_labels_2"] = pd.cut(df_p[pname_2], bins=bins, labels=False)
            bin_2_f = bins

        else:
            df_p["bin_labels_2"] = df_p[pname_2]
            bin_2_f = len(np.unique(pdata_2.iloc[:, 0].values))

        binned_means_total = df_p.groupby(
            ["bin_labels_1", "bin_labels_2"], as_index=False
        ).agg({response: ["mean", "count"]})

        squared_diff = (
            binned_means_total.iloc[
                :, binned_means_total.columns.get_level_values(1) == "mean"
            ].sub(rmean, level=0)
            ** 2
        )

        binned_means_total["mean_sq_diff"] = squared_diff

        weights_group = binned_means_total.iloc[
            :, binned_means_total.columns.get_level_values(1) == "count"
        ]

        weights_tot = binned_means_total.iloc[
            :, binned_means_total.columns.get_level_values(1) == "count"
        ].sum()

        binned_means_total["weight"] = weights_group.div(weights_tot)

        binned_means_total["mean_sq_diff_w"] = (
            binned_means_total["weight"] * binned_means_total["mean_sq_diff"]
        )

        plot_data = binned_means_total.pivot(
            index="bin_labels_1", columns="bin_labels_2", values="mean_sq_diff_w"
        )
        fig_dmr = go.Figure(data=[go.Surface(z=plot_data.values)])
        fig_dmr.update_layout(
            title=f"DMR (Weighted): {pname_1} and {pname_2}",
            autosize=True,
            scene=dict(xaxis_title=pname_1, yaxis_title=pname_2, zaxis_title=response),
        )

        msd_uw_group = binned_means_total["mean_sq_diff"].sum() * (
            1 / (bin_1_f * bin_2_f)
        )
        msd_w_group = binned_means_total["mean_sq_diff_w"].sum()

        pname_1_html = pname_1.replace(" ", "")
        pname_2_html = pname_2.replace(" ", "")

        fig_dmr_file_save = f"./plots/{pname_1_html}_{pname_2_html}_dmr.html"
        fig_dmr_file_open = f"./{pname_1_html}_{pname_2_html}_dmr.html"
        fig_dmr.write_html(file=fig_dmr_file_save, include_plotlyjs="cdn")
        fig_dmr_link = (
            "<a target='blank' href=" + fig_dmr_file_open + "><div>Plot</div></a>"
        )

        response_html = response.replace(" ", "")

        relationopen_1 = f"./{response_html}_{pname_1_html}_relate.html"
        relationlink_1 = (
            "<a target='blank' href=" + relationopen_1 + "><div>Plot</div></a>"
        )

        relationopen_2 = f"./{response_html}_{pname_2_html}_relate.html"
        relationlink_2 = (
            "<a target='blank' href=" + relationopen_2 + "><div>Plot</div></a>"
        )

        bfresults.loc[comb_pos] = pd.Series(
            {
                "Response": response,
                "Predictor 1": pname_1,
                "Predictor 2": pname_2,
                "Predictor 1 Type": ptype_1,
                "Predictor 2 Type": ptype_2,
                "DMR Unweighted": msd_uw_group,
                "DMR Weighted": msd_w_group,
                "DMR Weighted Plot": fig_dmr_link,
            }
        )

        pcresults.loc[comb_pos] = pd.Series(
            {
                "Response": response,
                "Predictor 1": pname_1,
                "Predictor 2": pname_2,
                "Predictor 1 Type": ptype_1,
                "Resp/Pred 1 Plot": relationlink_1,
                "Predictor 2 Type": ptype_2,
                "Resp/Pred 2 Plot": relationlink_2,
                "Correlation": corr,
            }
        )

        comb_pos += 1

    bfresults = bfresults.sort_values(["DMR Weighted"], ascending=False)

    pcresults = pcresults.sort_values(["Correlation"], ascending=False)

    return bfresults, pcresults


def corr_matrix(pcresults, pcol):

    types_df_1 = pcresults[["Predictor 1", "Predictor 1 Type"]]
    types_df_1.columns = ["Predictor", "Type"]

    types_df_2 = pcresults[["Predictor 2", "Predictor 2 Type"]]
    types_df_2.columns = ["Predictor", "Type"]

    types_df = types_df_1.append(types_df_2)

    types = np.unique(types_df["Type"])

    type_combo = list(combinations_with_replacement(types, 2))

    for t_comb in type_combo:

        var_type_1 = t_comb[0]
        var_type_2 = t_comb[1]

        var_names_1 = types_df.loc[types_df["Type"] == var_type_1, "Predictor"].unique()
        var_names_2 = types_df.loc[types_df["Type"] == var_type_2, "Predictor"].unique()

        var_df_1 = pcol[var_names_1]
        var_df_2 = pcol[var_names_2]

        if var_type_1 == var_type_2 == "Continuous":

            corr_cont_matrix = var_df_1.corr()

            cont_cont_matrix = px.imshow(
                corr_cont_matrix,
                labels=dict(color="Pearson correlation:"),
                title=f"Correlation Matrix: {var_type_1} vs {var_type_2}",
            )
            cont_cont_matrix_save = f"./plots/cont_cont_matrix.html"
            cont_cont_matrix.write_html(
                file=cont_cont_matrix_save, include_plotlyjs="cdn"
            )

        elif var_type_1 == var_type_2 == "Categorical":

            var_factorized = var_df_1.apply(lambda x: pd.factorize(x)[0])

            cat_combo = list(product(var_factorized.columns, repeat=2))
            cat_combo_len = range(0, len(cat_combo))

            cat_corr_cols = [
                "Predictor 1",
                "Predictor 2",
                "Correlation",
            ]
            cat_corr = pd.DataFrame(columns=cat_corr_cols, index=cat_combo_len)

            cat_pos = 0

            for cat_comb in cat_combo:

                cat_name_1 = cat_comb[0]
                cat_name_2 = cat_comb[1]

                corr = cat_correlation(
                    var_factorized[cat_name_1], var_factorized[cat_name_2]
                )

                cat_corr.loc[cat_pos] = pd.Series(
                    {
                        "Predictor 1": cat_name_1,
                        "Predictor 2": cat_name_2,
                        "Correlation": corr,
                    }
                )

                cat_pos += 1

            corr_cat_matrix = cat_corr.pivot(
                index="Predictor 1", columns="Predictor 2", values="Correlation"
            )

            cat_cat_matrix = px.imshow(
                corr_cat_matrix,
                labels=dict(color="Cramer's V"),
                title=f"Correlation Matrix: {var_type_1} vs {var_type_2}",
            )
            cat_cat_matrix_save = f"./plots/cat_cat_matrix.html"
            cat_cat_matrix.write_html(file=cat_cat_matrix_save, include_plotlyjs="cdn")

        elif (
            var_type_1 == "Categorical"
            and var_type_2 == "Continuous"
            or var_type_1 == "Continuous"
            and var_type_2 == "Categorical"
        ):

            cat_cont_combo = list(product(var_names_1, var_names_2))
            cat_cont_combo_len = range(0, len(cat_cont_combo))

            cat_cont_corr_cols = [
                "Predictor 1",
                "Predictor 2",
                "Correlation",
            ]
            cat_cont_corr = pd.DataFrame(
                columns=cat_cont_corr_cols, index=cat_cont_combo_len
            )

            cat_cont_pos = 0

            for cat_cont_comb in cat_cont_combo:

                cat_cont_name_1 = cat_cont_comb[0]
                cat_cont_name_2 = cat_cont_comb[1]

                if var_type_1 == "Categorical":

                    corr = cat_cont_correlation_ratio(
                        var_df_1[cat_cont_name_1], var_df_2[cat_cont_name_2]
                    )

                elif var_type_2 == "Categorical":

                    corr = cat_cont_correlation_ratio(
                        var_df_2[cat_cont_name_2], var_df_1[cat_cont_name_1]
                    )

                cat_cont_corr.loc[cat_cont_pos] = pd.Series(
                    {
                        "Predictor 1": cat_cont_name_1,
                        "Predictor 2": cat_cont_name_2,
                        "Correlation": corr,
                    }
                )

                cat_cont_pos += 1

            corr_cat_cont_matrix = cat_cont_corr.pivot(
                index="Predictor 1", columns="Predictor 2", values="Correlation"
            )

            cat_cont_matrix = px.imshow(
                corr_cat_cont_matrix,
                labels=dict(color="Correlation Ratio"),
                title=f"Correlation Matrix: {var_type_1} vs {var_type_2}",
            )
            cat_cont_matrix_save = f"./plots/cat_cont_matrix.html"
            cat_cont_matrix.write_html(
                file=cat_cont_matrix_save, include_plotlyjs="cdn"
            )

    return


def results_table(results, bfresults, pcresults):

    with open("./plots/results.html", "w") as html_open:
        results.to_html(html_open, escape=False)

    with open("./plots/bfresults.html", "w") as html_open:
        bfresults.to_html(html_open, escape=False)

    with open("./plots/pcresults.html", "w") as html_open:
        pcresults.to_html(html_open, escape=False)

    return


def main():
    np.random.seed(seed=1234)
    df, response, predictors = load()
    rcol, rtype, rmean, rcolnc = response_processing(df, response)
    pred_proc, results, pcol, bins = predictor_processing(
        df, predictors, response, rcol, rtype, rmean, rcolnc
    )
    bfresults, pcresults = predictor_processing_two_way(
        response, pcol, bins, rcol, rmean
    )
    corr_matrix(pcresults, pcol)
    results_table(results, bfresults, pcresults)
    return


if __name__ == "__main__":
    sys.exit(main())
