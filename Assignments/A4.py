import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix


def load():

    filepath = "ENTER DATABASE FILEPATH HERE"

    df = pd.read_csv(filepath)

    response = "ENTER RESPONSE VARIABLE HERE"
    predictors = "ENTER PREDICTORS HERE"

    return df, response, predictors


def response_processing(df, response):

    rcol = df[response]
    rcheck = isinstance(rcol.values, str)
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

    resultsnames = [
        "Response",
        "Predictor Type",
        "t Score",
        "p Value",
        "Regression Plot",
        "Diff Mean of Response (Unweighted)",
        "Diff Mean of Response (Weighted)",
        "Diff Mean Plot",
    ]
    results = pd.DataFrame(columns=resultsnames, index=predictors)

    for pname, pdata in pcol.iteritems():

        pcheck = isinstance(pdata, str)
        pratio = len(pdata.unique()) / len(pdata)
        if pcheck or pratio < 0.05:
            ptype = "Categorical"

            pdata = pd.Categorical(pdata, categories=pdata.unique())
            pdata, pred_labels = pd.factorize(pdata)

            pdata = pd.DataFrame(pdata, columns=[pname])
            pdata_uncoded = df[pname]

        else:
            ptype = "Continuous"
            pdata = pdata.to_frame()

        df_c = pd.concat([rcol, pdata], axis=1)
        df_c.columns = [response, pname]

        if rtype == "Categorical" and ptype == "Categorical":
            relationmatrix = confusion_matrix(pdata, rcol)
            relation = go.Figure(
                data=go.Heatmap(z=relationmatrix, zmin=0, zmax=relationmatrix.max())
            )
            relation.update_layout(
                title=f"Relationship Between {response} and {pname}",
                xaxis_title=pname,
                yaxis_title=response,
            )

        elif rtype == "Categorical" and ptype == "Continuous":

            relation = px.histogram(df_c, x=pname, color=rcolnc)
            relation.update_layout(
                title=f"Relationship Between {response} and {pname}",
                xaxis_title=pname,
                yaxis_title="count",
            )

        elif rtype == "Continuous" and ptype == "Categorical":

            relation = px.histogram(df_c, x=response, color=pdata_uncoded)
            relation.update_layout(
                title=f"Relationship Between {response} and {pname}",
                xaxis_title=response,
                yaxis_title="count",
            )

        elif rtype == "Continuous" and ptype == "Continuous":

            relation = px.scatter(y=rcol, x=pdata, trendline="ols")
            relation.update_layout(
                title=f"Relationship Between {response} and {pname}",
                xaxis_title=pname,
                yaxis_title=response,
            )

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
        reglink = "<a target='blank' href=" + regopen + "><div>Plot</div></a>"

        if ptype == "Continuous":
            bins = "ENTER NUMBER OF BINS HERE"

            df_c["bin_labels"] = pd.cut(df_c[pname], bins=bins, labels=False)
            binned_means = df_c.groupby("bin_labels").agg(
                {response: ["mean", "count"], pname: "mean"}
            )

        else:
            df_c.columns = [f"{response}", f"{pname}"]
            binned_means = df_c.groupby(pdata.iloc[:, 0]).agg(
                {response: ["mean", "count"], pname: "mean"}
            )
            bins = len(np.unique(pdata.iloc[:, 0].values))

        binned_means.columns = [f"{response} mean", "count", f"{pname} mean"]

        binned_means["weight"] = binned_means["count"] / binned_means["count"].sum()
        binned_means["mean_sq_diff"] = (
            binned_means[f"{response} mean"].subtract(rmean, fill_value=0) ** 2
        )
        binned_means["mean_sq_diff_w"] = (
            binned_means["weight"] * binned_means["mean_sq_diff"]
        )

        msd_uw = binned_means["mean_sq_diff"].sum() * (1 / bins)
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

        figdiff_file_save = f"./plots/{response_html}_{pname_html}_diff.html"
        figdiff_file_open = f"./{response_html}_{pname_html}_diff.html"
        figdiff.write_html(file=figdiff_file_save, include_plotlyjs="cdn")
        diff_link = (
            "<a target='blank' href=" + figdiff_file_open + "><div>Plot</div></a>"
        )

        if pname == pcol.columns[0]:
            pproc = pd.concat([rcol, pdata], axis=1)
        else:
            pproc = pd.concat([pproc, pdata], axis=1)

        results.loc[pname] = pd.Series(
            {
                "Response": response,
                "Predictor Type": relationlink,
                "t Score": t_score,
                "p Value": p_value,
                "Regression Plot": reglink,
                "Diff Mean of Response (Unweighted)": msd_uw,
                "Diff Mean of Response (Weighted)": msd_w,
                "Diff Mean Plot": diff_link,
            }
        )

    return pproc, results


def random_forest_importance(rtype, pproc, predictors):

    if rtype == "Categorical":
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()

    cols = pproc.columns[pproc.isna().any()].tolist()
    if cols == "":
        pass
    else:
        pproc.loc[:, cols] = pproc.loc[:, cols].fillna(pproc.loc[:, cols].mean())

    model.fit(pproc.iloc[:, 1:], pproc.iloc[:, 0])
    importance = model.feature_importances_
    importance = importance.reshape(len(predictors), 1)
    importance = pd.DataFrame(importance, index=predictors)
    importance.columns = ["Random Forest Importance"]

    return importance


def results_table(results, importance):
    results = pd.concat([results, importance], axis=1)

    with open("./plots/results.html", "w") as html_open:
        results.to_html(html_open, escape=False)

    return


def main():
    np.random.seed(seed=1234)
    df, response, predictors = load()
    rcol, rtype, rmean, rcolnc = response_processing(df, response)
    pproc, results = predictor_processing(
        df, predictors, response, rcol, rtype, rmean, rcolnc
    )
    importance = random_forest_importance(rtype, pproc, predictors)
    results_table(results, importance)
    return


if __name__ == "__main__":
    sys.exit(main())
