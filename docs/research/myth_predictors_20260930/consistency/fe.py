"""Fast fixed-effects OLS with run-clustered SEs (within transform over one FE group, other FEs as dummies)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fe_ols(df: pd.DataFrame, y: str, xs: list[str], absorb: str | None, dummies: list[str] = (),
           cluster: str = "run_id") -> dict:
    cols = list(dict.fromkeys([y, *xs, *dummies, cluster] + ([absorb] if absorb else [])))
    d = df[cols].dropna().copy()
    X = d[xs].astype(float)
    for c in dummies:
        X = X.join(pd.get_dummies(d[c], prefix=c, drop_first=True, dtype=float))
    Y = d[y].astype(float)
    if absorb:
        g = d[absorb]
        X = X - X.groupby(g).transform("mean")
        Y = Y - Y.groupby(g).transform("mean")
        X = X.loc[:, X.abs().sum() > 1e-12]
    else:
        X = sm.add_constant(X)
    fit = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    out = {"n": int(fit.nobs), "n_runs": d[cluster].nunique()}
    for x in xs:
        if x in fit.params:
            ci = fit.conf_int().loc[x]
            out[x] = (fit.params[x], ci[0], ci[1], fit.pvalues[x])
    return out


def rows_from(res: dict, xs: list[str], **meta) -> list[dict]:
    return [{**meta, "term": x, "coef": res[x][0], "ci_low": res[x][1], "ci_high": res[x][2], "p": res[x][3],
             "n": res["n"], "n_runs": res["n_runs"]} for x in xs if x in res]
