import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")

# Load the resulting dataframe
df_result = pd.read_pickle("df_result.pkl")

# Compute both mean and standard deviation for train and test error
df_summary = (
    df_result.groupby(["method", "estimator"])
    .agg(
        train_err_mean=("train_err", "mean"),
        train_err_sd=("train_err", np.std),
        test_err_mean=("test_err", "mean"),
        test_err_sd=("test_err", np.std),
    )
    .reset_index()
)

# Define the desired order
method_order = ["biased", "naive", "debiased"]
estimator_order = ["regression", "ipw", "dr"]

df_summary["method"] = pd.Categorical(df_summary["method"], categories=method_order, ordered=True)
df_summary["estimator"] = pd.Categorical(
    df_summary["estimator"], categories=estimator_order, ordered=True
)

df_summary = df_summary.sort_values(["method", "estimator"]).reset_index(drop=True)

df_summary_dr = (
    df_summary[df_summary["estimator"] == "dr"].drop(columns=["estimator"]).reset_index(drop=True)
)

# Display the result of the DR estimator
print(df_summary_dr)

# Transform the dataframe to latex table
df_summary_dr_latex = df_summary_dr.to_latex(index=False)
print(df_summary_dr_latex)
