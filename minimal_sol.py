import interview_prep as H   # <-- your helper module name

H.cvr_plot_pop_vs_sample("case.xlsx", output_img="chi2_hist.png", alpha=0.05, show=False)
print("Done (note: no plot is saved if p < alpha).")