using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
dat, evts = UnfoldSim.predef_eeg(; noiselevel = 10, return_epoched = true)

evts.continuous_random .= rand(size(evts,1))
evts.categorical_correlated .= evts.continuous .+ 10 .*rand(size(evts,1)).>7.5 # make random samples with a correlation of e.g. 0.5 to evts.continuous
f = @formula 0 ~ 1  + continuous 

f = @formula 0 ~ 1  + categorical_correlated 

f = @formula 0 ~ 1  + continuous + categorical_correlated


f = @formula 0 ~ 1 + condition + continuous + continuous_random
designDict = [Any => (f, range(0, 0.44, step = 1/100))]


b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5)
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]);
m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)
results = coeftable(m)
results.estimate = abs.(results.estimate)
results = results[results.coefname .!="(Intercept)",:]
#---

plot_erp(results)