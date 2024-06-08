using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames

dat, evts = UnfoldSim.predef_eeg(; noiselevel = 10, return_epoched = true)

f = @formula 0 ~ 1 + condition + continuous
designDict = Dict(Any => (f, range(0, 1, length = size(dat, 1))))




# b2b_solver = (x, y) -> Unfold.solver_b2b(x, y; ross_val_reps = 5) # 旧版代码
b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5)
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2])
m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)
results = coeftable(m)
results.estimate = abs.(results.estimate)
results = results[results.coefname .!="(Intercept)",:]

plot_erp(results) # These are the decoding results for conditionA while considering conditionB, and vice versa.