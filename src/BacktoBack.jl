using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames
dat, evts = UnfoldSim.predef_eeg(; noiselevel = 10, return_epoched = true)
# 如果名为condition的列的取值为"car"，则重新赋值为"dog"；
evts = rename(evts,:condition => :animal)
evts.animal .= ifelse.(evts.animal .== "car", "dog", evts.animal)
# 如果是"face"，则重新赋值为"cat"
evts.animal .= ifelse.(evts.animal .== "face", "cat", evts.animal)
evts.continuous_random .= rand(size(evts,1)) #生成了2000个0-1的随机数
evts.vegetable .= ["tomato","carrot"][1 .+ (evts.continuous .+ 10 .* rand(size(evts,1)) .> 7.5)] # make random samples with a correlation of e.g. 0.5 to evts.continuous

results_all = DataFrame()
for ix = 1:4
    if ix == 1
        f = @formula 0 ~ 1  + animal + continuous 
    elseif ix==2
        f = @formula 0 ~ 1  + animal + vegetable 
    elseif ix==3
        f = @formula 0 ~ 1  + animal + continuous + vegetable
    elseif ix ==4
        f = @formula 0 ~ 1 + animal + continuous + continuous_random
    end
designDict = [Any => (f, range(0, 0.44, step = 1/100))] # designDict是一个字典，key是Any，value是一个元组，元组的第一个元素是f，第二个元素是range(0, 0.44, step = 1/100)


b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5)
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]);
m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)
results = coeftable(m)
results.estimate = abs.(results.estimate)
results = results[results.coefname .!="(Intercept)",:]
#---
results.formula .= string(f)
results_all = vcat(results_all,results)


end
plot_erp(results_all;mapping=(;row=:formula))