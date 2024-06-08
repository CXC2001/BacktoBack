using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames

dat, evts = UnfoldSim.predef_eeg(; noiselevel = 10, return_epoched = true)

evts = rename(evts,:condition => :animal) # rename the column "condition" to "animal"
evts = rename(evts,:continuous => :eye_angle) # rename the column "continuous" to "eye_angle"
# evts.animal .= ifelse.(evts.animal .== "car", "dog", evts.animal) 
# evts.animal .= ifelse.(evts.animal .== "face", "cat", evts.animal)
if "car" in evts.animal # change the value of the column "animal" to "dog" if the value is "car"
    evts.animal .= "dog"    
end
if "face" in evts.animal # change the value of the column "animal" to "cat" if the value is "face"
    evts.animal .= "cat"    
end

evts.continuous_random .= rand(size(evts,1)) # add a new column "continuous_random" with random values 
evts.vegetable .= ["tomato","carrot"][1 .+ (evts.eye_angle .+ 10 .* rand(size(evts,1)) .> 7.5)] # make random samples with a correlation of e.g. 0.5 to evts.continuous

b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5)
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]);
dat_3d .+= rand(size(dat_3d)...)

results_all = DataFrame()
for ix = 1:4
    if ix == 1
        f = @formula 0 ~ 1  + animal + eye_angle 
    elseif ix==2
        f = @formula 0 ~ 1  + animal + vegetable 
    elseif ix==3
        f = @formula 0 ~ 1  + animal + eye_angle + vegetable
    elseif ix ==4
        f = @formula 0 ~ 1 + animal + eye_angle + continuous_random
    end
designDict = [Any => (f, range(0, 0.44, step = 1/100))] # designDict是一个字典，key是Any，value是一个元组，元组的第一个元素是f，第二个元素是range(0, 0.44, step = 1/100)


global m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)
results = coeftable(m)
results.estimate = abs.(results.estimate)
results = results[results.coefname .!="(Intercept)",:]
#---
results.formula .= string(f)
results_all = vcat(results_all,results)


end

plot_erp(results_all;mapping=(;row=:formula))

#---
T = Float64
m = 1
t = 1
X = modelmatrix(m)[1]
y = dat_3d