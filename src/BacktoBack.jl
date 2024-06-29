using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames
using Statistics
using MLBase
using MLJ
using MLJLinearModels,Tables,GLMNet

dat, evts = UnfoldSim.predef_eeg(; noiselevel = 0.1, return_epoched = true)

evts = rename(evts,:condition => :animal) # rename the column "condition" to "animal"
evts = rename(evts,:continuous => :eye_angle) # rename the column "continuous" to "eye_angle"
# evts.animal .= ifelse.(evts.animal .== "car", "dog", evts.animal) 
# evts.animal .= ifelse.(evts.animal .== "face", "cat", evts.animal)
# change the value of the column "animal" to "cat" if the value is "face", change the value of the column "animal" to "dog" if the value is "car"
evts.animal[evts.animal .== "car"] .= "dog"
evts.animal[evts.animal .== "face"] .= "cat"

evts.continuous_random .= rand(size(evts,1)) # add a new column "continuous_random" with random values 
evts.vegetable .= ["tomato","carrot"][1 .+ (evts.eye_angle .+ 10 .* rand(size(evts,1)) .> 7.5)] # make random samples with a correlation of e.g. 0.5 to evts.continuous

b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5)
b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5,regularization_method="LinearRegressor")
b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5,regularization_method="SVMRegressor")

# LinearRegressor
# b2bcv_solver = (x, y) -> UnfoldDecode.solver_b2bcv(x, y; cross_val_reps = 5)
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]); # repeat the dat 20 times and permute the dimensions
dat_3d .+= 0.1*rand(size(dat_3d)...)

#---
results_all = DataFrame()
for ix = 1:4
    if ix == 1
        f = @formula 0 ~ 1  + animal + eye_angle 
    elseif ix==2
        f = @formula 0 ~ 1  + animal + vegetable 
    elseif ix==3
        f = @formula 0 ~ 1  + animal + vegetable + eye_angle
    elseif ix ==4
        f = @formula 0 ~ 1 + animal + eye_angle + continuous_random + vegetable
    end
designDict = [Any => (f, range(0, 0.44, step = 1/100))] # designDict是一个字典，key是Any，value是一个元组，元组的第一个元素是f，第二个元素是range(0, 0.44, step = 1/100)

global m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver) # fit the model
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
X = modelmatrix(m)[1]
y = dat_3d
data = y
t = 15   


E = zeros(T,size(data, 2), size(X, 2), size(X, 2))
W = Array{T}(undef, size(data, 2), size(X, 2), size(data, 1))

k_ix = collect(Kfold(size(data, 3), 2))
X1 = @view X[k_ix[1], :] # view(X,k_ix[1],：)
X2 = @view X[k_ix[2], :]
                    
Y1 = @view data[:, t, k_ix[1]]
Y2 = @view data[:, t, k_ix[2]]


G = (Y1' \ X1)
X1_hat = Y1' * G
X2_hat = Y2' * G
H = X2 \ X2_hat

tm = TunedModel(model=RidgeRegressor(fit_intercept=false),
            resampling = CV(nfolds=5),
            tuning = Grid(resolution=10),
            range = range(RidgeRegressor(), :lambda, lower=1e-2, upper=1000, scale=:log10),
            measure = MLJ.rms)

function ridge(tm,data,X)
    G = Array{Float64}(undef,size(data,2),size(X,2))
    for pred in 1:size(X,2)
        #println(elscitype(data))
        mtm = machine(tm,table(data),X[:,pred])
        fit!(mtm,verbosity=0)
        G[:,pred] = Tables.matrix(fitted_params(mtm).best_fitted_params.coefs)[:,2]
    end
    return G
end

G = ridge(tm,Y1',X1)
H = ridge(tm,X2, (Y2'*G))



# plot!([1:10, 2, 1],[X1 X_hat], mc=[:blue :red])

# plot(X1[1:10, t, m], X_hat[1:10, 2, 1], xlabel = "True X", ylabel = "Estimated X", title = "True X vs. Estimated X")
#--
fig = Figure()
ax = Axis(fig[1, 1], 
    title = "X_true vs. X_hat",
    xlabel = "time",
    ylabel = "amplitude")
    ix = 3
scatter!(X1[1:30, ix], color=:blue, label="X1")
scatter!(X1_hat[1:30, ix], color=:red, label="X_hat")
display(fig)
#--
ix = 3
scatter(X1[:,ix],X1_hat[:, ix], color=:red, label="X_hat")
