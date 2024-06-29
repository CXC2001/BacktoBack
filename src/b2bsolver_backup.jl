# Code currently duplicated in Unfold.jl
# https://github.com/unfoldtoolbox/Unfold.jl/edit/main/src/solver.jl

# Basic implementation of https://doi.org/10.1016/j.neuroimage.2020.117028
function solver_b2b(X,data::AbstractArray{T,3};kwargs...) where {T<:Union{Missing,<:Number}}
    X, data = drop_missing_epochs(X, data) 
    solver_b2b(X,data; kwargs...)
end

function tunemodel(model;nfolds=5,resolution = 10,measure=MLJ.rms,kwargs...)
    range = Base.range(model, :lambda, lower=1e-2, upper=1000, scale=:log10)
    tm = TunedModel(model=model,
                    resampling=CV(nfolds=nfolds),
                    tuning=Grid(resolution=resolution),
                    range=range,
                    measure=measure)
end

function model_ridge()
    @load RidgeRegressor pkg=MLJLinearModels
    model = MLJLinearModels.RidgeRegressor(fit_intercept=false)
    return model
end
function model_svm(;kwargs...)
    @load SVMRegressor pkg=MLJScikitLearnInterface # Adjust the package if necessary
    model = SVMRegressor()
    return model
end

function solver_b2b(                     # when T is a number, 
        X, # design matrix
        data::AbstractArray{T,3};  # form a 3D array of data with type "T"
        cross_val_reps = 10,
        multithreading = true,
        show_progress = true,
        # model::MLJmodel = [RidgeRegressor],
        regularization_method::String="RidgeRegressor",
        # solver=(a,b,c)->ridge(a,b,c)
        # regularization_method = model_ridge(),
        tune_model = true,
        kwargs...
    ) where {T<:Number}

    # tm = TunedModel(model=RidgeRegressor(fit_intercept=false),
    #         resampling = CV(nfolds=5),
    #         tuning = Grid(resolution=10),
    #         range = range(RidgeRegressor(), :lambda, lower=1e-2, upper=1000, scale=:log10),
    #         measure = MLJ.rms)

    # #Choosing the regularization method
    if regularization_method == "RidgeRegressor"
        model = model_ridge()
    elseif regularization_method == "SVMRegressor"
        model = model_svm()
    end
    # #Choosing the model: 
    if tune_model
        println("ssssssssssssssssssssssss")
        model = regularization_method
        tm = tunemodel(model;kwargs...)
    else
        println("dddddddddddddddddddddddddddd")
        tm = regularization_method
    end

    E = zeros(size(data,2),size(X,2),size(X,2))
    W = Array{Float64}(undef,size(data,2),size(X,2),size(data,1))
    println("n = samples = $(size(X,1)) = $(size(data,3))")
    @showprogress 0.1 for t in 1:size(data,2)        
    
        for m in 1:cross_val_reps
            k_ix = collect(Kfold(size(data,3),2))
            Y1 = data[:,t,k_ix[1]]
            Y2 = data[:,t,k_ix[2]]
            X1 = X[k_ix[1],:]
            X2 = X[k_ix[2],:]

            G = solver(tune_model,tm,Y1',X1)
            H = solver(tune_model,tm,X2, (Y2'*G))

            E[t,:,:] = E[t,:,:]+Diagonal(H[diagind(H)])

        end
        E[t,:,:] = E[t,:,:] ./ cross_val_reps
        W[t,:,:] = (X*E[t,:,:])' / data[:,t,:]

    end
    
    # extract diagonal
    beta = mapslices(diag,E,dims=[2,3])
    # reshape to conform to ch x time x pred
    beta = permutedims(beta,[3 1 2])
    modelinfo = Dict("W"=>W,"E"=>E,"cross_val_reps"=>cross_val_reps) # no history implemented (yet?)
    return Unfold.LinearModelFit(beta, modelinfo)
end

function solver(tune_model,tm,data,X)
    G = Array{Float64}(undef,size(data,2),size(X,2))
    for pred in 1:size(X,2)
        #println(elscitype(data))
        mtm = machine(tm,table(data),X[:,pred])
        fit!(mtm,verbosity=0)
        @show typeof(mtm)
        G[:,pred] = Tables.matrix(get_coefs(tune_model))[:,2]
        println("G = $G")
    end
    return G
end

function get_coefs(tune_model::true)
    return fitted_params(mtm).best_fitted_params.coefs

end
function get_coefs(tune_model::false)
    return fitted_params(mtm).coefs
end


function ridge_glmnet(tm,data,X)
    G = Array{Float64}(undef,size(data,2),size(X,2))
    for pred in 1:size(X,2)
        #println(elscitype(data))
        cv = glmnetcv(data,X[:,pred],intercept=false)
        G[:,pred] =cv.path.betas[:,argmin(cv.meanloss)]
    end
    return G
end
    
function SVM(tm,data,X)
    G = Array{Float64}(undef,size(data,2),size(X,2))
    for pred in 1:size(X,2)
        #println(elscitype(data))
          X2 = @view X[k_ix[2], :]
                    
#                 Y1 = @view data[:, t, k_ix[1]]
#                 Y2 = @view data[:, t, k_ix[2]]


#                 G = (Y1' \ X1)
#                 H = X2 \ (Y2' * G)   mtm = machine(tm,table(data),X[:,pred])
        fit!(mtm,verbosity=0)
        G[:,pred] = Tables.matrix(fitted_params(mtm).best_fitted_params.coefs)[:,2]
    end
    
end    
    


#     E = zeros(T,size(data, 2), size(X, 2), size(X, 2))
#     W = Array{T}(undef, size(data, 2), size(X, 2), size(data, 1))

#     prog = Progress(size(data, 2) * cross_val_reps;dt=0.1,enabled=show_progress)
#     Unfold.@maybe_threads multithreading for t = 1:size(data, 2)
          
#              for m = 1:cross_val_reps
#                 k_ix = collect(Kfold(size(data, 3), 2))
#                 X1 = @view X[k_ix[1], :] # view(X,k_ix[1],：)
#            


#                 E[t, :, :] +=  Diagonal(H[diagind(H)])
#                 ProgressMeter.next!(prog; showvalues = [(:time, t), (:cross_val_rep, m)])
#             end
#             E[t, :, :] .= E[t, :, :] ./ cross_val_reps
#             W[t, :, :] .= (X * E[t, :, :])' / data[:, t, :]

        
#     end

#     # extract diagonal
#     beta = mapslices(diag, E, dims = [2, 3])
#     # reshape to conform to ch x time x pred
#     beta = permutedims(beta, [3 1 2])
#     modelinfo = Dict("W" => W, "E" => E, "cross_val_reps" => cross_val_reps) # no history implemented (yet?)
#     return Unfold.LinearModelFit(beta, modelinfo)
# end
