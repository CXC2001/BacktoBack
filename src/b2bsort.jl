function solver_b2b(X,data::AbstractArray{T,3};kwargs...) where {T<:Union{Missing,<:Number}}
    X, data = drop_missing_epochs(X, data) 
    solver_b2b(X,data; kwargs...)
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

    # Choosing the regularization method to calculate tm
    if regularization_method == "RidgeRegressor"
        model = model_ridge()
        tm = tunemodel(model;kwargs...)
    elseif regularization_method == "SVMRegressor"
        model = model_svm()
        tm = tunemodel(model;kwargs...)
    elseif regularization_method == "LinearRegressor"
        model = model_linear()
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

function model_ridge()
    @load RidgeRegressor pkg=MLJLinearModels
    model = MLJLinearModels.RidgeRegressor(fit_intercept=false)
    return model
end

function model_svm(;kwargs...)
    @load SVMRegressor pkg=MLJScikitLearnInterface # Adjust the package if necessary
    model = MLJScikitLearnInterface.SVMRegressor()
    return model
end

function model_linear()
    LinearRegressor = @load LinearRegressor pkg=MultivariateStats
    model = LinearRegressor()
    return model
    
end

function tunemodel(model;nfolds=5,resolution = 10,measure=MLJ.rms,kwargs...)
    range = Base.range(model, :lambda, lower=1e-2, upper=1000, scale=:log10)
    tm = TunedModel(model=model,
                    resampling=CV(nfolds=nfolds),
                    tuning=Grid(resolution=resolution),
                    range=range,
                    measure=measure)
    return tm
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
