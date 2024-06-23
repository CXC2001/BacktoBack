# Code currently duplicated in Unfold.jl
# https://github.com/unfoldtoolbox/Unfold.jl/edit/main/src/solver.jl

# Basic implementation of https://doi.org/10.1016/j.neuroimage.2020.117028
function solver_b2b(X,data::AbstractArray{T,3};kwargs...) where {T<:Union{Missing,<:Number}}
    X, data = drop_missing_epochs(X, data) 
    solver_b2b(X,data; kwargs...)
end

# when T is a number, 
function solver_b2b(                     
        X,                               # design matrix
        data::AbstractArray{T,3};        # form a 3D array of data with type "T"
        cross_val_reps = 10,             # 10-time cross validation repetitions
        multithreading = true,           # multithreading is enabled
        show_progress = true,            # show the progress
    ) where {T<:Number}

    E = zeros(T,size(data, 2), size(X, 2), size(X, 2))    # form a 3D array of zeros with type "T" with the size of a*b*c, where a is the size of data, b is the size of X, and c is the size of X
    W = Array{T}(undef, size(data, 2), size(X, 2), size(data, 1))    

    prog = Progress(size(data, 2) * cross_val_reps;dt=0.1,enabled=show_progress)
    Unfold.@maybe_threads multithreading for t = 1:size(data, 2)
          
             for m = 1:cross_val_reps
                k_ix = collect(Kfold(size(data, 3), 2))
                X1 = @view X[k_ix[1], :] # view(X,k_ix[1],：)
                X2 = @view X[k_ix[2], :]
                    
                Y1 = @view data[:, t, k_ix[1]]
                Y2 = @view data[:, t, k_ix[2]]


                G = (Y1' \ X1) # decoding weight  vector, for each column of X independently how much do I need to weight each channel (similar to decoding betas)
                H = X2 \ (Y2' * G) # 

                E[t, :, :] +=  Diagonal(H[diagind(H)])
                ProgressMeter.next!(prog; showvalues = [(:time, t), (:cross_val_rep, m)])
            end
            E[t, :, :] .= E[t, :, :] ./ cross_val_reps
            W[t, :, :] .= (X * E[t, :, :])' / data[:, t, :]

        
    end


    
    
    using MLJ,MLJLinearModels,Tables,LinearAlgebra
    function solver_b2bcv(
            X,
            data::AbstractArray{T,3},
            cross_val_reps = 10;
            solver=(a,b,c)->ridge(a,b,c)
        ) where {T<:Union{Missing, <:Number}}
    
        #
        X,data = Unfold.dropMissingEpochs(X,data)
    
        # Open MLJ Tuner
        @load RidgeRegressor pkg=MLJLinearModels
        ##
    
        tm = TunedModel(model=RidgeRegressor(fit_intercept=false),
                    resampling = CV(nfolds=5),
                    tuning = Grid(resolution=10),
                    range = range(RidgeRegressor(), :lambda, lower=1e-2, upper=1000, scale=:log10),
                    measure = MLJ.rms)
    
    
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
    
                G = solver(tm,Y1',X1)
                H = solver(tm,X2, (Y2'*G))
    
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
        return beta, modelinfo
    end
    
    
    
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
    
    using GLMNet
    function ridge_glmnet(tm,data,X)
        G = Array{Float64}(undef,size(data,2),size(X,2))
        for pred in 1:size(X,2)
            #println(elscitype(data))
            cv = glmnetcv(data,X[:,pred],intercept=false)
            G[:,pred] =cv.path.betas[:,argmin(cv.meanloss)]
        end
        return G
    end
    
    #import ScikitLearn
    #using ScikitLearn.GridSearch: GridSearchCV
    
    #@ScikitLearn.sk_import linear_model: Ridge
    #function ridge_sklearn(tm,data,X)
    #    G = Array{Float64}(undef,size(data,2),size(X,2))
    #    D = Dict(:C => 10 .^range(log10(tm.range.lower),stop=log10(tm.range.upper),length=10))##
    
    #    cv = GridSearchCV(Ridge(),Dict(:alpha => (10 .^range(log10(tm.range.lower),stop=log10(tm.range.upper),length=10))))
    #    ScikitLearn.fit!(cv,data,X)
    
    #    G = cv.best_estimator_.coef_'
    
    #    return G
    
    
    #end