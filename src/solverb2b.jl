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