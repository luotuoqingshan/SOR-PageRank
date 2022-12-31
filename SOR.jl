module SOR
using LinearAlgebra
using SparseArrays
using Arpack


function spectral_radius(A::SparseMatrixCSC)
    eigvals, _ = eigs(A, which=:LM)
    rho = maximum(abs.(eigvals))
    return rho
end


# formulate the personalized PageRank linear system
function create_ppr_ls(A::SparseMatrixCSC, seed::Int, alpha::Float64)
    n = size(A, 1)
    P = sparse(Matrix(I, n, n)) - alpha * A ./ sum(A, dims=1)
    v = zeros(n)
    v[seed] = 1
    b = (1 - alpha) * v
    return P, b
end

function create_sym_ppr_ls(A::SparseMatrixCSC, seed::Int, alpha::Float64)
    n = size(A, 1)
    d = vec(sum(A, dims=1))
    d = sqrt.(d)
    ai, aj, av = findnz(A)
    P = sparse(Matrix(I, n, n)) - alpha * sparse(ai, aj, av ./ (d[ai] .* d[aj])) 
    @show size(A)
    v = zeros(n)
    v[seed] = 1
    b = (1 - alpha) * v ./ d
    return P, b
end


function optim_omega(A::SparseMatrixCSC)
    D = Diagonal(A)
    n = size(A, 1)
    G_Jac = sparse(Matrix(1.0I, n, n)) - inv(D) * A 
    mu = spectral_radius(G_Jac)
    omega_opt = 1 + (mu / (1 + sqrt(1 - mu^2)))^2 
    return mu, omega_opt 
end


struct iterstats
    iterate::Int
    x::Vector
    error::Vector
    residual::Vector
end


function SOR_Iteration!(x::Vector, A::SparseMatrixCSC, b::Vector, omega::Float64)
    n = size(A, 1)
    x_old = deepcopy(x) 
    for i = 1 : length(x) 
        x[i] = (1 - omega) * x_old[i] + 
               omega * ((b[i] - A[i, 1:i-1]' * x[1:i-1] - A[i, i+1:n]' * x_old[i+1:n]) / A[i, i]) 
    end
end


function SOR_solve(A::SparseMatrixCSC, b::Vector, omega::Float64; maxiter::Int=1000, tol::Float64=1e-4) 
    n = size(A, 1)
    x_sol = zeros(n) 

    x_opt = A \ b
    results = iterstats[]
    for i = 1 : maxiter
        err = x_opt - x_sol
        res = b - A * x_sol 
        if norm(res) < tol
            break
        end
        push!(results, iterstats(i, deepcopy(x_sol), err, res))
        SOR_Iteration!(x_sol, A, b, omega)
    end
    return results
end


struct sorstepstats
    iterate::Int
    index::Int
    x::Vector
    error::Vector
    residual::Vector
end


function SOR_stepwise(A::SparseMatrixCSC, b::Vector, omega::Float64; maxiter::Int=1000, tol::Float64=1e-4)
    n = size(A, 1)
    x_sol = zeros(n)
    x_opt = A \ b
    results = sorstepstats[]
    j = 1
    for i = 1 : maxiter
        err = x_opt - x_sol
        res = b - A * x_sol 
        if norm(res) < tol
            break
        end
        x_sol[j] = (1 - omega) * x_sol[j] + 
            omega * ((b[j] - A[j, 1:j-1]' * x_sol[1:j-1] - A[j, j+1:n]' * x_sol[j+1:n]) / A[j, j]) 
        push!(results, sorstepstats(i, j, deepcopy(x_sol), err, res))
        j += 1
        if j > n
            j -= n
        end
    end
    return results
end

function Richardson(A::SparseMatrixCSC, b::Vector, omega::Float64; maxiter::Int=1000, tol::Float64=1e-4) 
    n = size(A, 1)
    x_sol = zeros(n)
    x_opt = A \ b
    results = iterstats[]
    for i = 1 : maxiter
        err = x_opt - x_sol
        res = b - A * x_sol
        if norm(res) < tol
            break
        end
        push!(results, iterstats(i, deepcopy(x_sol), err, res))
        x_sol = x_sol + omega * res
    end
    return results
end


function Jacobi_Iteration!(x::Vector, A::SparseMatrixCSC, b::Vector)
    n = length(x)
    x_old = deepcopy(x)
    for i = 1 : length(x)
        x[i] = (b[i] - A[i, :]' * x_old + A[i, i] * x_old[i]) / A[i, i]
    end
end


function Jacobi(A::SparseMatrixCSC, b::Vector; maxiter::Int=1000, tol::Float64=1e-4)
    n = size(A, 1)
    x_sol = zeros(n)
    x_opt = A \ b
    results = iterstats[]
    for i = 1 : maxiter 
        err = x_opt - x_sol
        res = b - A * x_sol
        if norm(res) < tol
            break
        end
        push!(results, iterstats(i, deepcopy(x_sol), err, res)) 
        Jacobi_Iteration!(x_sol, A, b)
    end
    return results
end


function Gauss_Seidel_Iteration!(x::Vector, A::SparseMatrixCSC, b::Vector)
    n = length(x)
    x_old = deepcopy(x)
    for i = 1 : length(x)
        x[i] = (b[i] - A[i, 1:i-1]' * x[1:i-1] - A[i, i+1:n]' * x_old[i+1:n]) / A[i, i]
    end
end


function Gauss_Seidel(A::SparseMatrixCSC, b::Vector; maxiter::Int=1000, tol::Float64=1e-4)
    n = size(A, 1)
    x_sol = zeros(n)
    x_opt = A \ b
    results = iterstats[]
    for i = 1 : maxiter 
        err = x_opt - x_sol
        res = b - A * x_sol
        if norm(res) < tol
            break
        end
        push!(results, iterstats(i, deepcopy(x_sol), err, res)) 
        Gauss_Seidel_Iteration!(x_sol, A, b)
    end
    return results
end

end # module end