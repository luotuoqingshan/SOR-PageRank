using Arpack
using LinearAlgebra
using SparseArrays
using Printf
using MatrixNetworks
using Random
using MAT
include("SOR.jl")

##

verbose = false

for trials = 1:10000
     m = 10 
     n = 10 
     B = sprand(m, n, 0.7) .> 0.0
     A = [sparse(Matrix(0.0I, n, n)) B';
          B sparse(Matrix(0.0I, m, m))]

     A, p = largest_component(A)
     if size(A)[1] != n + m
          continue
     end

     alpha = 0.9
     seed = rand(1:size(A)[1]) 

     P, b = SOR.create_sym_ppr_ls(A, seed, alpha)

     #@show P, b

     mu, omega_opt = SOR.optim_omega(P)
     omega_opt = omega_opt * 0.99

     #results = SOR.SOR(P, b, omega_opt)
     #richardson_results = SOR.Richardson(P, b, 1.0)
     #jacobi_results = SOR.Jacobi(P, b; maxiter=100)
     #gauss_seidel_results = SOR.Gauss_Seidel(P, b)
     sor_results = SOR.SOR_solve(P, b, omega_opt ;tol=1e-8)
     sor_step_results = SOR.SOR_stepwise(P, b, omega_opt; maxiter=1000, tol=1e-8)

     #T_omega = 
     # [-0.5       0.0       0.225     0.225;
     #   0.0      -0.5      0.225     0.225;
     # -0.1125  -0.1125  -0.39875   0.10125;
     # -0.1125  -0.1125   0.10125  -0.39875;]

     C = -P[1:n, n+1:n+m] 
     #@show size(C)

     L = [sparse(Matrix(1.0I, n, n)) spzeros(n, m);
          -omega_opt * C' sparse(Matrix(1.0I, m, m))]
     U = [(omega_opt - 1) * sparse(Matrix(1.0I, n, n)) -omega_opt * C;
          spzeros(m, n) (omega_opt - 1) *sparse(Matrix(1.0I, m, m))]

     G = [(1 - omega_opt) * sparse(Matrix(1.0I, n, n)) omega_opt * C;
          (omega_opt - omega_opt^2) * C' omega_opt^2 * C' * C + (1 - omega_opt) * sparse(Matrix(1.0I, m, m))]

     #@show norm(-inv(Matrix(L)) * U - G)

     b_prime = inv(Matrix(L)) * omega_opt * b

     x_opt = P \ b
     #@show norm(omega_opt * P - L - U)
     #@show norm(L * x_opt - omega_opt * b + U * x_opt)

     #@show norm(x_opt - G * x_opt - b_prime)

     #@show x_opt 

     error_decrease = true
     error_pos = true
     for i = 1:length(sor_results)   
       if i == length(sor_results) 
          break 
       end
       iter_res = sor_results[i]
       iter_res_nxt = sor_results[i+1]
       #@printf("Iter: %d Error %.8e %.8e %.8e\n", iter_res.iterate, minimum(iter_res.error), norm(iter_res.error, 2), minimum(iter_res.residual))
       if verbose
          @printf("Iter: %d Error: %.8e\n", iter_res.iterate, minimum(iter_res.error))
       end
       if minimum(iter_res.error) < 0
          error_pos = false
       end
       if minimum(iter_res.error - iter_res_nxt.error) < 0
          error_decrease = false
       end
       x1 = iter_res.error[1:n]
       x2 = iter_res.error[n+1:n+m]
       #@printf("Diff: %.8e %.8e\n", minimum(x1 - C * x2), minimum(x2 - C' * x1))
     end

     step_error_pos = true
     for i = 1:length(sor_step_results)
          step_res = sor_step_results[i]
          j = step_res.index
          if step_res.error[j] < 0 
               step_error_pos = false
          end
          if verbose
               @printf("Error: Iteration %d, index %d, error: %.8e\n", step_res.iterate, step_res.index, step_res.error[j])
          end
     end

     if step_error_pos == false
          matwrite("data/badexample.mat", Dict("B" => B, "seed" => seed))
          @printf("Found a bad example!")
          break
     end
end
#for iter_res in sor_results
#  @printf("iter: %d %.8e %.8e %.8e\n", iter_res.iterate, minimum(iter_res.error), norm(iter_res.error, 2), minimum(iter_res.residual))
#end

##
dict = matread("data/badexample.mat")
B = dict["B"]
seed = dict["seed"] 

m = 1
n = 1

A = sparse([0 1; 1 0])

seed = 1
A, p = largest_component(A)

alpha = 0.9

P, b = SOR.create_sym_ppr_ls(A, seed, alpha)

#@show P, b

mu, omega_opt = SOR.optim_omega(P)
omega_opt = omega_opt * 0.99

sor_results = SOR.SOR_solve(P, b, 0.9 ;tol=1e-8)
sor_step_results = SOR.SOR_stepwise(P, b, 0.9; maxiter=100000, tol=1e-8)

for i = 1:n+m:length(sor_step_results)
     @assert norm(sor_step_results[i].error - sor_results[div(i - 1, n + m) + 1].error) == 0 
end

for i = 1:length(sor_results)
     @printf("Minimum Element of Error: %d %.8e\n", argmin(sor_results[i].error), minimum(sor_results[i].error))
end

@printf("\n\n\n\n")
for i = 1:length(sor_step_results)
     j = sor_step_results[i].index
     @printf("Minimum Element of Error: %d %.8e %.8e\n", j, sor_step_results[i].error[j], minimum(sor_step_results[i].error))
end

#@show length(sor_results)
#@show length(sor_step_results)
#
#C = -P[1:n, n+1:n+m] 
#     #@show size(C)
#L = [sparse(Matrix(1.0I, n, n)) spzeros(n, m);
#     -omega_opt * C' sparse(Matrix(1.0I, m, m))]
#U = [(omega_opt - 1) * sparse(Matrix(1.0I, n, n)) -omega_opt * C;
#     spzeros(m, n) (omega_opt - 1) *sparse(Matrix(1.0I, m, m))]
#
#G = [(1 - omega_opt) * sparse(Matrix(1.0I, n, n)) omega_opt * C;
#     (omega_opt - omega_opt^2) * C' omega_opt^2 * C' * C + (1 - omega_opt) * sparse(Matrix(1.0I, m, m))]
#
###
#lams, vecs = eigs(G, which=:LM; )
#
#vecs


##

include("SORPageRank.jl")

m = 1 
n = 1 
tests = 1000
alpha = 0.9
@show B
A = sparse([0 1;
     1 0])
@show A
omega = SORPageRank.optimal_omega(alpha)
@show omega
M, N, T = SORPageRank.splitting(0.9, A, alpha)

perron_v =  SORPageRank.perron_vector(0.9, A, alpha)

##
w = 0.1 
sigma = 0.8
wstar = 2 / (1 + sqrt(1 - sigma^2))

@show w

T = [1 - w   -w*sigma;
     (w^2 - w) * sigma (1 - w) + w^2* sigma^2;
     ]

evals, evecs = eigen(T)
z = w * sigma / (1 - w - evals[2])

R = [evals[2] -w^2 * sigma;
    0         evals[1]]

@show R

l = sqrt(1 + z^2)
Z = [z / l -1 / l;
     1 / l z / l] 

@show (z - 1) / (1 + sigma) - (z + 1) / (1 - sigma)
##

B = [1;]

A = [0 B;
     B 0]

alpha = sigma 
v = [1, 0]

Q = [-1 0;
     0  1]

xstar = (1 - alpha) * ((I - alpha * A) \ v) 

val = sqrt(2) / 2
Z1 = [-val -val;
      val -val]

Q1 = Q * Z
Q2 = Q * Z1

@show Q1' * xstar

@show R^100

@show R^100 * Q1' * xstar

Q1 * R^100 * Q1' * xstar


##
A = [0 1; 1 0]

LambdaPage, QPage = eigen(I - alpha * A)

QPage = -QPage

vv = sqrt(w^2 * sigma^2 - 4(w - 1))
s = w / sqrt(w^2 * sigma^2 - 4(w - 1))
zprime = 2 / (alpha * w + sqrt(w^2 * alpha^2 - 4(w - 1)))

@show 2 * vv
@show w * (alpha * w + vv)
@show ((2 - w) * vv)^2
@show w^2 * alpha
@show w^2 * alpha^2 - 4 * (w - 1)
@show (4 - 4 * w + w^2) * (w^2 * alpha^2 - 4 * w + 4)

@show 2(alpha - z) - 2 * s * (1 + alpha * z) 
@show zprime

@show z

@show s