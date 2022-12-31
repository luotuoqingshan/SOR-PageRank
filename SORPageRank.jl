"""
Helpful routines to investigate PageRank with an undirected graph via
the SOR algorithm.

SOR(omega, A,alpha,v) # start from x0, give an iterator for the SOR iterate
solution(A,alpha,v) # return the solution

optimal_omega(alpha) # return the "conjectured" optimal omega
property_a_matrix(alpha,omega) # return the little matrix for property a

This code is not designed to be efficient for 100M edge graphs. It's designed
for 100k edge graphs where "dense" linear-algebra/ARPACK/etc should
run. If needed, it could be made much faster.
"""
module SORPageRank
using MatrixNetworks, SparseArrays, LinearAlgebra, Arpack, LinearMaps

# handle setting up the linear system.
function _set_b!(b::Vector, v::Int)
  b[v] = 1
end
function _set_b!(b::Vector, v::Set)
  for i in v
    b[i] = 1.0/length(v)
  end
end
function _set_b!(b::Vector, v::Vector)
  @assert(minimum(v) >= 0,"v is not non-negative")
  b += v
end

function SOR(omega,maxiter::Int,A::SparseMatrixCSC,alpha,v)
  M,N,T = splitting(omega,A,alpha)
  b = zeros(size(M,1))
  _set_b!(b, v)
  b .*= (1-alpha)
  return SORIterator(M,N,b,maxiter)
end

struct SORIterator
  M
  N
  b::Vector{Float64}
  maxiter::Int
end

residual(x::Vector,I::SORIterator) = I.b - I.M*x + I.N*x

import Base.iterate, Base.length, Base.eltype
Base.iterate(I::SORIterator, state=(1,zeros(size(I.M,1)))) =
  state[1] > I.maxiter ? nothing : (state[2], (state[1]+1,I.M\(I.N*state[2] + I.b)))
Base.length(I::SORIterator) = I.maxiter
Base.eltype(::Type{SORIterator}) = Vector{Float64}

using IterTools

solution(A,alpha,v) = seeded_pagerank(A,alpha,v)

function error_vectors(omega::Real,maxiter::Int,A,alpha,v)
  x = solution(A,alpha,v)
  return imap(xk -> x - xk, SOR(omega,maxiter,A,alpha,v))
end

""" Produce the solution, residual, and error vectors. """
function allvectors(omega::Real,maxiter::Int,A,alpha,v)
  x = solution(A,alpha,v)
  I = SOR(omega,maxiter,A,alpha,v)
  return imap(xk -> (iterate=xk, residual=residual(xk, I), error=x-xk), I)
end

function _validate(A::SparseMatrixCSC)
  okay::Bool = true
  if !issymmetric(A)
    return false
  end
  if norm(diag(A)) > 0
    return false
  end
  if any(nonzeros(A) .!= 1)
    return false
  end
  return true
end

function _build_Pt(A::SparseMatrixCSC)
  d = vec(sum(A;dims=2))
  AI,AJ,AV = findnz(A)
  Pt = sparse(AI,AJ,AV./d[AI],size(A)...)
end

function splitting(omega::Real,A::SparseMatrixCSC,alpha::Real)
  @assert(_validate(A),"symmetric, diagonal, or all 1 tests failed")
  Pt = _build_Pt(A)
  M = (1/omega)*I - LowerTriangular(alpha* Pt')
  N = ((1/omega) - 1)*I +UpperTriangular(alpha* Pt')
  Mt = M'
  rmul = u -> M\u
  lmul = u -> Mt\u
  invM = LinearMap(rmul,lmul,size(A)...)
  T = invM*LinearMap(N)
  return (M,N,T)
end

function spectral_range(omega::Real, A::SparseMatrixCSC, alpha::Real; kwargs...)
  M,N,T = splitting(omega, A, alpha)

  Nt = N'
  rmul = u -> N\u
  lmul = u -> Nt\u
  invN = LinearMap(rmul,lmul,size(A)...)
  iT = invN*LinearMap(M)

  maxlams, maxvecs = eigs(T, which=:LM; kwargs...)
  minlams, minvecs = eigs(iT, which=:LM; kwargs...)

  lams = hcat(maxlams,1.0./(minlams))
  vecs = hcat(maxvecs,minvecs)
  return extrema(abs,lams), abs.(lams), lams, vecs
end

function spectral_radius(omega::Real, A::SparseMatrixCSC, alpha::Real;
     tests::Bool=true, kwargs...)
  M,N,T = splitting(omega, A, alpha)
  lams, vecs = eigs(T, which=:LM; kwargs...)
  return maximum(abs,lams), abs.(lams), lams, vecs
end

optimal_omega(alpha) = 1 + (alpha/(1+sqrt(1-alpha^2)))^2

""" This is the matrix that arises when you study something with property A. """
property_a_matrix(omega,alpha) = [0 omega*alpha; omega^2*(1/omega - 1)*alpha omega^2*alpha^2]+(1-omega)*I

""" Return the Perron vectors for the given omega. """
function perron_vector(omega,A,alpha; nev=nothing, kwargs...)
  # using a sentinal check on nev to see if they set it.
  if nev != nothing
    @assert(nev == 1,"nev must be 1 for perron vector computation")
  end
  p = spectral_radius(omega, A, alpha;
    nev=1, tol=1e-4, maxiter=4000, kwargs...)[4]
  p .*= sign(p[1]) # normalize sign
  # TODO try and minimize small complex entries...
  return p
end

""" Return the Perron vectors for a list of given omegas values... """
perron_vectors(omegas, A, alpha; kwargs...) =
  hcat(map(omega -> perron_vector(omega,A,alpha; kwargs...), omegas)...)


end # end module

# Light testing
using Test, LinearAlgebra, SparseArrays, IterTools
function _sor_tests(A,alpha,v)
  @testset "iterative" begin
    ## Check that residual correspones to A*error
    T = Matrix(SORPageRank.splitting(1.3,A,alpha)[3])
    sorprob = (1.3, # omega
               10, # maxiter
               A,alpha,v) # pagerank
    iter = SORPageRank.SOR(sorprob...)
    for (i,vecs) = enumerate(SORPageRank.allvectors(sorprob...))
      @test norm(vecs.residual  - (iter.M*vecs.error - iter.N*vecs.error),1) <= 10*(1/(1-alpha))*eps(typeof(alpha))
    end

    ## Check that error corresponds to T^k*solution
    T = Matrix(SORPageRank.splitting(1.3,A,alpha)[3])
    for (i,ek) = enumerate(SORPageRank.error_vectors(1.3,10,A,alpha,v))
      @test norm(ek  - T^(i-1)*SORPageRank.solution(A,alpha,v),1) <= 10*(1/(1-alpha))*eps(typeof(alpha))
    end
  end

  @testset "spectral ops" begin
    @test SORPageRank.spectral_radius(1.0,A,alpha)[1][1] < 1
    rad199 = SORPageRank.spectral_radius(1.99,A,alpha)[1][1]
    @test rad199 < 1 # spd, so we get convergence when this happens
    range = SORPageRank.spectral_range(1.99,A,alpha)[1]
    @test range[1] > 0
    @test range[2] < 1
    @test range[2] ≈ rad199

    p = SORPageRank.perron_vector(1.0,A,alpha)
    @test norm(imag.(p)) <= size(A,1)*eps(1.0)
    @test minimum(real.(p)) >= -eps(1.0)
  end
end



@testset "property A, biclique" begin
  A = sparse([0  0  0  0  0  1  1  1  1  1
              0  0  0  0  0  1  1  1  1  1
              0  0  0  0  0  1  1  1  1  1
              0  0  0  0  0  1  1  1  1  1
              0  0  0  0  0  1  1  1  1  1
              1  1  1  1  1  0  0  0  0  0
              1  1  1  1  1  0  0  0  0  0
              1  1  1  1  1  0  0  0  0  0
              1  1  1  1  1  0  0  0  0  0
              1  1  1  1  1  0  0  0  0  0])
  _sor_tests(A,0.85,1)

  A = sparse([0  0  0  0  1  1  1  1  1
              0  0  0  0  1  1  1  1  1
              0  0  0  0  1  1  1  1  1
              0  0  0  0  1  1  1  1  1
              1  1  1  1  0  0  0  0  0
              1  1  1  1  0  0  0  0  0
              1  1  1  1  0  0  0  0  0
              1  1  1  1  0  0  0  0  0
              1  1  1  1  0  0  0  0  0])
  _sor_tests(A,0.85,1)
  _sor_tests(A,0.99,1)
end

