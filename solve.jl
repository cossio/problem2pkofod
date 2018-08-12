module Prob

#= common =#

struct Workspace{U<:Real}
	lz::Matrix{U}; lZ::Vector{U}; lzmax::Vector{U}
	g::Vector{U}; h::Vector{U}
end
Workspace{U}(S::Integer,T::Integer) where {U<:Real} = Workspace{U}(zeros(U,S,T),zeros(U,T),zeros(U,T),zeros(U,S),zeros(U,S))
Workspace(S::Integer,T::Integer) = Workspace{Float64}(S,T)

function update_workspace!(w::Workspace{U},b,c,p::SparseMatrixCSC,x::AbstractVector{U}) where {U<:Real}
	S,T = size(w.lz); N = length(x)
    @assert length(b) == T && size(p) == (S,N) && size(c) == (S,T)
	A_mul_B!(w.h,p,x)
	w.lz .= c .+ w.h
    for t = 1:T
        w.lzmax[t] = -Inf
        for s=1:S
            w.lzmax[t] = max(w.lzmax[t], w.lz[s,t])
        end
        Z = zero(U)
        if isfinite(w.lzmax[t])
            for s = 1:S
                Z += exp(w.lz[s,t] - w.lzmax[t])
            end
            w.lZ[t] = log(Z) + w.lzmax[t]
        else
            w.lZ[t] = w.lzmax[t]
        end
	end
	for s = 1:S
        w.g[s] = zero(U)
        for t = 1:T
            if isfinite(w.lZ[t])
                w.g[s] += b[t] * exp(w.lz[s,t] - w.lZ[t])
            end
        end
	end
end

function F(x::AbstractVector{U},b,a,w::Workspace{U}) where {U<:Real}
	@assert length(a) == length(x) && length(b) == length(w.lZ)
	return dot(b,w.lZ) - dot(a,x)
end

function grad!(G,x::AbstractVector{U},p::SparseMatrixCSC,a,w::Workspace{U}) where {U<:Real}
	@assert length(x) == length(a) == size(p,2) == length(G)
    @assert size(p,1) == length(w.h) 
	A_mul_B!(G,p',w.g)
	G .-= a
	return nothing
end

#= Optim.jl =#
import Optim

function solveWithOptim(b,c,p,a,alg::Union{Optim.ZerothOrderOptimizer,Optim.FirstOrderOptimizer}, opts::Optim.Options)
	S,T = size(c); N = length(a)
	@assert length(b) == T && size(p) == (S,N)
	@assert all(x -> 0≤x<Inf, b) && all(x -> 0≤x<Inf, p) && all(x -> 0≤x<Inf, a)

	w = Workspace(S,T)

	function fg!(O,G,x)
        update_workspace!(w,b,c,p,x)
        if G != nothing
            grad!(G,x,p,a,w)
        end
        if O != nothing
            return F(x,b,a,w)
        end
        return nothing
    end

    result = Optim.optimize(Optim.only_fg!(fg!),zeros(N),alg,opts)
    @show result
    return Optim.minimizer(result)
end

#= NLopt =#
import NLopt

function solveWithNLopt(b,c,p,a, alg::Symbol; 
                        xatol::Real=1e-10, xrtol::Real=1e-10, fatol::Real=1e-10, frtol::Real=1e-10, 
                        maxeval::Integer=10000, verbose::Bool=false)
	S,T = size(c); N = length(a)
	@assert length(b) == T && size(p) == (S,N)
	@assert all(x -> 0≤x<Inf, b) && all(x -> x<Inf, c) && all(x -> 0≤x<Inf, p) && all(x -> 0≤x<Inf, a)
	@assert xatol ≥ 0 && xrtol ≥ 0 && fatol ≥ 0 && frtol ≥ 0 && maxeval	≥ 0

	w = Workspace(S,T)

    function f!(x,G)
        update_workspace!(w,b,c,p,x)
        if G ≠ nothing
            grad!(G,x,p,a,w)
        end
        obj = F(x,b,a,w)
        if verbose
            if G ≠ nothing
                println("obj = ", obj, "; gradient computed")
            else
                println("obj = ", obj)
            end
        end
        return obj
    end

    opt = NLopt.Opt(alg,N)
    NLopt.min_objective!(opt, f!)
    NLopt.xtol_abs!(opt, xatol)
    NLopt.xtol_rel!(opt, xrtol)
    NLopt.ftol_abs!(opt, fatol)
    NLopt.ftol_rel!(opt, frtol)
    NLopt.maxeval!(opt, maxeval)

    x = zeros(N)
	optf, optx, status = NLopt.optimize!(opt, x)
    verbose && @show optf, status
    return optx
end

end # module Prob
