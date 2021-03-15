module MyFunctions
using SpecialFunctions
using Roots
export I0, InverseN, InverseM, Z, Q, Findη, Findβ

function I0(c,kr,r)
    I0 = c*besselj0(kr*r)^2
    return I0
end

function InverseN(Depth,Bi,beta_n)
    f = 2/((beta_n^2 + Bi^2)*Depth + 2Bi)
    return f
end

function InverseM(R_Si,η_m)
    f = 2/(R_Si^2 * besselj(1,R_Si*η_m)^2)
    return f
end

function Z(Bi,beta_n,z)
    Z = beta_n*cos(beta_n*z) + Bi*sin(beta_n*z)
    return Z
end

# If assume the pulse is a step function:
#function Q(α,η_m,β_n)
#    τ = 170e-9
#    λ = α*(η_m^2+β_n^2)
#    Q = (exp(λ*τ)-1)/λ
#    return Q
#end

# If assume the pulse is a triangle function:
function Q(α,η_m,β_n)
    τ = 170e-9
    λ = α*(η_m^2+β_n^2)
    Q = (4/λ + 4e8/17*(1/λ^2 - τ/λ)) * (exp(λ*τ)-exp(λ*τ/2)) + 4e8/(17λ^2)*(1-exp(λ*τ/2))
    return Q
end

# function used to find out the roots in r-direction:
function Findη(R_Si)
    x = 2 .* [1:10000;]                 # For this setting, the first 6366 roots (0-20000) are continous
    eta_n = zeros(length(x))
    for i = 1:length(x)
        try
            eta_n[i] = find_zero(besselj0,x[i])
        catch
        end
    end
    eta_n = unique(eta_n)
    filter!(y->y>0&&y<x[end], eta_n)
    sort!(eta_n)
    eta_diff = [2; diff(eta_n)]
    deleteat!(eta_n, findall(x->x<1,eta_diff))
    return eta_n[1:1000]/R_Si
end

# function used to find out the roots for β_n, only positive values need to be considered:
function Findβ(L,Bi)
    myBC(x) = tan(L*x) - 2*Bi*x/(x^2-Bi^2)  # Define my boundary coditions
    period = π/L                            # Period of tan(Lx)

    # We need to determine the root finding interval so that we can find every possible root
    # by using less trials. Determination of deltax is tricky. It should be small to capture
    # all the roots but large to increase the speed.
    # For larger range of x, smaller deltax is needed.
    # (detalx <= period/10 is accurate except near the end.)
    deltax = period/4                      # Define the root finding interval
    x_end = period/4 + 1000*period         # Determine the roots' upperbound by setting number
                                            # of period (It's also the approximated number of roots)
    x0 = [period/4 : deltax : x_end;]      # Generate the interval array.
    Beta_n = zeros(length(x0))

    for ind = 1 : length(x0)
        try
            Beta_n[ind] = find_zero(myBC,x0[ind])
        catch
        end
    end
    Beta_n = abs.(unique(Beta_n))
    Values = abs.(myBC.(Beta_n))
    # Equivalent to: Values = broadcast(x -> abs(myBC(x)), Roots_unique)
    deleteat!(Beta_n, findall(x->x>0.001, Values))
    
    Beta_new = []
    while Beta_n ≠ []
        i = Beta_n[1]
        Beta_new = append!(Beta_new,[i])
        filter!(x->x-i>deltax,Beta_n)
    end
    
    return Beta_new[1:600]
end

end