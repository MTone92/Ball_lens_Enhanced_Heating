# This program is used to calculate the temperature distribution in Si substrate by laser passing a ball lens.
# Import the necessary packages before running the main program:

include("Module MyFunctions.jl")        # Include the file directory and file name for using the local modules
using .MyFunctions                      # The dot before the module name is used to indicate it is a local module
using SpecialFunctions
using LinearAlgebra
using SharedArrays
using Distributed
using Dates
using Base.Threads
using MAT

StartTime = Dates.now()
# Add the available cpus as workers:
addprocs()
# Check the threads' number:
N_t = nthreads()
N_w = nworkers()
println("Number of Threads = ",N_t)
println("Number of workers = ",N_w)
open("Number_of_Threads.txt","w")
write("Number_of_Threads.txt",
        "Number of Threads = ","$N_t\n",
        "Number of workers = ","$N_w")

# Define the most mutable variables:
P = 1                           # Laser average power is set to be unity.
Ref = 0.35                      # Reflectance of the droplet (Ag20:0.06, Ge10:0.35,  Si05:0.28, Ge05:0.42, SDS:0.48)
γ = 770                       # Absorption coefficient of the droplet (Ag20:10000, Ge10:770, Si05:625, Ge05:417, SDS:20)
n1 = 1.358                      # Refractive index of the droplet (Ag:1.377, Ge10:1.358, Si05:1.3478, Ge05:1.34, SDS: 1.33)

# Define the laser parameters:
R_laser = 0.15*10^(-3)       # Radius of the laser spot
frequency = 30000               # Laser repetition Rate
tp = 1/frequency                # Laser period
λ = 1064*10^(-9)                # Laser wavelength 1064 nm
pulse = 170*10^(-9)             # Laser width (pulse-on time)
kr = 2π/λ                       # Wave vector
P_pulse = P/frequency/pulse     # Average laser power in one pulse
c = P_pulse /
    (besselj0(kr*R_laser)^2 + besselj(1,kr*R_laser)^2) /
    (π*R_laser^2)               # Laser intensity at center in one pulse

# Define the heat transfer variables:
k = 119                         # Thermal conductivity of Si @ 350K (https://www.efunda.com/materials/elements/TC_Table.cfm?Element_ID=Si)
cp = 21283.62/28.0855           # Specific heat of Si @ 350K (https://webbook.nist.gov/cgi/inchi?ID=C7440213&Type=JANAFS&Plot=on#JANAFS)
ρ = 2330                        # Density of Si
α = k/(ρ*cp)                    # Thermal diffusivity of Si @ 350K (8.8*10^(-5))
h = 100                         # Convection heat transfer coefficient in air (h = 10~100 or 10~1000)
Bi = h/k                        # Bi number

# Define the droplet and substrate parameters
R_drop = 50*10^(-6)             # Radius of the droplets
R_Incidence = 10e-6             # Radius of the incidence laser for calculate NA
v = 2                           # Speed of the droplet falling down
L_0 = 500e-6                    # Initial distance of the droplet bottom to Si substrate
t_total = L_0/v                 # Time elapse for droplet reaching Si substrate
Depth = 0.4*10^(-3)             # Depth of the Si substrate
R_Si = 1e-3                     # Radius where temperature is ambient
T_amb = 25                      # Ambient temperature

# Find the laser distribution on Si surface
x = [0:R_drop/1000:R_drop;]         # horizontal linear spacing of the original laser profile

EFL = 2*n1*R_drop/(4*(n1-1))        # Effective Focal Length of Ball Lens.
BFL = EFL - R_drop                  # Back Focal Length of Ball Lens.
NA = 2*(n1-1)/n1*R_Incidence/R_drop # Numerical Aperture.
w0 = λ/4/NA                         # Focal spot size, radius.
zR = pi*w0^2*n1/λ                   # Rayleigh length.

# Build up the calculation region:
r_max = 1e-6
Δr = 10e-9
z_max = 1e-6
Δz = 10e-9
r = [0:Δr:r_max;]
z = [0:Δz:z_max;]
r,z = repeat(r',length(z)), repeat(z,1,length(r))
# Find out the eigen values in z&r-direction:
β = Findβ(Depth,Bi)
η = Findη(R_Si)
N = length(β)
M = length(η)

# Find out the temperature distribution:
t = 7*tp + pulse
Ii_raw = t*frequency
Ii = Int64(floor(Ii_raw))
if Ii_raw - Ii < pulse/2tp
    Ii = Ii - 1
end

Σm = repeat(r,1,1,M)
fill!(Σm,0)
Σn = repeat(z,1,1,N)
fill!(Σn,0)
ΣQ = zeros(M,N)
ΣI = zeros(M,Ii+1)

for j = 1:1
    @threads for η_ind = 1:M
        η_m = η[η_ind]
        Σm[:,:,η_ind] = besselj0.(η_m.*r).*InverseM(R_Si,η_m)
    end

    @threads for β_ind = 1:N
        β_n = β[β_ind]
        Σn[:,:,β_ind] = Z.(Bi,β_n,z).*β_n.*InverseN(Depth,Bi,β_n)
    end

    @threads for η_ind = 1:M
        for β_ind = 1:N
            ΣQ[η_ind,β_ind] = Q(α,η[η_ind],β[β_ind])
        end
    end

    @threads for η_ind = 1:M
        for i = 0:Ii
            τ_i = tp*i
            l = L_0 - v*τ_i
            w1 = w0*sqrt(1+((l-BFL)/zR).^2)
            factor = R_Incidence/w1
            xx = x./factor
            I = I0.(c*factor^2,kr*factor,xx).*(1-Ref).*exp(-γ*2R_drop)
            rr = 0.5 .* (xx[1:end-1] + xx[2:end])
            drr = abs.(xx[2:end] - xx[1:end-1])
            Irrdrr = 0.5 .* (I[1:end-1] + I[2:end]) .* rr .* drr
            SI = 0
            for rr_ind = 1:length(rr)
                SI = SI + besselj0(η[η_ind].*rr[rr_ind]).*Irrdrr[rr_ind]
            end
            ΣI[η_ind,i+1] = SI
        end
    end

end

T = @distributed (+) for η_ind = 1:M
    
    Sn = 0
    for β_ind = 1:N
        
        St = 0
        for i = 0:Ii
            τ_i = tp*i
            St = St .+ ΣI[η_ind,i+1].*exp(-α*(η[η_ind]^2+β[β_ind]^2)*(t-τ_i))
        end

        Sn = Sn .+ Σn[:,:,β_ind].* ΣQ[η_ind,β_ind] .* St
    end

    Σm[:,:,η_ind].*Sn
end

T = (α/k).*T .+ T_amb

# Find out the time elapse for running this whole program:
EndTime = Dates.now()
ΔTime = EndTime - StartTime
ΔTime = convert(Dates.Nanosecond,Dates.Millisecond(ΔTime))
ΔTime = Dates.Time(ΔTime)

# Write out other parameters:
open("Parameters.txt","w")
write("Parameters.txt",
    "Ref = $Ref\n", "γ = $γ\n", "n1 = $n1\n",
    "P = $P\n","h = $h\n",
    "R_laser = $R_laser\n",
    "R_Si = $R_Si\n",
    "M = $M\n","N = $N\n",
    "ΔTime = $ΔTime")

matwrite("T.mat",Dict("T" => T, "r" => r, "z" => z, "t" => t,
        "P" => P, "h" => h,
        "R_laser" => R_laser, "R_Si" => R_Si,
        "M" => M, "N" => N, "TimeElapse" => "$ΔTime"))