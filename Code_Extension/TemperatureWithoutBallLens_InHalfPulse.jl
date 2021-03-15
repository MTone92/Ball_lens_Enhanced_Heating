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
using DelimitedFiles
using DataFrames
using CSV

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
Ref = 0                         # Reflectance of the droplet (Ag20:0.06, Ge10:0.35, Ge05:0.42, SDS:0.48)
γ = 0                           # Absorption coefficient of the droplet (Ag20:10000, Ge10:770, Ge05:417, SDS:20)
n1 = 1                          # Refractive index of the droplet (Ag:1.377, Ge10:1.358, Ge05:1.34, SDS: 1.33)

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
v = 2                           # Speed of the droplet falling down
L_0 = 500e-6                    # Initial distance of the droplet bottom to Si substrate
t_total = L_0/v                 # Time elapse for droplet reaching Si substrate
Depth = 0.4*10^(-3)             # Depth of the Si substrate
R_Si = 1e-3                     # Radius where temperature is ambient
T_amb = 25                      # Ambient temperature

# Find the laser distribution on Si surface
x = [0:R_drop/1000:R_drop;]         # horizontal linear spacing
# x[1] = x[2]/100                     # Avoid sigularity
I = I0.(c,kr,x)
rr = 0.5 .* (x[1:end-1] + x[2:end])
drr = abs.(x[2:end] - x[1:end-1])
Irrdrr = 0.5 .* (I[1:end-1] + I[2:end]) .* rr .* drr

# Build up the calculation region:
r = [0:R_drop/100:R_drop; R_drop+R_drop/10 : R_drop/10 : 10*R_drop]
z = [0:Depth/100:Depth;]
r,z = repeat(r',length(z)), repeat(z,1,length(r))
# Find out the eigen values in z&r-direction:
β = Findβ(Depth,Bi)
η = Findη(R_Si)
N = length(β)
M = length(η)

# Find out the temperature distribution:
t = 85e-9

Σm = repeat(r,1,1,M)
fill!(Σm,0)
Σn = repeat(z,1,1,N)
fill!(Σn,0)
ΣQ = zeros(M,N)

for i = 1:1
    @threads for η_ind = 1:M
        η_m = η[η_ind]
        SI = 0
        for rr_ind = 1:length(rr)
            SI = SI + besselj0(η_m.*rr[rr_ind]).*Irrdrr[rr_ind]
        end
        Σm[:,:,η_ind] = besselj0.(η_m.*r).*InverseM(R_Si,η_m).*SI
    end

    @threads for β_ind = 1:N
        β_n = β[β_ind]
        Σn[:,:,β_ind] = Z.(Bi,β_n,z).*β_n.*InverseN(Depth,Bi,β_n)
    end

end

T = @distributed (+) for η_ind = 1:M
    
    Sn = 0
    for β_ind = 1:N
        λ = α*(η[η_ind]^2 + β[β_ind]^2)
        Sn = Sn .+ Σn[:,:,β_ind]*4e8/17*(t/λ - (1-exp(-λ*t))/λ^2)
        #Sn = Sn .+ Σn[:,:,β_ind]*(1-exp(-λ*t))/λ
    end

    Σm[:,:,η_ind].*Sn
end

T = (α/k).*T .+ T_amb

open("T.txt","w")
writedlm("T.txt",T)
T1 = convert(DataFrame,T)
CSV.write("T.csv",T1)

EndTime = Dates.now()
ΔTime = EndTime - StartTime
ΔTime = convert(Dates.Nanosecond,Dates.Millisecond(ΔTime))
ΔTime = Dates.Time(ΔTime)

# Write out frames:
open("r.txt","w")
writedlm("r.txt",r)
r1 = convert(DataFrame,r)
CSV.write("r.csv",r1)

open("z.txt","w")
writedlm("z.txt",z)
z1 = convert(DataFrame,z)
CSV.write("z.csv",z1)

open("Parameters.txt","w")
write("Parameters.txt",
    "P = $P\n","h = $h\n",
    "R_laser = $R_laser\n",
    "R_Si = $R_Si\n",
    "t = $t\n",
    "M = $M\n","N = $N\n",
    "ΔTime = $ΔTime")