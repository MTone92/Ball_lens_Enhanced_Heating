# Ball_lens_Enhanced_Heating
Calculate the temperature distribution in a silicon substrate heated by a pulsed Bessel beam with or without a microdroplet falling down.
## Model
The pulsed Bessel beam is used to heat a silicon substrate. part of the laser energe gets reflected by the ratio *reflectance* and the rest energe is absorbed by the silicon substrate on the surface. The heat conduction model in Cylindrical coordinates is applied to solve the temperature distribution with homogeneous boundary condition on all boundaries except on the surface heated by laser. For the surface heated by laser, the nonhomogeneous, direct heat flux boundary condition is applied. When considering a droplet falling down, the laser beam gets defocused or refocused on the silicon substrate causing decreased or enhanced local laser heating. The reflectance, absorption coefficients and refractive index of the droplet also affect the rate of decreased or enhanced local laser heating.
## Run codes
To get the transient and spatial temperature distribution with direct laser heating without droplet, one can put ```Module MyFunctions.jl``` and ```TemperatureWithoutBallLens.jl``` in one folder and run ```TemperatureWithoutBallLens.jl``` directly.
To get the transient and spatial temperature distribution with droplet which induces decreased or enhanced local heating, one can put ```Module MyFunctions.jl``` and ```TemperatureByBallLens.jl``` in one folder and run ```TemperatureByBallLens.jl``` directly.

Some modified codes to calculate temperature at one place with finner time steps or in a zoom-in area at a singler time point or several time points are put in folder **Code_Extension**. Once can review these codes to pick up the most relevant one for their own applications. To run the modified code, one can put ```Module MyFunctions.jl``` and the modified code (```TemperatureWithoutBallLens_***.jl``` or ```TemperatureByBallLens_***.jl```) in one folder and run the modified code directly.
