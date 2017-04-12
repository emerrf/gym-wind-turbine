# gym-wind-turbine
OpenAI Gym environment which reproduces the behaviour of a wind turbine realistically using CCBlade aeroelastic code

## Introduction
The _gym-wind-turbine_ is a [OpenAI Gym](https://gym.openai.com/) environment which reproduces the behaviour of a wind turbine realistically. Behind the  scenes, the environment interfaces with the aerolastic code called [CCBlade](https://github.com/WISDEM/CCBlade)
 in order to compute the aerodynamic forces. Then, a simplified model of the drivetrain is added so that a driver can be implemented. The aim is to create a controller using reinforcement learning algorithms which can maximise the energy produced (power) and minimise the unwanted forces (thrust). The progress of the learning process can be also monitored by plotting metrics in real time.
 
 Here is an example of the real control in action which can be used as a reference.
 
 [Plots](https://vimeo.com/212414445)
 
 ### Context
 
 CCBlade<sup>[1](#ref1)</sup> performs accurate and fast aerodynamic analysis using Blade Element Momentum Theory ([BEMT](http://wisdem.github.io/CCBlade/theory.html)). The code used the NREL 5-MW Reference Wind Turbine model for its tests and I found very appropriate to use the same in the gym environment. I highly recommend to checkout the [technical details](www.nrel.gov/docs/fy09osti/38060.pdf
) as the document also contains a description of the real control (conventional variable-speed, variable blade-pitch-to-feather configuration) and its source code. If that is too dense, I'll try to explain it as a statistician would do in pub having some drinks.

The individual control of a wind turbine can perform 2 main actions, increase (decrease) the generator torque and increase (decrease) the blade pitch.
 
Generator torque can be seen as the force required to move the generator shaft. If the value is too low for a given wind speed, the blades will spin fast as the generator is not opposing resistance. If contrary, the value is too high for a given wind speed, the blades will barely move as the wind does not have enough power to overcome the resistance imposed by the generator. In both cases the energy produced will be very low. As you can see, the generator torque acts as a "break" which regulates the speed of the blades spinning (rotor speed). So the main question here is _what torque should be applied so that the blades rotate at the optimal speed which generates the maximum energy given the current wind?_ That's the job of the learning algorithm! 
 
The good news is that exists an optimal rotor speed which captures the maximum power given the wind speed and the aerodynamic configuration. The curve is usually plotted as C<sub>p</sub> vs Tip Speed Ratio and the real control has those optimal values "hard-coded" ([more details](http://www.nrel.gov/docs/fy04osti/36265.pdf))

But what happens if the wind is so strong that the generator torque is not able to "deal" with it? Then the pitch control comes into action. The pitch is the angle of the blades. Assume the wind comes perpendicular to the virtual disc formed by the blades. If the pitch is 0 degrees, the widest section of the blade (chord) is facing to the wind so the maximum power is captured. If instead, the pitch is 90 degrees, the narrowest section of the blade is facing to the wind like the wings of a plane. Thus, the pitch helps to regulate how much power you want to capture from the wind. If it is too strong for your generator or structure, increasing the pitch angle will let the air go through the blades by offering less resistance (thrust).

  
That is basically it, the control should use generator torque and pitch angle to get the maximum power from the wind. Remember that I tried to find a balance between a simple but yet realistic environment so that the results obtained could be an approximation of the results expected from more extensive aerolastic codes such as [FAST](https://nwtc.nrel.gov/FAST).

For those unfamiliar with OpenAI Gym environments, there is excellent [documentation](https://gym.openai.com/docs) in their site.

## Getting started
Instructions about how to install the package and run the examples is exposed next
### Installation
The package has been coded and tested on a Linux environment. It requires Python 2.7 and the Fortran compiler.
```
$ virtualenv gwt-devenv
(gwt-devenv)$ pip install numpy==1.12.1
(gwt-devenv)$ pip install -r requirements.txt
```
### Run the samples
Ensure all the pieces are working correctly:
```
(gwt-devenv)$ ccblade
(gwt-devenv)$ gwt-run-real-control
```
### Acknoledgments

I created this environment for learning purposes, from one side it satisfies my curiosity in understanding how wind turbines work and how they are controlled and also my interest in learning AI techniques. I thought that opening this challenge to the AI community could bring interesting results, and who knows if we can eventually improve existing drivers with them.  

Special thanks to @andrewning for creating the CCBlade code and the NREL Wind team in general for opening their work to the community. Also thanks to the OpenAI for starting the Gym initiative and contributing to democratise the AI. Thanks to @cvillescas for supporting and helping me to understand the theory. 

### References

<a name="ref1">[1]</a>: S. Andrew Ning. A simple solution method for the blade element momentum equations with guaranteed convergence. Wind Energy, June 2013. doi:10.1002/we.1636.

<a name="ref1">[2]</a>: Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, Wojciech Zaremba. OpenAI Gym. June 2016. arXiv:1606.01540.
