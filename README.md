# Gating Neural Networks (GNN)

This repository is a collection of codes used in the paper, titled *Discovery of the Hidden State in Ionic Models Using a Domain-Specific
Recurrent Neural Network*. 

## Abstract

Ionic models, the set of ordinary differential equations (ODEs) describing the time evolution of the state of excitable cells, 
are the cornerstone of modeling in neuro-and cardiac electrophysiology. Modern ionic models can have tens of state variables 
and hundreds of tunable parameters. On the other hand, available experimental data usually is limited to a subset of observable 
state variables. Fine-tuning ionic models based on experimental data remains a challenging problem. In this paper, we describe a 
recurrent neural network architecture designed specifically to encode ionic models. The core of the model is a Gating Neural 
Network (GNN) layer, capturing the dynamics of classic (Hodgkin-Huxley) gating variables. The network is trained in two steps: 
first, it learns the theoretical model coded in a set of ODEs, and second, and second, it is retrained on observables. The 
retrained network is interpretable, and its results can be incorporated back into the model ODEs. We tested the GNN networks 
using simulated ventricular action potential signals and showed that it can deduce physiologically-feasible alterations of 
ionic currents. Therefore, it is reasonable to use such domain-specific neural networks in the exploratory phase of data 
assimilation before further fine-tuning using standard optimization techniques.

## How to Run

Data generation and visualization are performed separately. 

To generate the test data used in the paper, 

```julia
  include("generate.jl")
  
  generate_all()
```
This function will create a new directory (**models**) and generates the model data. **Warning!** ```generate_all``` run can take hours. 
On my Ubuntu workstation (Intel Core i7, GPU GTX 1080), the run lasts more than 12 hours. 

After the data is prepared, we can create the figures used in the paper (except Figure 1, which is a schematic) as 

```julia

include("paper_plots.jl")

# Figure 2
plot_signal_base()

# Figure 3
plot_signal_perturbed()

# Figure 4
plot_neural_ode()

# Figure 5
plot_currents_longqt(; η=0.0015)

# Figure 6
plot_currents_shortqt(; η=0.0015)

# Figure 7
plot_currents_ito(; η=0.0003)

```
