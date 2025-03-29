import numpy as np
import matplotlib.pyplot as plt
dt = 0.1
T = 200
time = np.arange(0, T, dt)
num_neurons = 10
tau_m = 10
V_rest = -65
V_threshold = -50
V_reset = -70
mean_I = 15
std_I = 5
def lif_neuron_simulation(num_neurons, dt, T, tau_m, V_rest, V_threshold, V_reset, mean_I, std_I):
    time = np.arange(0, T, dt)
    V = np.full((num_neurons, len(time)), V_rest)
    spikes = [[] for _ in range(num_neurons)]
    for t in range(1, len(time)):
        I = np.random.normal(mean_I, std_I, num_neurons)
        dV = (-(V[:, t-1] - V_rest) + I) * (dt / tau_m)
        V[:, t] = V[:, t-1] + dV
        for n in range(num_neurons):
            if V[n, t] >= V_threshold:
                V[n, t] = V_reset
                spikes[n].append(time[t])
    return time, V, spikes
time, V, spikes = lif_neuron_simulation(num_neurons, dt, T, tau_m, V_rest, V_threshold, V_reset, mean_I, std_I)
plt.figure(figsize=(10, 6))
for n in range(min(num_neurons, 5)):
    plt.plot(time, V[n], label=f"Neuron {n+1}")
plt.axhline(V_threshold, linestyle="--", color="gray", label="Threshold", linewidth=1.2)
plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("Membrane Potential (mV)", fontsize=12)
plt.title("LIF Neuron Simulation with Stochastic Input", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.figure(figsize=(10, 5))
for n, neuron_spikes in enumerate(spikes):
    plt.scatter(neuron_spikes, [n] * len(neuron_spikes), color="red", marker="|", s=100)
plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("Neuron Index", fontsize=12)
plt.title("Raster Plot of Neuron Spiking Activity", fontsize=14)
plt.grid(alpha=0.3)
plt.show()

