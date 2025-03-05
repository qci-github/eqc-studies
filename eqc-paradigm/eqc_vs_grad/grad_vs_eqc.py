import zipfile
import os
import json

import jax
import jax.numpy as jnp
from jax import grad, jit, lax
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import optuna
from optuna.samplers import TPESampler

# set plot background color
plt.rcParams['figure.facecolor'] = "#F6F3ED"


@jit
def calc_energy(soln, linear_terms, quad_terms):
    """Calculate energy for solution"""
    linear_terms = jnp.array(linear_terms)
    quad_terms = jnp.array(quad_terms)

    linear_energy = jnp.dot(linear_terms, soln)
    quad_energy = jnp.dot(soln, jnp.dot(quad_terms, soln))
    energy = linear_energy + quad_energy
    return energy

def project_to_simplex(x, C):
    """ Projects x onto the simplex sum(x) = C, x >= 0 """
    x = jnp.array(x)
    n = x.shape[0]
    x_sorted = jnp.sort(x)[::-1]
    x_cumsum = jnp.cumsum(x_sorted)

    # Compute valid rho values
    rho_candidates = jnp.arange(1, n + 1)
    tau_candidates = (x_cumsum - C) / rho_candidates
    mask = x_sorted - tau_candidates >= 0
    rho = jnp.sum(mask) - 1
    
    tau = (x_cumsum[rho] - C) / (rho + 1)
    return jnp.maximum(x - tau, 0)

@jit
def gradient_descent(linear_terms, quad_terms, x0, C, lr=0.01, max_iter=2000):
    """ Gradient descent with projection to sum constraint and non-negativity """
    # apply simplex projection
    x = project_to_simplex(x0, C)
    # apply auto grad
    grad_f = grad(lambda x: calc_energy(x, linear_terms, quad_terms))
    # create energy history vec
    energy_history = jnp.zeros(max_iter)

    def cond_fn(state):
        x, i, energy_history = state
        return i < max_iter  # Stop if max_iter is reached

    def body_fn(state):
        x, i, energy_history = state  # Unpack all three elements of the state
        grad_x = grad_f(x)
        x_new = project_to_simplex(x - lr * grad_x, C)  # Always project
        energy = calc_energy(x_new, linear_terms, quad_terms)  # Calculate energy

        # Update energy history
        energy_history = lax.dynamic_update_slice(energy_history, jnp.array([energy]), (i,))
        return x_new, i + 1, energy_history

    x, _, energy_history = lax.while_loop(cond_fn, body_fn, (x, 0, energy_history))  # JAX loop
    
    return x, energy_history, energy_history[-1]

def objective(trial):
    # Search in log scale
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # Run gradient descent
    energies_trial = []
    for i in range(500):
        # start at a random point
        x0 = jnp.array(np.random.rand(len(linear_terms)))
        _, _, final_energy = gradient_descent(linear_terms, quad_terms, x0, C, lr=lr)
        energies_trial.append(float(final_energy))
    if np.mean(energies_trial)<=-6.38:
        print(energies_trial)
    # learning rate minimized based on mean energy of 500 runs
    return np.mean(energies_trial)


if __name__ == "__main__":
    # set seeds for reproducibility
    sampler = TPESampler(seed=13)
    np.random.seed(13)
    # extract eqc data into current directory from zip file
    zip_file_path = 'eqc_qplib18_500_runs.zip'
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(".")

    # Load QPLIB problem data
    C = 1.0  # Constraint sum(x) = C
    qplib_18 = np.loadtxt('QPLIB_0018_OBJ.csv', delimiter=',')
    # linear terms are in first columns
    linear_terms = jnp.array(qplib_18[:,0])
    # quadratic terms all other data besides first column
    quad_terms = jnp.array(qplib_18[:,1:])
    counter = 0

    # run Tree Parzen estimator to determine best learning rate for grad descent
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=50)  # Adjust n_trials as needed
    # Get best learning rate
    best_lr = study.best_params["lr"]
    print(f"Optimal learning rate: {best_lr}")

    plt.figure(figsize=(8, 6))
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.title("Convergence of Dirac-3 vs Gradient Descent")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Run gradient descent with best learning rate from study
    iteration = list(range(10000))
    energies_gd = []
    for i in range(500):
        x0 = jnp.array(np.random.rand(len(linear_terms)))
        _, energy_history, best_energy = gradient_descent(linear_terms, quad_terms, x0, C, lr=best_lr)
        if best_energy <= -6.38:
            counter += 1
        energies_gd.append(best_energy)

        # Plot the energy history for each run of gradient descent
        plt.plot(iteration[:len(energy_history)], energy_history, color='red', alpha=0.04)

    # Add invisible line so labels show up for series in legend only once
    plt.plot([], [], color='red', alpha=1, label="Gradient Descent")
    plt.plot([], [], color='blue', alpha=1, label="Dirac-3")

    # plot dirac evolution 
    dirac_energies = []
    iteration = list(range(2000))
    for i in range(5):
        with open(f"data/results_500_sched_2_{i}.json", 'r') as fr:
            x = json.load(fr)
        dirac_energies.extend(x["energy"])
        for j in x["energy_evolution"]:
            # Plot the energy history for Dirac-3 with the desired opacity
            plt.plot(iteration, j, color='blue', alpha=0.04)

    # Add inlaid legend inside the plot area
    plt.legend(loc="upper left", bbox_to_anchor=(0.7, 0.8), fontsize=10)
    ax = plt.gca()
    ax.set_facecolor('#F6F3ED')
    plt.gcf().savefig("grad_vs_dirac.svg", format="svg")

    # Show the plot
    plt.show()

    print(f"Gradient descent solutions below threshold: {counter}")

    # plot coefficient evolution

    # create custom color map for plot
    colors = ["#5090C9", "#97C272", "#F4D685"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    # 100 is to show only first 100 iterations and 50 is the number of coefficients
    x_dim, y_dim = 100, 50 

    x_vals = np.linspace(0, x_dim, x_dim)
    y_vals = np.linspace(0, 50, y_dim)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    Z = np.array(x["state_vector"][0][0:x_dim])

    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = np.zeros_like(x_flat)  # Base of bars
    dz_flat = Z.flatten()

    dx = (x_vals[-1] - x_vals[0]) / x_dim * 0.9  
    dy = (y_vals[-1] - y_vals[0]) / y_dim * 0.9 

    # Normalize Z values to [0, 1] for colormap
    norm = Normalize(vmin=np.min(dz_flat), vmax=np.max(dz_flat))
    colors = custom_cmap(norm(dz_flat))

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    background_color = "#F6F3ED"
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    ax.bar3d(x_flat, y_flat, z_flat, dx, dy, dz_flat, 
             color=colors, alpha = 0.5)

    ax.view_init(elev=30, azim=200)

    ax.set_box_aspect(None, zoom=0.85)
    # Labels
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel('$i$',fontsize=10)
    ax.set_zlabel('$x_i$', fontsize=10, rotation = 180)

    fig.savefig("coefficients_iterations.svg", format="svg", dpi=300)
    plt.show()

    # Create energy histogram for grad vs eqc

    # create single set of bins for both histograms so compared on same basis
    bin_start = np.min([dirac_energies, energies_gd])
    bin_end = np.max([dirac_energies, energies_gd])
    bin_size = 0.05
    bins = np.arange(bin_start, bin_end + bin_size, bin_size)

    fig, ax = plt.subplots(figsize=(8, 6))
    background_color = "#F6F3ED"  # You can change this to match your preferred color
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.hist(energies_gd, bins=bins, alpha=0.5, label="Gradient Descent", color="red", edgecolor="black")
    ax.hist(dirac_energies, bins=bins, alpha=0.5, label="Dirac-3", color="blue", edgecolor="black")
    ax.set_xlabel("Energy", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(loc="upper right", frameon=True, facecolor="#F6F3ED", edgecolor="black")
    plt.tight_layout()

    fig.savefig("dirac_vs_grad_energy.svg", format="svg", dpi=300)
    plt.show()
