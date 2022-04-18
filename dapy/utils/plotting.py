import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_rank_histogram(particles, reference, num_bin=None, figsize=(12, 3)):
    num_particle = particles.shape[1]
    if num_bin is None:
        num_bin = num_particle + 1
    combined = np.concatenate(
        [
            reference.flatten()[None],
            particles.swapaxes(0, 1).reshape((num_particle, -1)),
        ]
    )
    fig, ax = plt.subplots(figsize=figsize)
    ranks = np.apply_along_axis(rankdata, 0, combined)[0]
    _ = ax.hist(ranks - 1, num_bin, density=True)
    ax.plot(ranks - 1, np.ones_like(ranks) / (num_bin), "k:", lw=0.5)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Density")
    fig.tight_layout()
    return fig, ax


def plot_slices_through_time(
    results,
    space_indices,
    true_state_sequence=None,
    plot_particles=True,
    plot_region=True,
    particle_skip=1,
    trace_alpha=1.0,
):
    observation_time_indices = np.arange(results["state_mean_sequence"].shape[0])
    fig, axes = plt.subplots(
        nrows=len(space_indices),
        ncols=1,
        sharex=True,
        figsize=(12, len(space_indices) * 3),
    )
    for i, ax in zip(space_indices, axes):
        ax.plot(
            observation_time_indices,
            results["state_mean_sequence"][:, i],
            "g-",
            lw=1,
            label="Est. mean",
        )
        if plot_region:
            ax.fill_between(
                observation_time_indices,
                results["state_mean_sequence"][:, i]
                - 3 * results["state_std_sequence"][:, i],
                results["state_mean_sequence"][:, i]
                + 3 * results["state_std_sequence"][:, i],
                alpha=0.25,
                color="g",
                label="Est. mean ± 3 standard deviation",
            )
        if plot_particles:
            lines = ax.plot(
                observation_time_indices,
                results["state_particles_sequence"][:, ::particle_skip, i],
                "r-",
                lw=0.25,
                alpha=trace_alpha,
            )
            lines[0].set_label("Particles")
        if true_state_sequence is not None:
            ax.plot(
                observation_time_indices,
                true_state_sequence[:, i],
                "k--",
                label="Truth",
            )
        ax.set_ylabel(f"$x_{{{i},t}}$")
        ax.legend(loc="upper center", ncol=4)
    ax.set_xlabel("Time index $t$")
    fig.tight_layout()
    return fig, axes


def plot_slices_through_space(
    results,
    mesh_shape,
    space_indices,
    observation_time_indices,
    true_state_sequence=None,
    plot_particles=True,
    plot_region=True,
    particle_skip=1,
    trace_alpha=1.0,
):
    plot_space_indices = np.arange(mesh_shape[space_indices.index(slice(None))])
    space_subscript = ",".join(
        "s" if isinstance(i, slice) else str(i) for i in space_indices
    )
    num_time_step = results["state_mean_sequence"].shape[0]
    fig, axes = plt.subplots(
        nrows=len(observation_time_indices),
        ncols=1,
        sharex=True,
        figsize=(12, len(observation_time_indices) * 3),
    )
    mean = results["state_mean_sequence"].reshape((num_time_step,) + mesh_shape)
    std = results["state_std_sequence"].reshape((num_time_step,) + mesh_shape)
    particles = results["state_particles_sequence"].reshape(
        (num_time_step, -1) + mesh_shape
    )
    for t, ax in zip(observation_time_indices, axes):
        ax.plot(
            plot_space_indices, mean[(t, *space_indices)], "g-", lw=1, label="Est. mean"
        )
        if plot_region:
            ax.fill_between(
                plot_space_indices,
                mean[(t, *space_indices)] - 3 * std[(t, *space_indices)],
                mean[(t, *space_indices)] + 3 * std[(t, *space_indices)],
                alpha=0.25,
                color="g",
                label="Est. mean ± 3 standard deviation",
            )
        if plot_particles:
            lines = ax.plot(
                plot_space_indices,
                particles[(t, slice(None, None, particle_skip), *space_indices)].T,
                "r-",
                lw=0.25,
                alpha=trace_alpha,
            )
            lines[0].set_label("Particles")
        if true_state_sequence is not None:
            ax.plot(
                plot_space_indices,
                true_state_sequence.reshape((num_time_step,) + mesh_shape)[
                    (t, *space_indices)
                ],
                "k--",
                label="Truth",
            )
        ax.set_ylabel(f"$x_{{{space_subscript},{t}}}$")
        ax.legend(loc="upper center", ncol=4)
    ax.set_xlabel("Space index $s$")
    fig.tight_layout()
    return fig, axes


def animate_2d_fields(field_sequences, spatial_domain_size=(1, 1), interval=50):
    fig, axes = plt.subplots(
        1, len(field_sequences), figsize=(4 * len(field_sequences), 4)
    )
    extent = (0, spatial_domain_size[0], 0, spatial_domain_size[1])
    min_val = min(field_sequence.min() for field_sequence in field_sequences.values())
    max_val = max(field_sequence.max() for field_sequence in field_sequences.values())
    im_artists = [
        ax.imshow(
            field_sequence[-1],
            cmap="viridis",
            extent=extent,
            vmin=min_val,
            vmax=max_val,
        )
        for ax, field_sequence in zip(axes, field_sequences.values())
    ]
    for ax, label in zip(axes, field_sequences.keys()):
        ax.set(xlabel="Spatial coordinate $s_0$", ylabel="Spatial coordinate $s_1$")
        ax.set_title(label)
    fig.tight_layout()

    def update(i):
        for im, field_sequence in zip(im_artists, field_sequences.values()):
            im.set_data(field_sequence[i])
        return im_artists

    anim = animation.FuncAnimation(
        fig,
        update,
        min(len(seq) for seq in field_sequences.values()),
        interval=interval,
        blit=True,
    )
    return anim, fig, axes
