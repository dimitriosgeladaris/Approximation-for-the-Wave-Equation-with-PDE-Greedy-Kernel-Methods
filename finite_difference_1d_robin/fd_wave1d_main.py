import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.animation import FuncAnimation
import time

# --------------------------------------------------
# plotting style
# --------------------------------------------------
tol_bright = [
    "#4477AA", "#EE6677", "#228833",
    "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"
]
plt.rcParams["axes.prop_cycle"] = cycler(color=tol_bright)

# --------------------------------------------------
# project imports
# --------------------------------------------------
import auxFunctions
import wave1d_model as model

# --------------------------------------------------
# global settings
# --------------------------------------------------
dt   = 0.001
endT = 2.0

# --------------------------------------------------
# generate full-order wave snapshots
# --------------------------------------------------
N_full = 1000 # number of interior spatial nodes

fullWave = model.LinearWave1D(
    L=4.0,
    c=4.0,
    N=N_full,
    datastring="waves"
)

# generate initial conditions
Y0_full = fullWave.getInitialState(
    endT=endT,
    dt=dt
)

snapshots = []
print("Generating full-order snapshots ...")
start_time = time.time()

for s in range(Y0_full.shape[1]):
    print(f"IC {s+1}/{Y0_full.shape[1]}")
    y0 = Y0_full[:, s]

    sol, tspan = auxFunctions.implicit_midpoint(
        y0,
        fullWave.ode,
        dt,
        endT
    )

    snapshots.append((sol, tspan))

    # --------------------------------------------------
    # Save snapshots to file
    # --------------------------------------------------
    dt_snap = 0.01
    skip = int(dt_snap / dt)

    downsampled_snapshots = [sol[:, ::skip] for sol, _ in snapshots]

    np.savez(
        "wave1d_full_snapshots_compact.npz",
        snapshots=downsampled_snapshots,
        dt_snap=dt_snap,
        endT=endT,
        L=fullWave.L,
        c=fullWave.c,
        N=fullWave.N
    )

    print("Snapshots saved to wave1d_full_snapshots_compact.npz")
end_time = time.time()
print(f"Snapshot generation completed in {end_time - start_time:.2f} seconds.")

# --------------------------------------------------
# choose which snapshot to animate
# --------------------------------------------------
snapshot_id = 0  # change this index to animate others
sol, tspan = snapshots[snapshot_id]

# --------------------------------------------------
# animation setup
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2)

x = np.linspace(0, fullWave.L, fullWave.N)

ax.set_xlim(0, fullWave.L)
ax.set_ylim(-1, 1)
ax.set_xlabel("Spatial coordinate x")
ax.set_ylabel("Displacement q")
ax.grid(True, alpha=0.3)

def animate(ti):
    line.set_data(x, sol[:fullWave.N, ti])
    ax.set_title(f"Wave Evolution – t = {tspan[ti]:.3f} s")
    return line,


# --------------------------------------------------
# animation timing (decoupled from dt)
# --------------------------------------------------
frame_stride = 10   # show every 10th timestep

anim = FuncAnimation(
    fig,
    animate,
    frames=range(0, len(tspan), frame_stride),
    interval=50,
    blit=False
)


plt.tight_layout()
plt.show()

# # --------------------------------------------------
# # Save snapshots to file
# # --------------------------------------------------

# # Downsample snapshots to every 0.01 s
# dt_snap = 0.01
# skip = int(dt_snap / dt)  # number of timesteps to skip

# downsampled_snapshots = [sol[:, ::skip] for sol in snapshots]

# np.savez(
#     "wave1d_full_snapshots_compact.npz",
#     snapshots=downsampled_snapshots,
#     dt_snap=dt_snap,     # effective timestep after downsampling
#     endT=endT,
#     L=fullWave.L,
#     c=fullWave.c,
#     N=fullWave.N
# )

# # --------------------------------------------------
# # stage 1: load full-order snapshots
# # --------------------------------------------------
# data = np.load("wave1d_full_snapshots_compact.npz", allow_pickle=True)
# snapshots = [np.array(snap) for snap in data["snapshots"]]

# fullWave = model.LinearWave1D(
#     L=data["L"],
#     c=data["c"],
#     N=data["N"],
#     datastring="waves"
# )

# # --------------------------------------------------
# # reconstruct correct time axis
# # --------------------------------------------------
# # Use the dt_snap that was saved during downsampling
# dt_snap = 0.1
# endT = float(data["endT"])

# Nt = snapshots[0].shape[1]

# # The correct time span based on the downsampled data
# tspan = np.linspace(0, endT, Nt)

# # Verify: the time step should match dt_snap
# print(f"Loaded dt_snap: {dt_snap:.4f}s")
# print(f"Calculated dt from tspan: {tspan[1] - tspan[0]:.4f}s")
# print(f"Number of snapshots: {Nt}")
# print(f"Total time: {endT}s")

# # spatial grid
# x = np.linspace(0, fullWave.L, fullWave.N)

# # --------------------------------------------------
# # plot selected snapshots
# # --------------------------------------------------
# plt.figure(figsize=(10, 6))
# times_to_show = [0, Nt//4, Nt//2, Nt-1]

# for s, sol in enumerate(snapshots):
#     for ti in times_to_show:
#         plt.plot(
#             x,
#             sol[:fullWave.N, ti],
#             label=f"Sim {s+1}, t={tspan[ti]:.2f}s"
#         )

# plt.xlabel("Spatial coordinate x")
# plt.ylabel("Displacement q")
# plt.title("Full-order wave snapshots")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # --------------------------------------------------
# # animate a single simulation
# # --------------------------------------------------
# fig, ax = plt.subplots(figsize=(10, 6))
# line, = ax.plot([], [], lw=2)
# ax.set_xlim(0, fullWave.L)
# ax.set_ylim(np.min(snapshots[0]), np.max(snapshots[0]))
# ax.set_xlabel("Spatial coordinate x")
# ax.set_ylabel("Displacement q")
# ax.grid(True, alpha=0.3)

# def animate(ti):
#     line.set_data(x, snapshots[0][:fullWave.N, ti])
#     ax.set_title(f"Wave Evolution - t={tspan[ti]:.3f}s")
#     return line,

# # Calculate interval to make animation play at real-time speed
# # interval is in milliseconds
# interval = dt_snap * 1000  # convert to ms

# anim = FuncAnimation(fig, animate, frames=Nt, interval=interval, blit=False)
# plt.tight_layout()
# plt.show()