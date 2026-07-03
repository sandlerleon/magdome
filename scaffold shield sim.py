"""
Magnetic Scaffold Shielding Simulation
--------------------------------------
Tests whether the 'scaffold' design principles (periodic multi-channel arrays,
Cooper Scaffold; helical/chiral winding, Chiral Scaffold) transfer to
heliosphere-style charged-particle shielding for a lunar surface installation.

Three geometries, equal total magnetic moment budget Sigma|m| = M_tot:
  A) MONOLITH:  single dipole at origin (mini-magnetosphere standoff shield)
  B) SCAFFOLD:  two parallel fences of alternating-polarity dipoles
                (multi-channel periodic array -> distributed scattering screen,
                 heliosphere-like diffusion barrier)
  C) CHIRAL:    same fence positions, moments wind helically along the fence
                (Parker-spiral analog; run both handednesses to test
                 chiral discrimination of the transmitted flux)

Protons launched as an aimed beam at the protected zone (sphere R=60 m at
origin). Relativistic Boris pusher. Transmission vs energy = the gyroradius
filter curve, direct analog of heliospheric cosmic-ray modulation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv, time

# ---------------- constants ----------------
MU0_4PI = 1e-7
Q  = 1.602176634e-19
MP = 1.67262192e-27
C  = 2.99792458e8
A_CORE = 25.0          # dipole core softening [m]
R_TARGET = 60.0        # protected zone radius [m]
M_TOT = 4e10           # total magnetic moment budget [A m^2]

# ---------------- geometries ----------------
def geom_monolith():
    return (np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, M_TOT]]))

def fence_positions():
    ys = np.arange(-500.0, 500.1, 100.0)      # 11 sites per fence
    xf = [-350.0, -150.0]                     # two channels (multi-channel)
    pos = np.array([[x, y, 0.0] for x in xf for y in ys])
    return pos, ys, xf

def geom_scaffold():
    pos, ys, xf = fence_positions()
    mom = []
    for i, x in enumerate(xf):
        for j, y in enumerate(ys):
            s = 1.0 if (i + j) % 2 == 0 else -1.0   # alternating polarity
            mom.append([0.0, 0.0, s])
    mom = np.array(mom)
    mom *= M_TOT / len(pos)
    return pos, mom

def geom_chiral(hand=+1, pitch=400.0):
    pos, ys, xf = fence_positions()
    mom = []
    for i, x in enumerate(xf):
        for y in ys:
            phi = hand * 2*np.pi * y / pitch + i * np.pi/2   # phase offset between channels
            mom.append([np.sin(phi), 0.0, np.cos(phi)])      # helical texture in x-z vs y
    mom = np.array(mom)
    mom *= M_TOT / len(pos)
    return pos, mom

# ---------------- field ----------------
def B_field(r, dpos, dmom):
    """Summed softened point-dipole field. r: (N,3) -> (N,3)."""
    d  = r[:, None, :] - dpos[None, :, :]
    r2 = (d*d).sum(-1) + A_CORE**2
    r1 = np.sqrt(r2)
    rhat = d / r1[..., None]
    mdotr = (dmom[None, :, :] * rhat).sum(-1)
    B = MU0_4PI * (3.0*mdotr[..., None]*rhat - dmom[None, :, :]) / (r2**1.5)[..., None]
    return B.sum(axis=1)

# ---------------- particle push ----------------
def run_beam(dpos, dmom, E_MeV, N=200, seed=0, keep_traj=0, max_steps=20000):
    rng = np.random.default_rng(seed)
    E  = E_MeV * 1e6 * Q
    gamma = 1.0 + E / (MP * C**2)
    u0 = C * np.sqrt(gamma**2 - 1.0)          # |u| = gamma*v
    v0 = u0 / gamma

    pos = np.column_stack([np.full(N, -900.0),
                           rng.uniform(-55, 55, N),
                           rng.uniform(-55, 55, N)])
    th = rng.normal(0.0, 0.005, (N, 2))
    dirs = np.column_stack([np.ones(N), th[:, 0], th[:, 1]])
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    u = u0 * dirs

    active = np.ones(N, bool)
    hit    = np.zeros(N, bool)
    trajs  = [[pos[k].copy()] for k in range(keep_traj)]

    for step in range(max_steps):
        if not active.any():
            break
        idx = np.where(active)[0]
        B = B_field(pos[idx], dpos, dmom)
        Bmax = max(np.linalg.norm(B, axis=1).max(), 1e-9)
        Tg = 2*np.pi * gamma * MP / (Q * Bmax)
        dt = min(Tg / 25.0, 10.0 / v0)

        # relativistic Boris (B only; gamma constant)
        t = (Q * dt / (2.0 * gamma * MP)) * B
        up = u[idx] + np.cross(u[idx], t)
        s  = 2.0 / (1.0 + (t*t).sum(-1))
        u[idx] = u[idx] + s[:, None] * np.cross(up, t)
        pos[idx] += (u[idx] / gamma) * dt

        for k in range(keep_traj):
            if active[k]:
                trajs[k].append(pos[k].copy())

        rmag = np.linalg.norm(pos[idx], axis=1)
        newly_hit = rmag < R_TARGET
        hit[idx[newly_hit]] = True
        out = ((pos[idx, 0] >  950) | (pos[idx, 0] < -950) |
               (np.abs(pos[idx, 1]) > 1200) | (np.abs(pos[idx, 2]) > 1200))
        active[idx[newly_hit | out]] = False

    return hit.mean(), [np.array(t) for t in trajs]

# ---------------- experiment ----------------
if __name__ == "__main__":
    t0 = time.time()
    energies = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]   # MeV
    geoms = {
        "Monolith (single dipole)": geom_monolith(),
        "Cooper scaffold (2-channel array)": geom_scaffold(),
        "Chiral scaffold RH": geom_chiral(+1),
        "Chiral scaffold LH": geom_chiral(-1),
    }

    results = {name: [] for name in geoms}
    for name, (dp, dm) in geoms.items():
        for E in energies:
            T, _ = run_beam(dp, dm, E, N=200, seed=42)
            results[name].append(T)
            print(f"{name:38s} E={E:7.2f} MeV  transmission={T:.3f}  "
                  f"[{time.time()-t0:5.1f}s]", flush=True)

    with open("/home/claude/transmission_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["E_MeV"] + list(results.keys()))
        for i, E in enumerate(energies):
            w.writerow([E] + [f"{results[n][i]:.4f}" for n in results])

    # ---- Fig 1: field maps ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    maps = [("Monolith", geom_monolith()),
            ("Cooper scaffold", geom_scaffold()),
            ("Chiral scaffold (RH)", geom_chiral(+1))]
    gx = np.linspace(-700, 400, 300); gy = np.linspace(-650, 650, 300)
    GX, GY = np.meshgrid(gx, gy)
    P = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)])
    for ax, (title, (dp, dm)) in zip(axes, maps):
        Bm = np.linalg.norm(B_field(P, dp, dm), axis=1).reshape(GX.shape)
        im = ax.pcolormesh(GX, GY, np.log10(Bm + 1e-12), cmap="magma",
                           vmin=-8, vmax=-1.5, shading="auto")
        ax.add_patch(plt.Circle((0, 0), R_TARGET, fill=False, color="cyan", lw=1.6))
        ax.scatter(dp[:, 0], dp[:, 1], s=12, c="white", marker="o")
        ax.set_title(title); ax.set_xlabel("x [m]"); ax.set_aspect("equal")
    axes[0].set_ylabel("y [m]")
    fig.colorbar(im, ax=axes, label=r"$\log_{10}|B|$ [T]", shrink=0.85)
    fig.suptitle("Field geometry in z=0 plane (equal total moment budget)", y=1.02)
    fig.savefig("/home/claude/fig1_geometry.png", dpi=150, bbox_inches="tight")

    # ---- Fig 2: sample trajectories at 0.3 MeV ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (title, (dp, dm)) in zip(axes, maps):
        _, trajs = run_beam(dp, dm, 0.3, N=30, seed=7, keep_traj=30)
        for tr in trajs:
            ended_in = np.linalg.norm(tr[-1]) < R_TARGET + 5
            ax.plot(tr[:, 0], tr[:, 1], lw=0.8,
                    color="crimson" if ended_in else "steelblue", alpha=0.8)
        ax.add_patch(plt.Circle((0, 0), R_TARGET, fill=False, color="k", lw=1.5))
        ax.scatter(dp[:, 0], dp[:, 1], s=14, c="k", marker="s")
        ax.set_title(title); ax.set_xlabel("x [m]")
        ax.set_xlim(-950, 500); ax.set_ylim(-700, 700)
    axes[0].set_ylabel("y [m]")
    fig.suptitle("Proton trajectories, 0.3 MeV (red = reached target, blue = deflected)")
    fig.savefig("/home/claude/fig2_trajectories.png", dpi=150, bbox_inches="tight")

    # ---- Fig 3: transmission vs energy ----
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    styles = {"Monolith (single dipole)": ("o-", "tab:red"),
              "Cooper scaffold (2-channel array)": ("s-", "tab:blue"),
              "Chiral scaffold RH": ("^-", "tab:green"),
              "Chiral scaffold LH": ("v--", "tab:olive")}
    for name in results:
        st, col = styles[name]
        ax.plot(energies, results[name], st, color=col, label=name, ms=6)
    ax.set_xscale("log"); ax.set_xlabel("Proton kinetic energy [MeV]")
    ax.set_ylabel("Transmission to protected zone")
    ax.set_title("Gyroradius filter: transmission vs energy\n(heliosphere-analog modulation curve)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(-0.03, 1.05)
    fig.savefig("/home/claude/fig3_transmission.png", dpi=150, bbox_inches="tight")

    # ---- Fig 4: chiral asymmetry ----
    rh = np.array(results["Chiral scaffold RH"])
    lh = np.array(results["Chiral scaffold LH"])
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where((rh + lh) > 0, 2*(rh - lh)/(rh + lh), 0.0)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(energies, g, "d-", color="purple")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("Proton kinetic energy [MeV]")
    ax.set_ylabel(r"Chiral discrimination  $g = 2(T_{RH}-T_{LH})/(T_{RH}+T_{LH})$")
    ax.set_title("Handedness-dependent transmission through helical scaffold")
    ax.grid(alpha=0.3)
    fig.savefig("/home/claude/fig4_chirality.png", dpi=150, bbox_inches="tight")

    print(f"\nDone in {time.time()-t0:.1f}s")
