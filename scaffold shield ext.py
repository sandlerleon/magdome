"""Extension runs for the revised manuscript: validation, convergence,
parameter sweeps, high-statistics chiral runs, and SEP dose-proxy folding."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, time
from scaffold_shield_sim import (B_field, geom_monolith, geom_scaffold, geom_chiral,
                                 fence_positions, M_TOT, MU0_4PI, Q, MP, C, R_TARGET)

t0 = time.time()
OUT = {}

# ---------- run_beam with configurable dt divisor ----------
def run_beam2(dpos, dmom, E_MeV, N=200, seed=0, dt_div=25.0, max_steps=30000):
    rng = np.random.default_rng(seed)
    E = E_MeV * 1e6 * Q
    gamma = 1.0 + E / (MP * C**2)
    u0 = C * np.sqrt(gamma**2 - 1.0); v0 = u0 / gamma
    pos = np.column_stack([np.full(N, -900.0), rng.uniform(-55, 55, N), rng.uniform(-55, 55, N)])
    th = rng.normal(0.0, 0.005, (N, 2))
    dirs = np.column_stack([np.ones(N), th[:, 0], th[:, 1]])
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    u = u0 * dirs
    active = np.ones(N, bool); hit = np.zeros(N, bool)
    for _ in range(max_steps):
        if not active.any(): break
        idx = np.where(active)[0]
        B = B_field(pos[idx], dpos, dmom)
        Bmax = max(np.linalg.norm(B, axis=1).max(), 1e-9)
        Tg = 2*np.pi * gamma * MP / (Q * Bmax)
        dt = min(Tg / dt_div, 10.0 / v0)
        t = (Q * dt / (2.0 * gamma * MP)) * B
        up = u[idx] + np.cross(u[idx], t)
        s = 2.0 / (1.0 + (t*t).sum(-1))
        u[idx] = u[idx] + s[:, None] * np.cross(up, t)
        pos[idx] += (u[idx] / gamma) * dt
        rmag = np.linalg.norm(pos[idx], axis=1)
        nh = rmag < R_TARGET
        hit[idx[nh]] = True
        out = ((pos[idx,0] > 950)|(pos[idx,0] < -950)|(np.abs(pos[idx,1])>1200)|(np.abs(pos[idx,2])>1200))
        active[idx[nh | out]] = False
    return hit.mean()

def ci95(T, N):
    return 1.96*np.sqrt(max(T*(1-T),1e-9)/N)

# =========================================================
# 1. VALIDATION A: Boris vs analytic gyroradius, uniform B
# =========================================================
print("== Validation: uniform-field gyroradius ==", flush=True)
Bu = 1e-3  # T
E_MeV = 1.0
E = E_MeV*1e6*Q; gamma = 1+E/(MP*C**2)
u0 = C*np.sqrt(gamma**2-1); v = u0/gamma
p_mom = gamma*MP*v
rg_analytic = p_mom/(Q*Bu)
divs = [5, 10, 25, 50, 100]
rg_err = []
for dv in divs:
    Tg = 2*np.pi*gamma*MP/(Q*Bu); dt = Tg/dv
    u = np.array([u0, 0, 0.]); x = np.array([0,0,0.])
    Bv = np.array([0,0,Bu]); xs=[]
    for _ in range(int(3*dv)):
        t = (Q*dt/(2*gamma*MP))*Bv
        up = u+np.cross(u,t); u = u+2*np.cross(up,t)/(1+(t*t).sum())
        x = x+(u/gamma)*dt; xs.append(x.copy())
    xs = np.array(xs)
    cx, cy = xs[:,0].mean(), xs[:,1].mean()
    r_num = np.sqrt((xs[:,0]-cx)**2+(xs[:,1]-cy)**2).mean()
    rg_err.append(abs(r_num-rg_analytic)/rg_analytic)
    sp_err = abs(np.linalg.norm(u)-u0)/u0
    print(f"  dt=Tg/{dv:3d}: r_g err={rg_err[-1]:.2e}, |u| drift={sp_err:.2e}", flush=True)
OUT["validation_gyro"] = {"divs": divs, "rg_err": rg_err, "rg_analytic_m": rg_analytic}

# =========================================================
# 2. VALIDATION B: convergence of transmission (scaffold, 10 MeV)
# =========================================================
print("== Convergence: dt and N ==", flush=True)
dp, dm = geom_scaffold()
conv_dt = {dv: run_beam2(dp, dm, 10.0, N=500, seed=1, dt_div=dv) for dv in [10, 25, 50]}
print("  dt:", conv_dt, flush=True)
conv_N = {}
for N in [100, 200, 500, 1000]:
    Ts = [run_beam2(dp, dm, 10.0, N=N, seed=s) for s in range(3)]
    conv_N[N] = (float(np.mean(Ts)), float(np.std(Ts)))
    print(f"  N={N}: T={np.mean(Ts):.3f} +/- {np.std(Ts):.3f}", flush=True)
OUT["conv_dt"] = conv_dt; OUT["conv_N"] = conv_N

# =========================================================
# 3. SWEEP: moment budget -> knee scaling (scaffold)
# =========================================================
print("== Sweep: moment budget ==", flush=True)
E_grid = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
budget_factors = [0.25, 1.0, 4.0]
budget_curves = {}
for f in budget_factors:
    Ts = [run_beam2(dp, dm*f, E, N=500, seed=2) for E in E_grid]
    budget_curves[f] = Ts
    print(f"  f={f}: {['%.3f'%t for t in Ts]}", flush=True)
OUT["budget_curves"] = {"E": E_grid, "curves": {str(k): v for k, v in budget_curves.items()}}

def knee(E, T, level=0.5):
    E = np.log10(E); T = np.array(T)
    for i in range(len(T)-1):
        if T[i] < level <= T[i+1]:
            x = E[i]+(level-T[i])*(E[i+1]-E[i])/(T[i+1]-T[i])
            return 10**x
    return np.nan
knees = {f: knee(E_grid, budget_curves[f]) for f in budget_factors}
OUT["knees"] = {str(k): v for k, v in knees.items()}
print("  knees E50 [MeV]:", knees, flush=True)

# =========================================================
# 4. SWEEP: fence spacing (Sigma|m| fixed)
# =========================================================
print("== Sweep: fence spacing ==", flush=True)
def scaffold_spacing(s):
    ys = np.arange(-500.0, 500.1, s); xf = [-350.0, -150.0]
    pos, mom = [], []
    for i, x in enumerate(xf):
        for j, y in enumerate(ys):
            pos.append([x, y, 0.0]); mom.append([0,0, 1.0 if (i+j)%2==0 else -1.0])
    pos = np.array(pos); mom = np.array(mom); mom *= M_TOT/len(pos)
    return pos, mom
spacing_res = {}
for s in [50.0, 100.0, 200.0, 250.0]:
    dps, dms = scaffold_spacing(s)
    spacing_res[s] = {E: run_beam2(dps, dms, E, N=500, seed=3) for E in [3.0, 10.0, 30.0]}
    print(f"  s={s:.0f} m ({len(dps)} dipoles): {spacing_res[s]}", flush=True)
OUT["spacing"] = {str(k): {str(e): t for e, t in v.items()} for k, v in spacing_res.items()}

# =========================================================
# 5. SWEEP: chiral pitch map, N=1000
# =========================================================
print("== Sweep: chiral pitch x energy, N=1000 ==", flush=True)
pitches = [100.0, 200.0, 400.0, 800.0, 1600.0]
E_chi = [3.0, 10.0, 30.0]
chiral_map = {}
for L in pitches:
    for E in E_chi:
        dpr, dmr = geom_chiral(+1, pitch=L)
        dpl, dml = geom_chiral(-1, pitch=L)
        Trh = run_beam2(dpr, dmr, E, N=1000, seed=4)
        Tlh = run_beam2(dpl, dml, E, N=1000, seed=4)
        g = 2*(Trh-Tlh)/(Trh+Tlh) if (Trh+Tlh) > 0 else 0.0
        chiral_map[(L, E)] = (Trh, Tlh, g)
        print(f"  L={L:6.0f} E={E:5.1f}: RH={Trh:.3f} LH={Tlh:.3f} g={g:+.3f}  [{time.time()-t0:5.1f}s]", flush=True)
OUT["chiral_map"] = {f"{k[0]}_{k[1]}": v for k, v in chiral_map.items()}

# =========================================================
# 6. DOSE PROXY: fold T(E) into representative SEP spectrum
# =========================================================
print("== Dose proxy ==", flush=True)
E_dose = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
T_mono  = [run_beam2(*geom_monolith(), E, N=500, seed=5) for E in E_dose]
T_scaf  = [run_beam2(dp, dm, E, N=500, seed=5) for E in E_dose]
print("  T_mono:", ["%.3f"%t for t in T_mono], flush=True)
print("  T_scaf:", ["%.3f"%t for t in T_scaf], flush=True)
OUT["dose_T"] = {"E": E_dose, "mono": T_mono, "scaf": T_scaf}

def T_interp(Egrid, Tvals, Eq):
    # log-E linear interp; T=0 below grid; ramps to 1 at 300 MeV above grid
    lE = np.log10(Egrid); lq = np.log10(Eq)
    T = np.interp(lq, lE, Tvals, left=0.0, right=np.nan)
    hi = lq > lE[-1]
    T_end = Tvals[-1]
    ramp = np.clip((lq[hi]-lE[-1])/(np.log10(300)-lE[-1]), 0, 1)
    T[hi] = T_end + (1.0-T_end)*ramp
    return np.clip(T, 0, 1)

E0 = 30.0  # MeV, spectral e-folding of representative large SEP event
Eq = np.logspace(np.log10(0.1), np.log10(500), 800)
J = np.exp(-Eq/E0)/Eq          # differential proton fluence proxy
w_num, w_en = J, J*Eq
res_dose = {}
for name, Tv in [("monolith", T_mono), ("scaffold", T_scaf)]:
    Tq = T_interp(E_dose, Tv, Eq)
    A_num = 1 - np.trapezoid(Tq*w_num, Eq)/np.trapezoid(w_num, Eq)
    A_en  = 1 - np.trapezoid(Tq*w_en, Eq)/np.trapezoid(w_en, Eq)
    res_dose[name] = (float(A_num), float(A_en))
    print(f"  {name}: number-fluence attenuation={A_num:.3f}, energy-fluence attenuation={A_en:.3f}", flush=True)
OUT["dose_attenuation"] = res_dose

with open("/home/claude/ext_results.json", "w") as f:
    json.dump(OUT, f, indent=1, default=float)

# =========================================================
# FIGURES 5-8
# =========================================================
# Fig 5: validation
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].loglog(divs, rg_err, "o-", color="tab:blue")
ax[0].loglog(divs, [rg_err[0]*(divs[0]/d)**2 for d in divs], "k--", lw=1, label=r"$\propto \Delta t^2$")
ax[0].set_xlabel("Steps per gyroperiod"); ax[0].set_ylabel("Relative gyroradius error")
ax[0].set_title("(a) Boris integrator vs analytic gyroradius\n(uniform 1 mT, 1 MeV proton)")
ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
Ns = sorted(conv_N); means = [conv_N[n][0] for n in Ns]; stds = [conv_N[n][1] for n in Ns]
ax[1].errorbar(Ns, means, yerr=stds, fmt="s-", color="tab:green", capsize=4, label="seed-to-seed spread")
for dv, T in conv_dt.items():
    ax[1].axhline(T, ls=":", lw=1, alpha=0.6)
ax[1].set_xscale("log"); ax[1].set_xlabel("Particles N"); ax[1].set_ylabel("Transmission")
ax[1].set_title("(b) Convergence, scaffold @ 10 MeV\n(dotted: dt = T$_g$/10, /25, /50 at N=500)")
ax[1].grid(alpha=0.3); ax[1].legend()
fig.tight_layout(); fig.savefig("/home/claude/fig5_validation.png", dpi=150, bbox_inches="tight")

# Fig 6: budget + spacing sweeps
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
cols = {0.25: "tab:orange", 1.0: "tab:blue", 4.0: "tab:purple"}
for f in budget_factors:
    lab = f"$M$ = {f}$\\times M_0$ ($E_{{50}}$ = {knees[f]:.0f} MeV)" if not np.isnan(knees[f]) else f"$M$ = {f}$\\times M_0$"
    ax[0].plot(E_grid, budget_curves[f], "o-", color=cols[f], label=lab)
ax[0].set_xscale("log"); ax[0].set_xlabel("Energy [MeV]"); ax[0].set_ylabel("Transmission")
ax[0].set_title("(a) Rigidity knee vs moment budget (scaffold)")
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
ss = sorted(spacing_res); mk = {3.0: "o-", 10.0: "s-", 30.0: "^-"}
for E in [3.0, 10.0, 30.0]:
    ax[1].plot(ss, [spacing_res[s][E] for s in ss], mk[E], label=f"{E:.0f} MeV")
ax[1].set_xlabel("Fence site spacing [m]"); ax[1].set_ylabel("Transmission")
ax[1].set_title("(b) Spacing sweep, $\\Sigma|m|$ fixed")
ax[1].grid(alpha=0.3); ax[1].legend()
fig.tight_layout(); fig.savefig("/home/claude/fig6_sweeps.png", dpi=150, bbox_inches="tight")

# Fig 7: chiral pitch map with error bars
fig, ax = plt.subplots(figsize=(7.5, 5))
mkE = {3.0: ("o-", "tab:green"), 10.0: ("s-", "tab:purple"), 30.0: ("^-", "tab:red")}
for E in E_chi:
    gs, errs = [], []
    for L in pitches:
        Trh, Tlh, g = chiral_map[(L, E)]
        gs.append(g)
        Tm = (Trh+Tlh)/2
        errs.append(2*np.sqrt(2)*np.sqrt(max(Tm*(1-Tm),1e-9)/1000)/max(Tm,1e-3))
    st, col = mkE[E]
    ax.errorbar(pitches, gs, yerr=errs, fmt=st, color=col, capsize=3, label=f"{E:.0f} MeV")
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("Winding pitch $\\Lambda$ [m]")
ax.set_ylabel("Chiral discrimination $g$")
ax.set_title("Chiral discrimination vs winding pitch (N = 1000 per point)")
ax.grid(alpha=0.3); ax.legend(title="Proton energy")
fig.savefig("/home/claude/fig7_pitchmap.png", dpi=150, bbox_inches="tight")

# Fig 8: dose folding
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
Tq_m = T_interp(E_dose, T_mono, Eq); Tq_s = T_interp(E_dose, T_scaf, Eq)
ax[0].loglog(Eq, J/J.max(), "k-", label="SEP fluence proxy $J(E) \\propto e^{-E/30\\,\\mathrm{MeV}}/E$")
ax[0].loglog(Eq, Tq_s*J/J.max(), "b-", label="transmitted (scaffold)")
ax[0].loglog(Eq, Tq_m*J/J.max(), "r--", label="transmitted (monolith)")
ax[0].set_ylim(1e-8, 2); ax[0].set_xlabel("Energy [MeV]"); ax[0].set_ylabel("Normalized differential fluence")
ax[0].set_title("(a) Spectrum folding"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
names = ["Monolith", "Scaffold"]
An = [res_dose["monolith"][0], res_dose["scaffold"][0]]
Ae = [res_dose["monolith"][1], res_dose["scaffold"][1]]
xp = np.arange(2)
ax[1].bar(xp-0.18, An, 0.36, label="number fluence", color="steelblue")
ax[1].bar(xp+0.18, Ae, 0.36, label="energy fluence", color="indianred")
for i,(a,b) in enumerate(zip(An,Ae)):
    ax[1].text(i-0.18, a+0.01, f"{a:.2f}", ha="center"); ax[1].text(i+0.18, b+0.01, f"{b:.2f}", ha="center")
ax[1].set_xticks(xp); ax[1].set_xticklabels(names); ax[1].set_ylim(0, 1.05)
ax[1].set_ylabel("Attenuated fraction"); ax[1].set_title("(b) Spectrum-integrated attenuation")
ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig("/home/claude/fig8_dose.png", dpi=150, bbox_inches="tight")

print(f"\nAll extension runs done in {time.time()-t0:.1f}s")
