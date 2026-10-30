"""
mrta_robotarium.py -- Robotarium experiment for Section 9 of
"Certificate-Native Coordination for Robot Teams".

ONE FILE, numpy + rps only, so it can be uploaded to the Robotarium as-is.
The same file runs in the Robotarium Python simulator for a dry run.

What it does
------------
Runs a schedule of missions back to back. Each mission: N GRITSbots, N task
sites, one allocation method, one packet-loss level, one seed. Between missions
the robots are driven back to the mission's initial poses.

Every control step (TIME_STEP = 0.033 s):
  1. a disk radio graph of radius R_COMM is built from the measured poses;
     each link is down in a radio slot with probability P_LOSS (both ways);
  2. the allocator advances by SLOTS_PER_STEP radio slots (one synchronous
     broadcast round per slot), counting every broadcast and its bytes
     (payload model of Sec. 6.1: 4 bytes per scalar + 20-byte header);
  3. each active robot drives toward the task of its current assignment with
     the standard unicycle position controller; the command passes through the
     Robotarium barrier certificate (collision + boundary);
  4. nominal and filtered commands, measured poses and the barrier-active flag
     are logged for the price-of-safety split (Definition "pos").

A robot within AT_TASK m of its assigned task completes it and retires (stays
in place, still an obstacle). Completion is a structural event: the allocation
restarts on the remaining robots and tasks.

Methods: greedy (sequential flooded claims), cbaa (Choi et al. 2009),
auction (Bertsekas auction + max-consensus), proposed (Alg. 2 with Alg. 1
consensus Sinkhorn on the live graph, event trigger, local stopping rule,
certificate ledger of Theorem "ledger").

Outputs one .npz per mission plus a summary line per mission on stdout.
"""

import numpy as np

import rps.robotarium as robotarium
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
from rps.utilities.controllers import create_uni_position_controller, create_pose_controller_hybrid
from rps.utilities.misc import generate_random_poses, create_at_pose
from rps.utilities.transformations import create_si_to_uni_mapping, create_uni_to_si_mapping

# Same mapping and speed limit as create_uni_barrier_certificate_with_boundary()
# (projection_distance=0.03, velocity_magnitude_limit=0.2), WITHOUT the safety
# constraints: the difference between this and the barrier output is the
# barrier's intervention and nothing else.
_SI_UNI, _ = create_si_to_uni_mapping(projection_distance=0.03)
_UNI_SI, _ = create_uni_to_si_mapping(projection_distance=0.03)


def unconstrained_filter(dxu, x, vmax=0.2):
    dxi = _UNI_SI(dxu, x)
    nrm = np.linalg.norm(dxi, axis=0)
    scale = np.where(nrm > vmax, vmax / np.maximum(nrm, 1e-12), 1.0)
    return _SI_UNI(dxi * scale, x)

# ----------------------------------------------------------------------------
# Configuration (edit here for a Robotarium submission)
# ----------------------------------------------------------------------------
N_ROBOTS = 6
SEEDS = [1]
METHODS = ["greedy", "cbaa", "auction", "proposed_fast", "proposed_accurate"]
P_LOSS_LEVELS = [0.0, 0.2]
R_COMM = 1.0            # m, radio range
SLOTS_PER_STEP = 6      # radio slots per 33 ms control step (~5.5 ms per slot)
MAX_MISSION_TIME = 60.0 # s, per mission
AT_TASK = 0.05          # m, completion radius (Robotarium at_position default)
COST_SCALE = 1.0        # cost = travel time to the task in s (dist / v_max) + heading term
HEADING_WEIGHT = 0.2    # s per rad of heading error
SIMULATION = True       # False on the Robotarium (also enables initialization)
SAVE_PREFIX = "mrta"
RELAY_RETIRED = True    # retired robots keep their radios on and relay (all methods)

PROPOSED_CONFIGS = {
    # alpha raised from the benchmark's 0.08 because costs here are travel
    # times of 0-20 s rather than 5-100 cost units.
    "proposed_fast":     dict(alpha=0.5, beta=0.35, lam0=0.5, laminf=0.02, Kf=200, M=5, T=2,  delta=0.05, stop_K=20),
    "proposed_accurate": dict(alpha=0.5, beta=0.35, lam0=0.5, laminf=0.02, Kf=200, M=5, T=50, delta=0.05, stop_K=20),
}

HEADER_BYTES = 20.0
SCALAR_BYTES = 4.0


# ----------------------------------------------------------------------------
# Radio emulation
# ----------------------------------------------------------------------------
class Radio:
    """Disk graph from measured poses, i.i.d. symmetric link loss per slot,
    byte accounting. Every transmission is a local broadcast."""

    def __init__(self, rng, p_loss, r_comm):
        self.rng, self.p_loss, self.r_comm = rng, p_loss, r_comm
        self.alloc_bytes = 0.0
        self.aux_bytes = 0.0
        self.alloc_msgs = 0
        self.aux_msgs = 0
        self.slots = 0
        self.adj = None

    def set_positions(self, xy):
        d = np.linalg.norm(xy[:, :, None] - xy[:, None, :], axis=0)
        self.adj = (d <= self.r_comm) & ~np.eye(xy.shape[1], dtype=bool)

    def slot_links(self, ids):
        """Links that are up in this slot among robots `ids` (symmetric)."""
        self.slots += 1
        A = self.adj[np.ix_(ids, ids)].copy()
        if self.p_loss > 0:
            up = self.rng.random(A.shape) >= self.p_loss
            up = np.triu(up, 1); up = up | up.T
            A &= up
        return A

    def broadcast(self, count, scalars, layer):
        b = count * (SCALAR_BYTES * scalars + HEADER_BYTES)
        if layer == "alloc":
            self.alloc_bytes += b; self.alloc_msgs += count
        else:
            self.aux_bytes += b; self.aux_msgs += count


def metropolis(A):
    deg = A.sum(1)
    n = A.shape[0]
    W = np.zeros((n, n))
    ii, jj = np.nonzero(A)
    W[ii, jj] = 1.0 / (1.0 + np.maximum(deg[ii], deg[jj]))
    W[np.arange(n), np.arange(n)] = 1.0 - W.sum(1)
    return W


def round_greedy(P):
    """Rounding of Cor. 'rounding': greedy by largest entry."""
    n, d = P.shape
    order = np.argsort(-P, axis=None)
    a = -np.ones(n, dtype=int)
    used_r = np.zeros(n, bool); used_c = np.zeros(d, bool)
    for idx in order:
        i, j = divmod(int(idx), d)
        if not used_r[i] and not used_c[j]:
            a[i] = j; used_r[i] = used_c[j] = True
    return a


# ----------------------------------------------------------------------------
# Allocators. Each is a generator: one `yield` per radio slot consumed. After
# every slot the attribute `assignment` (array of local task indices, -1 = none)
# is the robots' current belief. `restart(C)` is called at structural events.
# ----------------------------------------------------------------------------
class Greedy:
    """Robot k claims its cheapest unclaimed task once claims 0..k-1 have
    reached it; each claim is flooded (every informed robot rebroadcasts
    each slot until the claim has reached every robot)."""
    name = "greedy"

    def __init__(self, radio):
        self.radio = radio

    def run(self, C, ids, act):
        n, d = C.shape
        N = len(ids)
        self.assignment = -np.ones(n, dtype=int)
        taken = np.zeros(d, bool)
        for k in range(n):
            free = np.where(~taken)[0]
            j = free[np.argmin(C[k, free])]
            informed = np.zeros(N, bool); informed[act[k]] = True
            while not informed[act].all():
                A = self.radio.slot_links(ids)
                self.radio.broadcast(int(informed.sum()), 2, "alloc")
                informed = informed | (A[informed].any(0))
                yield
            taken[j] = True
            self.assignment[k] = j
        while True:
            yield


class CBAA:
    """CBAA (Choi, Brunet & How 2009), single assignment, max score
    s_ij = Cmax + 1 - C_ij. A robot broadcasts (y_i, z_i) in a slot while its
    view has changed within the last QUIET slots."""
    name = "cbaa"
    QUIET = 3

    def __init__(self, radio):
        self.radio = radio

    def run(self, C, ids, act):
        n, d = C.shape
        N = len(ids)
        s = np.zeros((N, d)); s[act] = C.max() + 1.0 - C      # relays never bid
        bidder = np.zeros(N, bool); bidder[act] = True
        y = np.zeros((N, d)); z = -np.ones((N, d), dtype=int)
        x = -np.ones(N, dtype=int)
        last_change = np.zeros(N, dtype=int)
        t = 0
        self.assignment = -np.ones(n, dtype=int)
        while True:
            t += 1
            for i in np.where(bidder)[0]:
                if x[i] < 0:
                    beats = (s[i] > y[i]) | ((s[i] == y[i]) & ((z[i] < 0) | (i < z[i])))
                    if beats.any():
                        J = int(np.argmax(np.where(beats, s[i], -np.inf)))
                        x[i] = J; y[i, J] = s[i, J]; z[i, J] = i; last_change[i] = t
            A = self.radio.slot_links(ids)
            talk = (t - last_change) <= self.QUIET
            self.radio.broadcast(int(talk.sum()), 2 * d, "alloc")
            y2, z2 = y.copy(), z.copy()
            for i in range(N):
                for k in np.where(A[i] & talk)[0]:
                    better = (y[k] > y2[i]) | ((y[k] == y2[i]) & (y[k] > 0) & (z[k] >= 0) & ((z2[i] < 0) | (z[k] < z2[i])))
                    y2[i, better] = y[k, better]; z2[i, better] = z[k, better]
            for i in range(N):
                if (y2[i] != y[i]).any() or (z2[i] != z[i]).any():
                    last_change[i] = t
            y, z = y2, z2
            for i in np.where(bidder)[0]:
                if x[i] >= 0 and z[i, x[i]] != i:
                    x[i] = -1; last_change[i] = t
            self.assignment = x[act].copy()
            yield


class Auction:
    """Bertsekas auction on benefit -C (eps = 0.01 s), Jacobi bidding,
    prices and bidder identities by max-consensus. Same quiescence rule."""
    name = "auction"
    QUIET = 3
    EPS = 0.01

    def __init__(self, radio):
        self.radio = radio

    def run(self, C, ids, act):
        n, d = C.shape
        N = len(ids)
        row = -np.ones(N, dtype=int); row[act] = np.arange(n)   # relays never bid
        p = np.zeros((N, d)); b = -np.ones((N, d), dtype=int)
        last_change = np.zeros(N, dtype=int)
        t = 0
        self.assignment = -np.ones(n, dtype=int)
        while True:
            t += 1
            for i in act:
                if not (b[i] == i).any():
                    val = -C[row[i]] - p[i]
                    j1 = int(np.argmax(val))
                    v2 = np.max(np.delete(val, j1)) if d > 1 else val[j1] - 1.0
                    p[i, j1] = p[i, j1] + (val[j1] - v2) + self.EPS
                    b[i, j1] = i; last_change[i] = t
            A = self.radio.slot_links(ids)
            talk = (t - last_change) <= self.QUIET
            self.radio.broadcast(int(talk.sum()), 2 * d, "alloc")
            p2, b2 = p.copy(), b.copy()
            for i in range(N):
                for k in np.where(A[i] & talk)[0]:
                    better = (p[k] > p2[i]) | ((p[k] == p2[i]) & (b[k] > b2[i]))
                    p2[i, better] = p[k, better]; b2[i, better] = b[k, better]
            for i in range(N):
                if (p2[i] != p[i]).any() or (b2[i] != b[i]).any():
                    last_change[i] = t
            p, b = p2, b2
            a = -np.ones(n, dtype=int)
            for li, i in enumerate(act):
                held = np.where(b[i] == i)[0]
                if held.size:
                    a[li] = held[0]
            self.assignment = a
            yield


class Proposed:
    """Algorithm 2 with Algorithm 1 on the live radio graph.

    One allocation iteration = mirror step + damping (local, 0 slots), then M
    sweeps of [row scaling, T consensus slots, column scaling], then one slot
    for event-triggered allocation-layer broadcasts. Costs are re-read from the
    current poses at the start of every iteration (C^k), so motion produces the
    drift term of the ledger. Local stopping rule: a robot falls silent when
    its own argmax has been unchanged for stop_K iterations; the team stops
    iterating when all are silent, until the next structural event."""
    name = "proposed"

    def __init__(self, radio, cfg, cost_fn):
        self.radio, self.cfg, self.cost_fn = radio, cfg, cost_fn
        self.ledger = []        # per-iteration dicts

    def lam(self, k):
        c = self.cfg
        return c["laminf"] if k >= c["Kf"] else max(c["laminf"], c["lam0"] / np.log(k + 2.0))

    def V(self, P, C, lam):
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
        return (C * P).sum(1) + lam * ent

    def run(self, C0, ids, act):
        c = self.cfg
        n, d = C0.shape
        N = len(ids)
        P = np.full((n, d), 1.0 / d)
        vlast = np.full(n, np.inf)
        amax = -np.ones(n, dtype=int); stable = np.zeros(n, dtype=int)
        self.assignment = -np.ones(n, dtype=int)
        k = 0
        while True:
            Ck = self.cost_fn()            # C^k from current poses
            lam_k = self.lam(k)
            W_k = self.V(P, Ck, lam_k).sum()
            # (1)-(2) mirror step + dual damping
            with np.errstate(divide="ignore"):
                logP = np.log(np.maximum(P, 1e-300))
            t_ = (1 + c["alpha"] * lam_k) * logP - c["alpha"] * Ck
            t_ -= t_.max(1, keepdims=True); Pt = np.exp(t_); Pt /= Pt.sum(1, keepdims=True)
            h = (1 - c["beta"]) * np.log(np.maximum(Pt, 1e-300)) + c["beta"] * logP
            h -= h.max(1, keepdims=True); Q = np.exp(h); Q /= Q.sum(1, keepdims=True)
            # (3) Algorithm 1 on the live graph
            for _ in range(c["M"]):
                Q /= Q.sum(1, keepdims=True)
                Z = np.zeros((N, d)); Z[act] = Q        # relays hold zero rows
                for _ in range(c["T"]):
                    A = self.radio.slot_links(ids)
                    self.radio.broadcast(N, d, "aux")
                    Z = metropolis(A) @ Z
                    yield
                Q = Q / (N * np.maximum(Z[act], 1e-300))
            P_new = Q
            # (4)-(5) certificates and trigger (one slot)
            v = self.V(P_new, Ck, lam_k)
            fire = (vlast - v) > c["delta"]
            self.radio.slot_links(ids)
            self.radio.broadcast(int(fire.sum()), d, "alloc")
            vlast[fire] = v[fire]
            A_k = v.sum() - W_k          # algorithmic term V_{lam_k,C^k}(P^{k+1}) - V_{lam_k,C^k}(P^k)
            self.ledger.append(dict(k=k, W_k=W_k, A_k=A_k, lam_k=lam_k, lam_next=self.lam(k + 1),
                                    P_next=P_new.copy(), C_k=Ck.copy(), n=n, fires=int(fire.sum()),
                                    proj_residual=float(np.abs(P_new.sum(0) - 1).max())))
            P = P_new
            self.assignment = round_greedy(P)
            a = np.argmax(P, 1)
            stable = np.where(a == amax, stable + 1, 0); amax = a
            k += 1
            yield
            if (stable >= c["stop_K"]).all():
                self.radio.slot_links(ids)
                self.radio.broadcast(n, 1, "alloc")   # each robot announces it is done
                while True:
                    yield


# ----------------------------------------------------------------------------
# Pose access: the Robotarium allows exactly one get_poses() per step().
# ----------------------------------------------------------------------------
class Poses:
    def __init__(self, r):
        self.r = r
        self.x = None

    def get(self):
        if self.x is None:
            self.x = self.r.get_poses().copy()
        return self.x

    def step(self, dxu):
        self.get()                       # a step must follow a get_poses()
        self.r.set_velocities(np.arange(dxu.shape[1]), dxu)
        self.r.step()
        self.x = None


# ----------------------------------------------------------------------------
# Mission
# ----------------------------------------------------------------------------
def make_mission(seed, n):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)    # rps helpers use the global RNG
    for _ in range(1000):
        x0 = generate_random_poses(n, spacing=0.4, width=2.9, height=1.7)
        sites = generate_random_poses(n, spacing=0.4, width=2.9, height=1.7)[:2]
        d = np.linalg.norm(x0[:2, :, None] - x0[:2, None, :], axis=0)
        A = (d <= R_COMM) & ~np.eye(n, dtype=bool)
        reach = np.zeros(n, bool); reach[0] = True
        for _ in range(n):
            reach = reach | A[reach].any(0)
        if reach.all():          # radio graph connected at the start
            return x0, sites
    raise RuntimeError("could not draw a connected initial configuration")


def cost_matrix(x, sites, robots, tasks):
    dx = sites[:, tasks][:, None, :] - x[:2, robots][:, :, None]     # 2 x n x d
    dist = np.linalg.norm(dx, axis=0)
    head = np.arctan2(dx[1], dx[0]) - x[2, robots][:, None]
    head = np.abs(np.arctan2(np.sin(head), np.cos(head)))
    return COST_SCALE * (dist / 0.2 + HEADING_WEIGHT * head)


def threshold_uni(dxu, r):
    """Same wheel-speed saturation as the simulator, so that the nominal
    prediction differs from the measured pose only by the barrier filter
    (in simulation) or by the barrier filter plus actuation/localization
    error (on hardware)."""
    return r._threshold(dxu)


def unicycle_predict(x, dxu, dt):
    th = x[2]
    th1 = th + dt * dxu[1]
    return np.vstack([x[0] + dt * dxu[0] * np.cos(th), x[1] + dt * dxu[0] * np.sin(th),
                      np.arctan2(np.sin(th1), np.cos(th1))])


def run_mission(r, ps, barrier, ctrl, method, p_loss, seed, x0, sites):
    n = N_ROBOTS
    rng = np.random.default_rng(10_000 + seed)
    radio = Radio(rng, p_loss, R_COMM)
    dt = r.TIME_STEP
    active_r = list(range(n)); active_t = list(range(n))
    state = dict(x=ps.get())

    def costs():
        return cost_matrix(state["x"], sites, active_r, active_t)

    def new_alloc():
        if method == "greedy": return Greedy(radio)
        if method == "cbaa": return CBAA(radio)
        if method == "auction": return Auction(radio)
        return Proposed(radio, PROPOSED_CONFIGS[method], costs)

    def wiring():
        if RELAY_RETIRED:
            return np.arange(n), np.array(active_r)
        return np.array(active_r), np.arange(len(active_r))

    alloc = new_alloc()
    gen = alloc.run(costs(), *wiring())
    alloc_list = [alloc]
    t = 0.0; steps = 0
    first_cf = None; done_time = None
    pos_mot = 0.0; pos_safe = 0.0; pos_safe_active = 0.0
    barrier_active_steps = 0
    trace = []   # (t, robot, task) completions
    traj = [ps.get()[:3].copy()]   # measured poses, one per control step (for figures)
    bar_active = []
    epoch = 0

    while t < MAX_MISSION_TIME and active_r:
        x = ps.get(); state["x"] = x
        radio.set_positions(x[:2])
        for _ in range(SLOTS_PER_STEP):
            next(gen)
        a_local = alloc.assignment
        assign_global = {}
        if a_local is not None and len(a_local) == len(active_r):
            for li, rid in enumerate(active_r):
                if 0 <= a_local[li] < len(active_t):
                    assign_global[rid] = active_t[a_local[li]]
        if first_cf is None and len(assign_global) == len(active_r) \
                and len(set(assign_global.values())) == len(active_r):
            first_cf = t

        goals = x[:2].copy()
        for rid, tid in assign_global.items():
            goals[:, rid] = sites[:, tid]
        dxu_nom = ctrl(x, goals)
        retired = [i for i in range(n) if i not in active_r]
        dxu_nom[:, retired] = 0.0
        dxu = barrier(dxu_nom.copy(), x)
        dxu_free = unconstrained_filter(dxu_nom.copy(), x)
        dxu_free_t = threshold_uni(dxu_free.copy(), r)
        dxu_t = threshold_uni(dxu.copy(), r)
        active_flag = bool(np.abs(dxu_t - dxu_free_t).max() > 1e-3)   # > 1 mm/s or 1 mrad/s
        barrier_active_steps += active_flag
        xhat = unicycle_predict(x, dxu_free_t, dt)   # pose without the safety constraints

        ps.step(dxu)
        t += dt; steps += 1
        x1 = ps.get()

        # price of safety on the assignment actually being executed (all methods)
        for rid, tid in assign_global.items():
            c0 = cost_matrix(x, sites, [rid], [tid])[0, 0]
            ch = cost_matrix(xhat, sites, [rid], [tid])[0, 0]
            c1 = cost_matrix(x1, sites, [rid], [tid])[0, 0]
            pos_mot += ch - c0
            pos_safe += c1 - ch
            if active_flag:
                pos_safe_active += c1 - ch
        traj.append(x1[:3].copy()); bar_active.append(active_flag)

        # completions (structural events)
        completed = [rid for rid, tid in assign_global.items()
                     if np.linalg.norm(x1[:2, rid] - sites[:, tid]) <= AT_TASK]
        if completed:
            for rid in completed:
                tid = assign_global[rid]
                if tid in active_t and rid in active_r:
                    trace.append((t, rid, tid))
                    active_r.remove(rid); active_t.remove(tid)
            if active_r:
                state["x"] = x1
                epoch += 1
                alloc = new_alloc(); alloc_list.append(alloc)
                gen = alloc.run(costs(), *wiring())
            else:
                done_time = t

    # ---- ledger (proposed only): close each iteration with C^{k+1}
    ledger = dict(A=[], schedule=[], G=[], W=[], identity_err=[], epoch=[])
    if method.startswith("proposed"):
        for al in alloc_list:
            L = al.ledger
            for k in range(len(L) - 1):     # within one epoch only (inter-event interval)
                cur, nxt = L[k], L[k + 1]
                P1 = cur["P_next"]
                sched = (cur["lam_next"] - cur["lam_k"]) * float(
                    -(np.where(P1 > 0, P1 * np.log(np.maximum(P1, 1e-300)), 0)).sum())
                G = float(((nxt["C_k"] - cur["C_k"]) * P1).sum())
                W1 = nxt["W_k"]
                ident = W1 - cur["W_k"] - (cur["A_k"] + sched + G)
                ledger["A"].append(cur["A_k"]); ledger["schedule"].append(sched)
                ledger["G"].append(G); ledger["W"].append(cur["W_k"]); ledger["identity_err"].append(ident)
                ledger["epoch"].append(alloc_list.index(al))

    # H3 check, per epoch (inter-event interval): triggered broadcasts after the
    # epoch's opening broadcast vs (W_first - W_min + sum G+) / (kappa delta),
    # with the paper's kappa = 0.93. Summed over epochs.
    h3 = None
    if method.startswith("proposed"):
        msgs = 0; budget = 0.0
        delta = PROPOSED_CONFIGS[method]["delta"]
        for al in alloc_list:
            L = al.ledger
            if len(L) < 2:
                continue
            W = np.array([e["W_k"] for e in L])
            Gp = sum(max(0.0, float(((L[k + 1]["C_k"] - L[k]["C_k"]) * L[k]["P_next"]).sum()))
                     for k in range(len(L) - 1))
            budget += (W[0] - W.min() + Gp) / (0.93 * delta)
            msgs += sum(e["fires"] for e in L[1:])
        h3 = dict(alloc_msgs=int(msgs), budget=float(budget))

    out = dict(method=method, p_loss=p_loss, seed=seed, n=n,
               tasks_done=len(trace), mission_time=(done_time if done_time is not None else t),
               first_cf=first_cf if first_cf is not None else np.nan,
               alloc_kB=radio.alloc_bytes / 1000.0, aux_kB=radio.aux_bytes / 1000.0,
               total_kB=(radio.alloc_bytes + radio.aux_bytes) / 1000.0,
               alloc_msgs=radio.alloc_msgs, slots=radio.slots,
               pos_motion=pos_mot, pos_safe=pos_safe, pos_safe_active=pos_safe_active,
               barrier_active_frac=barrier_active_steps / max(1, steps),
               max_A=float(np.max(ledger["A"])) if ledger["A"] else np.nan,
               max_identity_err=float(np.max(np.abs(ledger["identity_err"]))) if ledger["identity_err"] else np.nan,
               n_iters=sum(len(a.ledger) for a in alloc_list if hasattr(a, "ledger")),
               h3=h3, trace=trace)
    np.savez(f"{SAVE_PREFIX}_{method}_p{int(100*p_loss)}_s{seed}_n{n}.npz",
             summary=np.array([out], dtype=object),
             ledger_A=np.array(ledger["A"]), ledger_G=np.array(ledger["G"]),
             ledger_schedule=np.array(ledger["schedule"]), ledger_W=np.array(ledger["W"]),
             ledger_epoch=np.array(ledger["epoch"]), traj=np.array(traj), sites=sites,
             barrier_active=np.array(bar_active), completions=np.array(trace).reshape(-1, 3))
    return out


def drive_to(r, ps, barrier, x_target, tmax=40.0):
    """Reset phase between missions (not part of any measurement)."""
    pose_ctrl = create_pose_controller_hybrid()
    at_pose = create_at_pose()
    t = 0.0
    while t < tmax:
        x = ps.get()
        if at_pose(x, x_target)[0]:
            return True
        ps.step(barrier(pose_ctrl(x, x_target), x)); t += r.TIME_STEP
    return False


def main():
    n = N_ROBOTS
    missions = [(s, m, p) for s in SEEDS for p in P_LOSS_LEVELS for m in METHODS]
    x0, _ = make_mission(missions[0][0], n)
    r = robotarium.Robotarium(number_of_robots=n, show_figure=False, initial_conditions=x0,
                              sim_in_real_time=False, skip_initialization=SIMULATION)
    barrier = create_uni_barrier_certificate_with_boundary()
    ctrl = create_uni_position_controller()
    ps = Poses(r)
    results = []
    for seed, method, p_loss in missions:
        x0, sites = make_mission(seed, n)
        drive_to(r, ps, barrier, x0)
        out = run_mission(r, ps, barrier, ctrl, method, p_loss, seed, x0, sites)
        results.append(out)
        print("RESULT", {k: v for k, v in out.items() if k != "trace"}, flush=True)
    np.save(f"{SAVE_PREFIX}_summary_n{n}.npy", np.array(results, dtype=object), allow_pickle=True)
    r.debug()


if __name__ == "__main__":
    main()
