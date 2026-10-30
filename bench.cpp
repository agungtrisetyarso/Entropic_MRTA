// bench.cpp -- allocation-layer benchmark for the revised manuscript.
//
// Every method runs on the SAME fixed-seed instances and the SAME random
// geometric communication graph, and every message is counted in one unit:
// a local broadcast (one transmission heard by all one-hop neighbours).
// Bytes use the payload model of Sec. 6.1: 4 bytes per scalar + 20-byte header.
//
// Methods
//   EMD-Auto (proposed)           Alg. 2 with Alg. 1 consensus Sinkhorn on the graph
//   EMD-Auto, exact projection    same, column sums computed exactly (no consensus)
//   Greedy nearest-task           sequential claims by robot index, claims flooded
//   CBAA                          Choi, Brunet & How (2009), single-assignment
//   Distributed auction           Bertsekas auction, Jacobi bidding + max-consensus
//   Convex entropic OT            Sinkhorn on exp(-C/eps) (entropy SUBTRACTED), rounded
//   Centralized Hungarian         optimality reference
//
// Static instances only: no motion, no time windows, no task arrivals. This is
// NOT the Gazebo benchmark of Table 4 and its numbers must not be mixed with it.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;
static const double BIG = 1e18;

// ------------------------------------------------------------------ instances
// Same generator as main.cpp (uniform base cost, 20% capability-mismatch
// penalty, additive time-window pressure), but with an explicit seed.
Mat gen_cost(int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> base(5.0, 40.0), pen(0.0, 30.0), u(0.0, 1.0);
    Mat C(n, Vec(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            C[i][j] = base(rng);
            if (u(rng) < 0.2) C[i][j] += 50.0 + pen(rng);
            C[i][j] += pen(rng) * 0.3;
        }
    return C;
}

struct Graph {
    int n;
    std::vector<std::vector<int>> nbr;
    Mat W;          // Metropolis weights (doubly stochastic)
    int diameter;
};

Graph gen_graph(int n, double radius, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (;;) {
        std::vector<double> x(n), y(n);
        for (int i = 0; i < n; ++i) { x[i] = u(rng); y[i] = u(rng); }
        Graph g; g.n = n; g.nbr.assign(n, {});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j && std::hypot(x[i] - x[j], y[i] - y[j]) <= radius) g.nbr[i].push_back(j);
        // BFS for connectivity and diameter
        int diam = 0; bool connected = true;
        for (int s = 0; s < n && connected; ++s) {
            std::vector<int> d(n, -1); d[s] = 0; std::vector<int> q{s};
            for (size_t h = 0; h < q.size(); ++h)
                for (int v : g.nbr[q[h]]) if (d[v] < 0) { d[v] = d[q[h]] + 1; q.push_back(v); }
            for (int v = 0; v < n; ++v) { if (d[v] < 0) connected = false; diam = std::max(diam, d[v]); }
        }
        if (!connected) continue;
        g.diameter = diam;
        g.W.assign(n, Vec(n, 0.0));
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int j : g.nbr[i]) {
                double w = 1.0 / (1.0 + std::max(g.nbr[i].size(), g.nbr[j].size()));
                g.W[i][j] = w; s += w;
            }
            g.W[i][i] = 1.0 - s;
        }
        return g;
    }
}

// ------------------------------------------------------------------ helpers
double hungarian(const Mat& cost, std::vector<int>* assign = nullptr) {
    int n = cost.size();
    Vec u(n + 1, 0), v(n + 1, 0), minv;
    std::vector<int> p(n + 1, 0), way(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i; int j0 = 0;
        minv.assign(n + 1, BIG);
        std::vector<bool> used(n + 1, false);
        do {
            used[j0] = true; int i0 = p[j0], j1 = 0; double delta = BIG;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
                double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    double total = 0;
    if (assign) assign->assign(n, -1);
    for (int j = 1; j <= n; ++j) { total += cost[p[j] - 1][j - 1]; if (assign) (*assign)[p[j] - 1] = j - 1; }
    return total;
}

// Rounding used for the entropic methods: greedy by largest P_ij (as in emd.cpp).
std::vector<int> round_greedy(const Mat& P) {
    int n = P.size();
    std::vector<std::tuple<double, int, int>> c;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) c.emplace_back(P[i][j], i, j);
    std::sort(c.rbegin(), c.rend());
    std::vector<int> a(n, -1); std::vector<bool> used(n, false);
    for (auto& [p, i, j] : c) if (a[i] < 0 && !used[j]) { a[i] = j; used[j] = true; }
    return a;
}

bool is_perm(const std::vector<int>& a) {
    std::vector<bool> used(a.size(), false);
    for (int j : a) { if (j < 0 || j >= (int)a.size() || used[j]) return false; used[j] = true; }
    return true;
}

double cost_of(const Mat& C, const std::vector<int>& a) {
    double s = 0; for (size_t i = 0; i < a.size(); ++i) s += C[i][a[i]]; return s;
}

void rownorm(Vec& r) { double s = std::accumulate(r.begin(), r.end(), 0.0); if (s > 0) for (auto& x : r) x /= s; }

struct Result {
    bool conflict_free = false;   // a valid permutation was produced
    bool converged = false;       // method's own stopping criterion met within budget
    double cost = 0;
    int iters = 0;                // method iterations (EMD/OT) or communication rounds (auctions)
    long alloc_bc = 0;            // allocation-layer broadcasts
    long aux_bc = 0;              // auxiliary broadcasts (consensus, relays)
    double bytes = 0;             // all streams
    double max_dW = 0;            // EMD only: largest relative increase of W_k
    // Table 5 accounting (radio slots = synchronous broadcast rounds)
    double alloc_bytes = 0, aux_bytes = 0;
    long rounds = 0;              // radio slots used until the method stops
    long first_cf_round = -1;     // first slot at which a conflict-free assignment is available
    long opt_round = -1;          // first slot after which the held assignment is optimal and stays so (-1: never)
};

// ------------------------------------------------------------------ proposed
struct EMDCfg {
    double alpha = 0.08, beta = 0.35, lam0 = 0.5, laminf = 0.02;
    int Kf = 200, M = 15, T = 5, max_iter = 500;
    double delta = 0.05, purity = 0.99;
    bool exact_proj = false;
    int stable_K = 0;   // >0: stop when every robot's own argmax is unchanged for K iterations (local rule)
};

Result run_emd(const Mat& C, const Graph& g, const EMDCfg& cf, double opt = NAN) {
    int n = C.size(), d = n;
    Mat P(n, Vec(d, 1.0 / d));
    Vec vlast(n, std::numeric_limits<double>::infinity());
    Result r;
    double Wprev = std::numeric_limits<double>::quiet_NaN();
    double lam_prev = cf.lam0;
    std::vector<int> amax(n, -1), stable_for(n, 0);
    auto lam_of = [&](int k) { return k >= cf.Kf ? cf.laminf : std::max(cf.laminf, cf.lam0 / std::log(k + 2.0)); };
    for (int k = 0; k < cf.max_iter; ++k) {
        double lam = lam_of(k);
        // (1)-(2) mirror step + dual damping
        for (int i = 0; i < n; ++i) {
            Vec t(d), h(d);
            for (int j = 0; j < d; ++j) t[j] = std::pow(std::max(P[i][j], 1e-300), 1 + cf.alpha * lam) * std::exp(-cf.alpha * C[i][j]);
            rownorm(t);
            for (int j = 0; j < d; ++j) h[j] = std::pow(std::max(t[j], 1e-300), 1 - cf.beta) * std::pow(std::max(P[i][j], 1e-300), cf.beta);
            rownorm(h); P[i] = h;
        }
        // (3) Alg. 1: M sweeps, each = local row scaling + T consensus rounds + local column scaling
        for (int m = 0; m < cf.M; ++m) {
            for (int i = 0; i < n; ++i) rownorm(P[i]);
            Mat z = P;
            if (cf.exact_proj) {
                Vec avg(d, 0); for (int i = 0; i < n; ++i) for (int j = 0; j < d; ++j) avg[j] += P[i][j] / n;
                for (int i = 0; i < n; ++i) z[i] = avg;
            } else {
                for (int t = 0; t < cf.T; ++t) {
                    Mat z2(n, Vec(d, 0));
                    for (int i = 0; i < n; ++i)
                        for (int k2 = 0; k2 < n; ++k2) if (g.W[i][k2] != 0)
                            for (int j = 0; j < d; ++j) z2[i][j] += g.W[i][k2] * z[k2][j];
                    z = z2;
                    r.aux_bc += n;   // every robot broadcasts its d-vector once per round
                }
            }
            for (int i = 0; i < n; ++i) for (int j = 0; j < d; ++j) P[i][j] /= (n * std::max(z[i][j], 1e-300));
        }
        // (4) certificates
        double W = 0;
        Vec v(n);
        for (int i = 0; i < n; ++i) {
            double c = 0, e = 0;
            for (int j = 0; j < d; ++j) { c += P[i][j] * C[i][j]; if (P[i][j] > 0) e -= P[i][j] * std::log(P[i][j]); }
            v[i] = c + lam * e; W += v[i];
        }
        if (k > 0 && std::isfinite(Wprev)) r.max_dW = std::max(r.max_dW, (W - Wprev) / std::max(1.0, std::fabs(Wprev)));
        Wprev = W; lam_prev = lam;
        // (5) event trigger
        for (int i = 0; i < n; ++i) if (vlast[i] - v[i] > cf.delta) { r.alloc_bc++; vlast[i] = v[i]; }
        // stopping: min row purity > threshold
        double minpur = 1;
        for (int i = 0; i < n; ++i) { double s = 0; for (double p : P[i]) s += p * p; minpur = std::min(minpur, s); }
        r.iters = k + 1;
        {   // Table 5 tracking: anytime rounding (Cor. rounding) each iteration
            long per_it = (cf.exact_proj ? 0 : (long)cf.M * cf.T) + 1;
            r.rounds = (long)r.iters * per_it;
            auto a = round_greedy(P);
            if (r.first_cf_round < 0 && is_perm(a)) r.first_cf_round = r.rounds;
            if (!std::isnan(opt)) {
                bool is_opt = is_perm(a) && cost_of(C, a) <= opt + 1e-7;
                if (is_opt && r.opt_round < 0) r.opt_round = r.rounds;
                if (!is_opt) r.opt_round = -1;
            }
        }
        if (cf.stable_K > 0) {
            bool all = true;
            for (int i = 0; i < n; ++i) {
                int a = (int)(std::max_element(P[i].begin(), P[i].end()) - P[i].begin());
                stable_for[i] = (a == amax[i]) ? stable_for[i] + 1 : 0; amax[i] = a;
                if (stable_for[i] < cf.stable_K) all = false;
            }
            if (all) { r.converged = true; r.alloc_bc += n; break; }   // each robot announces it is done
            continue;
        }
        if (minpur > cf.purity) { r.converged = true; break; }
    }
    (void)lam_prev;
    auto a = round_greedy(P);
    r.conflict_free = is_perm(a);
    r.cost = cost_of(C, a);
    r.bytes = (r.alloc_bc + r.aux_bc) * (4.0 * d + 20.0);
    r.alloc_bytes = r.alloc_bc * (4.0 * d + 20.0);
    r.aux_bytes = r.aux_bc * (4.0 * d + 20.0);
    return r;
}

// ------------------------------------------------------------------ greedy
// Robot i claims its cheapest free task after hearing claims 0..i-1.
// Each claim is flooded: every robot relays it once (n broadcasts per claim).
Result run_greedy(const Mat& C, const Graph& g) {
    int n = C.size(); Result r;
    std::vector<bool> used(n, false); std::vector<int> a(n, -1);
    for (int i = 0; i < n; ++i) {
        int bj = -1; double best = BIG;
        for (int j = 0; j < n; ++j) if (!used[j] && C[i][j] < best) { best = C[i][j]; bj = j; }
        a[i] = bj; used[bj] = true;
        r.alloc_bc += 1; r.aux_bc += n - 1;
    }
    r.iters = n * g.diameter;  // rounds: each claim needs up to diameter hops before the next robot acts
    r.conflict_free = r.converged = is_perm(a);
    r.cost = cost_of(C, a);
    r.bytes = (r.alloc_bc + r.aux_bc) * (4.0 * 2 + 20.0);   // payload: (robot, task)
    r.alloc_bytes = r.alloc_bc * 28.0; r.aux_bytes = r.aux_bc * 28.0;
    r.rounds = r.first_cf_round = r.iters;
    return r;
}

// ------------------------------------------------------------------ CBAA
// Choi, Brunet & How (2009), Algorithms 1-2, single assignment per robot.
// Score s_ij = Cmax + 1 - C_ij > 0 (CBAA maximizes score).
// Per round every robot broadcasts its winning-bid list y_i and winner list z_i.
Result run_cbaa(const Mat& C, const Graph& g, int max_rounds = 2000) {
    int n = C.size(); Result r;
    double cmax = 0; for (auto& row : C) for (double c : row) cmax = std::max(cmax, c);
    Mat s(n, Vec(n)); for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) s[i][j] = cmax + 1 - C[i][j];
    Mat y(n, Vec(n, 0.0)); std::vector<std::vector<int>> z(n, std::vector<int>(n, -1));
    std::vector<int> x(n, -1);  // task held by robot i
    int stable = 0;
    for (int round = 1; round <= max_rounds; ++round) {
        // Phase 1: auction
        for (int i = 0; i < n; ++i) if (x[i] < 0) {
            int J = -1; double best = -BIG;
            for (int j = 0; j < n; ++j) {
                bool beats = s[i][j] > y[i][j] || (s[i][j] == y[i][j] && (z[i][j] < 0 || i < z[i][j]));
                if (beats && s[i][j] > best) { best = s[i][j]; J = j; }
            }
            if (J >= 0) { x[i] = J; y[i][J] = s[i][J]; z[i][J] = i; }
        }
        // Phase 2: consensus (synchronous, on copies)
        Mat y2 = y; auto z2 = z;
        for (int i = 0; i < n; ++i)
            for (int k : g.nbr[i])
                for (int j = 0; j < n; ++j)
                    if (y[k][j] > y2[i][j] || (y[k][j] == y2[i][j] && y[k][j] > 0 && z[k][j] >= 0 && (z2[i][j] < 0 || z[k][j] < z2[i][j]))) {
                        y2[i][j] = y[k][j]; z2[i][j] = z[k][j];
                    }
        bool changed = (y2 != y) || (z2 != z);
        y = y2; z = z2;
        for (int i = 0; i < n; ++i) if (x[i] >= 0 && z[i][x[i]] != i) { x[i] = -1; changed = true; }
        r.alloc_bc += n;   // each robot broadcasts (y_i, z_i) every round
        r.iters = round;
        bool all = std::all_of(x.begin(), x.end(), [](int t) { return t >= 0; });
        stable = (all && !changed) ? stable + 1 : 0;
        if (stable >= g.diameter + 1) { r.converged = true; break; }
    }
    r.conflict_free = is_perm(x);
    r.cost = r.conflict_free ? cost_of(C, x) : NAN;
    r.bytes = r.alloc_bc * (4.0 * 2 * n + 20.0);
    r.alloc_bytes = r.bytes;          // bid and winner lists ARE the allocation messages
    r.rounds = r.iters; if (r.conflict_free) r.first_cf_round = r.iters;
    return r;
}

// ------------------------------------------------------------------ distributed auction
// Bertsekas auction (benefit a_ij = -C_ij), Jacobi bidding, prices and
// highest-bidder identities propagated by max-consensus over the graph.
// Each robot broadcasts its (price, bidder) lists every round.
Result run_auction(const Mat& C, const Graph& g, double eps, int max_rounds = 20000) {
    int n = C.size(); Result r;
    Mat p(n, Vec(n, 0.0)); std::vector<std::vector<int>> b(n, std::vector<int>(n, -1));
    int stable = 0;
    for (int round = 1; round <= max_rounds; ++round) {
        // a robot believes it holds task j if it is the recorded bidder in its own view
        for (int i = 0; i < n; ++i) {
            bool holds = false;
            for (int j = 0; j < n; ++j) if (b[i][j] == i) holds = true;
            if (holds) continue;
            int j1 = -1; double v1 = -BIG, v2 = -BIG;
            for (int j = 0; j < n; ++j) {
                double val = -C[i][j] - p[i][j];
                if (val > v1) { v2 = v1; v1 = val; j1 = j; } else if (val > v2) v2 = val;
            }
            double bid = p[i][j1] + (v1 - v2) + eps;
            p[i][j1] = bid; b[i][j1] = i;
        }
        Mat p2 = p; auto b2 = b;
        for (int i = 0; i < n; ++i)
            for (int k : g.nbr[i])
                for (int j = 0; j < n; ++j)
                    if (p[k][j] > p2[i][j] || (p[k][j] == p2[i][j] && b[k][j] > b2[i][j])) { p2[i][j] = p[k][j]; b2[i][j] = b[k][j]; }
        bool changed = (p2 != p) || (b2 != b);
        p = p2; b = b2;
        r.alloc_bc += n; r.iters = round;
        // global check (simulation bookkeeping only): every robot holds a task in its own view
        std::vector<int> a(n, -1);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) if (b[i][j] == i) a[i] = j;
        bool all = is_perm(a);
        stable = (all && !changed) ? stable + 1 : 0;
        if (stable >= g.diameter + 1) {
            r.converged = r.conflict_free = true; r.cost = cost_of(C, a); break;
        }
    }
    if (!r.converged) r.cost = NAN;
    r.bytes = r.alloc_bc * (4.0 * 2 * n + 20.0);
    r.alloc_bytes = r.bytes;
    r.rounds = r.iters; if (r.conflict_free) r.first_cf_round = r.iters;
    return r;
}

// ------------------------------------------------------------------ convex entropic OT
// Sinkhorn on K = exp(-C/eps): the entropy is SUBTRACTED (convex, biased),
// column sums by the same T-round consensus as Alg. 1. Stops when the
// marginal residual falls below tol; each robot then broadcasts its row once.
Result run_entropic_ot(const Mat& C, const Graph& g, double eps, int T, double tol, int max_sweeps = 5000, double opt = NAN) {
    int n = C.size(), d = n; Result r;
    double cmin = BIG; for (auto& row : C) for (double c : row) cmin = std::min(cmin, c);
    Mat P(n, Vec(d)); for (int i = 0; i < n; ++i) for (int j = 0; j < d; ++j) P[i][j] = std::exp(-(C[i][j] - cmin) / eps);
    for (int sw = 1; sw <= max_sweeps; ++sw) {
        for (int i = 0; i < n; ++i) rownorm(P[i]);
        Mat z = P;
        for (int t = 0; t < T; ++t) {
            Mat z2(n, Vec(d, 0));
            for (int i = 0; i < n; ++i) for (int k = 0; k < n; ++k) if (g.W[i][k] != 0)
                for (int j = 0; j < d; ++j) z2[i][j] += g.W[i][k] * z[k][j];
            z = z2; r.aux_bc += n;
        }
        for (int i = 0; i < n; ++i) for (int j = 0; j < d; ++j) P[i][j] /= (n * std::max(z[i][j], 1e-300));
        double res = 0;
        for (int i = 0; i < n; ++i) res = std::max(res, std::fabs(std::accumulate(P[i].begin(), P[i].end(), 0.0) - 1));
        for (int j = 0; j < d; ++j) { double s = 0; for (int i = 0; i < n; ++i) s += P[i][j]; res = std::max(res, std::fabs(s - 1)); }
        r.iters = sw;
        r.rounds = (long)sw * T + 1;
        {
            auto a = round_greedy(P);
            if (r.first_cf_round < 0 && is_perm(a)) r.first_cf_round = r.rounds;
            if (!std::isnan(opt)) {
                bool is_opt = is_perm(a) && cost_of(C, a) <= opt + 1e-7;
                if (is_opt && r.opt_round < 0) r.opt_round = r.rounds;
                if (!is_opt) r.opt_round = -1;
            }
        }
        if (res < tol) { r.converged = true; break; }
    }
    r.alloc_bc = n;
    auto a = round_greedy(P);
    r.conflict_free = is_perm(a);
    r.cost = cost_of(C, a);
    r.bytes = (r.alloc_bc + r.aux_bc) * (4.0 * d + 20.0);
    r.alloc_bytes = r.alloc_bc * (4.0 * d + 20.0);
    r.aux_bytes = r.aux_bc * (4.0 * d + 20.0);
    return r;
}

// ------------------------------------------------------------------ driver
struct Acc {
    std::vector<double> cost, gap, iters, alloc, total, kb, maxdw;
    int cf = 0, conv = 0, runs = 0;
    void add(const Result& r, double opt) {
        ++runs; if (r.conflict_free) ++cf; if (r.converged) ++conv;
        if (r.conflict_free) { cost.push_back(r.cost); gap.push_back(100.0 * (r.cost - opt) / opt); }
        iters.push_back(r.iters); alloc.push_back(r.alloc_bc); total.push_back(r.alloc_bc + r.aux_bc);
        kb.push_back(r.bytes / 1000.0); maxdw.push_back(r.max_dW);
    }
};
std::pair<double, double> ms(const std::vector<double>& v) {
    if (v.empty()) return {NAN, NAN};
    double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size(), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return {m, v.size() > 1 ? std::sqrt(s / (v.size() - 1)) : 0.0};
}

int sweep_main(int n, int runs, double radius) {
    std::printf("Sweep: n=%d runs=%d radius=%.2f, alpha=0.08 paper params, max 5000 it\n", n, runs, radius);
    std::printf("%-22s %5s %5s %12s %14s %12s %10s\n", "Variant", "CF%", "Conv%", "Gap%", "Iters", "Total bc", "kB");
    std::vector<int> Ts = {5, 10, 20, 50, 100, -1};
    for (int T : Ts) {
        EMDCfg cf; cf.max_iter = 5000; if (T < 0) cf.exact_proj = true; else cf.T = T;
        Acc a; double diam = 0;
        for (int s = 1; s <= runs; ++s) {
            std::mt19937 rng(1000 + s);
            Mat C = gen_cost(n, rng); Graph g = gen_graph(n, radius, rng); diam += g.diameter;
            a.add(run_emd(C, g, cf), hungarian(C));
        }
        auto gp = ms(a.gap), it = ms(a.iters), tt = ms(a.total), kb = ms(a.kb);
        char lab[64]; std::snprintf(lab, 64, T < 0 ? "exact projection" : "T=%d", T);
        std::printf("%-22s %5.0f %5.0f %5.2f+/-%-5.2f %6.0f+/-%-6.0f %10.0f %10.1f   (mean diam %.1f)\n", lab,
                    100.0 * a.cf / runs, 100.0 * a.conv / runs, gp.first, gp.second, it.first, it.second,
                    tt.first, kb.first, diam / runs);
    }
    return 0;
}

// ------------------------------------------------------------------ Table 5
// Per-method radio split, time to first conflict-free assignment, time to
// a held optimal assignment, and gap. Writes one CSV row per (seed, method).
int table5_main(int n, int runs, double radius, const char* csv) {
    struct M5 { std::string name; std::function<Result(const Mat&, const Graph&, double)> f; };
    auto emd = [](int M, int T, int maxit) {
        return [=](const Mat& C, const Graph& g, double opt) {
            EMDCfg cf; cf.M = M; cf.T = T; cf.max_iter = maxit; return run_emd(C, g, cf, opt); };
    };
    std::vector<M5> meths = {
        {"Greedy nearest-task", [](const Mat& C, const Graph& g, double) { return run_greedy(C, g); }},
        {"CBAA", [](const Mat& C, const Graph& g, double) { return run_cbaa(C, g); }},
        {"Distributed auction (eps=0.01)", [](const Mat& C, const Graph& g, double) { return run_auction(C, g, 0.01); }},
        {"Convex entropic OT (eps=1, T=5)", [](const Mat& C, const Graph& g, double o) { return run_entropic_ot(C, g, 1.0, 5, 1e-6, 2000, o); }},
        {"Proposed M=15 T=5", emd(15, 5, 500)},
        {"Proposed M=5 T=2", emd(5, 2, 500)},
        {"Proposed M=100 T=100", emd(100, 100, 300)},
        {"Proposed M=5 T=2, local stop K=20", [](const Mat& C, const Graph& g, double o) {
            EMDCfg cf; cf.M = 5; cf.T = 2; cf.max_iter = 500; cf.stable_K = 20; return run_emd(C, g, cf, o); }},
        {"Proposed M=15 T=50, local stop K=20", [](const Mat& C, const Graph& g, double o) {
            EMDCfg cf; cf.M = 15; cf.T = 50; cf.max_iter = 500; cf.stable_K = 20; return run_emd(C, g, cf, o); }},
    };
    FILE* f = csv ? std::fopen(csv, "w") : nullptr;
    if (f) std::fprintf(f, "seed,method,conflict_free,converged,gap_pct,alloc_kB,aux_kB,total_kB,rounds,first_cf_round,opt_round\n");
    struct A { std::vector<double> al, ax, tt, cf, op, gap, rd; int optc = 0, runs = 0; };
    std::vector<A> acc(meths.size());
    for (int s = 1; s <= runs; ++s) {
        std::mt19937 rng(1000 + s);
        Mat C = gen_cost(n, rng); Graph g = gen_graph(n, radius, rng);
        double opt = hungarian(C);
        for (size_t m = 0; m < meths.size(); ++m) {
            Result r = meths[m].f(C, g, opt);
            double gap = 100.0 * (r.cost - opt) / opt;
            A& a = acc[m]; a.runs++;
            a.al.push_back(r.alloc_bytes / 1000); a.ax.push_back(r.aux_bytes / 1000);
            a.tt.push_back((r.alloc_bytes + r.aux_bytes) / 1000); a.gap.push_back(gap); a.rd.push_back(r.rounds);
            if (r.first_cf_round >= 0) a.cf.push_back(r.first_cf_round);
            if (gap <= 1e-6) { a.optc++; a.op.push_back(r.opt_round >= 0 ? r.opt_round : r.rounds); }
            if (f) std::fprintf(f, "%d,\"%s\",%d,%d,%.6f,%.4f,%.4f,%.4f,%ld,%ld,%ld\n", s, meths[m].name.c_str(),
                                r.conflict_free, r.converged, gap, r.alloc_bytes / 1000, r.aux_bytes / 1000,
                                (r.alloc_bytes + r.aux_bytes) / 1000, r.rounds, r.first_cf_round, r.opt_round);
        }
    }
    if (f) std::fclose(f);
    auto med = [](std::vector<double> v) -> double { if (v.empty()) return NAN; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    std::printf("Table 5 mode: n=%d runs=%d radius=%.2f (medians; gap mean+/-sd)\n", n, runs, radius);
    std::printf("%-32s %10s %12s %12s %10s %12s %10s %14s\n", "Method", "Alloc kB", "Aux kB", "Total kB", "1st CF",
                "To optimum", "Rounds", "Gap %");
    for (size_t m = 0; m < meths.size(); ++m) {
        A& a = acc[m]; auto g = ms(a.gap);
        std::printf("%-32s %10.1f %12.1f %12.1f %10.0f %8.0f(%2d) %10.0f %6.2f+/-%-6.2f\n", meths[m].name.c_str(),
                    med(a.al), med(a.ax), med(a.tt), med(a.cf), med(a.op), a.optc, med(a.rd), g.first, g.second);
    }
    return 0;
}

// Lever study: total traffic vs accuracy over (M, T) for the proposed method.
int lever_main(int n, int runs, double radius) {
    std::printf("Lever: n=%d runs=%d radius=%.2f, 500 iterations max\n", n, runs, radius);
    std::printf("%4s %4s %12s %14s %10s\n", "M", "T", "Total MB", "Gap %", "Optimal");
    for (int M : {5, 15, 50})
        for (int T : {2, 5, 20, 50}) {
            EMDCfg cf; cf.M = M; cf.T = T; cf.max_iter = 500;
            std::vector<double> tb, gp; int optc = 0;
            for (int s = 1; s <= runs; ++s) {
                std::mt19937 rng(1000 + s);
                Mat C = gen_cost(n, rng); Graph g = gen_graph(n, radius, rng);
                double opt = hungarian(C);
                Result r = run_emd(C, g, cf, opt);
                tb.push_back((r.alloc_bytes + r.aux_bytes) / 1e6);
                double gap = 100.0 * (r.cost - opt) / opt; gp.push_back(gap); if (gap <= 1e-6) optc++;
            }
            auto a = ms(tb), b = ms(gp);
            std::printf("%4d %4d %12.1f %6.2f+/-%-6.2f %6d/%d\n", M, T, a.first, b.first, b.second, optc, runs);
        }
    return 0;
}

// ------------------------------------------------------------------ Table 7
// Team-size scaling, d = n. Radius shrinks as r_n = r10 * sqrt(9/(n-1)) so the
// expected radio degree stays fixed (the regime of Eq. (14)); pass fixed=1 to
// keep the radius constant instead (degree then grows with n).
int scale_main(int runs, double r10, int fixed, const char* csv) {
    std::vector<int> sizes = {5, 10, 20, 40};
    struct M7 { std::string name; std::function<Result(const Mat&, const Graph&, double)> f; };
    std::vector<M7> meths = {
        {"Greedy nearest-task", [](const Mat& C, const Graph& g, double) { return run_greedy(C, g); }},
        {"CBAA", [](const Mat& C, const Graph& g, double) { return run_cbaa(C, g); }},
        {"Distributed auction (eps=0.01)", [](const Mat& C, const Graph& g, double) { return run_auction(C, g, 0.01); }},
        {"Proposed M=15 T=5", [](const Mat& C, const Graph& g, double o) {
            EMDCfg cf; cf.M = 15; cf.T = 5; cf.max_iter = 500; return run_emd(C, g, cf, o); }},
        {"Proposed M=5 T=50", [](const Mat& C, const Graph& g, double o) {
            EMDCfg cf; cf.M = 5; cf.T = 50; cf.max_iter = 500; return run_emd(C, g, cf, o); }},
        {"Proposed M=5 T=50 local stop K=20", [](const Mat& C, const Graph& g, double o) {
            EMDCfg cf; cf.M = 5; cf.T = 50; cf.max_iter = 500; cf.stable_K = 20; return run_emd(C, g, cf, o); }},
    };
    FILE* f = csv ? std::fopen(csv, "w") : nullptr;
    if (f) std::fprintf(f, "n,seed,method,radius,diameter,mean_degree,conflict_free,converged,gap_pct,total_kB_per_robot,first_cf_round,rounds\n");
    auto med = [](std::vector<double> v) -> double { if (v.empty()) return NAN; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    std::printf("Scale mode: runs=%d, r10=%.2f, %s radius\n", runs, r10, fixed ? "fixed" : "fixed-degree");
    for (int n : sizes) {
        double radius = fixed ? r10 : r10 * std::sqrt(9.0 / (n - 1));
        struct A { std::vector<double> kb, cf, gap; int optc = 0; };
        std::vector<A> acc(meths.size());
        double diam = 0, deg = 0;
        for (int s = 1; s <= runs; ++s) {
            std::mt19937 rng(7000 + 100 * n + s);
            Mat C = gen_cost(n, rng); Graph g = gen_graph(n, radius, rng);
            double md = 0; for (auto& nb : g.nbr) md += nb.size(); md /= n;
            diam += g.diameter; deg += md;
            double opt = hungarian(C);
            for (size_t m = 0; m < meths.size(); ++m) {
                Result r = meths[m].f(C, g, opt);
                double gap = 100.0 * (r.cost - opt) / opt;
                double kbr = (r.alloc_bytes + r.aux_bytes) / 1000.0 / n;
                acc[m].kb.push_back(kbr); acc[m].gap.push_back(gap);
                if (r.first_cf_round >= 0) acc[m].cf.push_back(r.first_cf_round);
                if (gap <= 1e-6) acc[m].optc++;
                if (f) std::fprintf(f, "%d,%d,\"%s\",%.4f,%d,%.3f,%d,%d,%.6f,%.4f,%ld,%ld\n", n, s, meths[m].name.c_str(),
                                    radius, g.diameter, md, r.conflict_free, r.converged, gap, kbr, r.first_cf_round, r.rounds);
            }
        }
        std::printf("\nn=%d  radius=%.3f  mean diameter=%.2f  mean degree=%.2f\n", n, radius, diam / runs, deg / runs);
        std::printf("%-36s %14s %10s %16s %8s\n", "Method", "kB/robot(med)", "1stCF(med)", "Gap% mean+/-sd", "Optimal");
        for (size_t m = 0; m < meths.size(); ++m) {
            auto g = ms(acc[m].gap);
            std::printf("%-36s %14.2f %10.0f %7.2f+/-%-7.2f %4d/%d\n", meths[m].name.c_str(), med(acc[m].kb), med(acc[m].cf),
                        g.first, g.second, acc[m].optc, runs);
        }
        std::fflush(stdout);
    }
    if (f) std::fclose(f);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "scale")
        return scale_main(argc > 2 ? std::atoi(argv[2]) : 30, argc > 3 ? std::atof(argv[3]) : 0.45,
                          argc > 4 ? std::atoi(argv[4]) : 0, argc > 5 ? argv[5] : nullptr);
    if (argc > 1 && std::string(argv[1]) == "table5")
        return table5_main(argc > 2 ? std::atoi(argv[2]) : 10, argc > 3 ? std::atoi(argv[3]) : 50,
                           argc > 4 ? std::atof(argv[4]) : 0.45, argc > 5 ? argv[5] : nullptr);
    if (argc > 1 && std::string(argv[1]) == "lever")
        return lever_main(argc > 2 ? std::atoi(argv[2]) : 10, argc > 3 ? std::atoi(argv[3]) : 20,
                          argc > 4 ? std::atof(argv[4]) : 0.45);
    if (argc > 1 && std::string(argv[1]) == "sweep")
        return sweep_main(argc > 2 ? std::atoi(argv[2]) : 10, argc > 3 ? std::atoi(argv[3]) : 20,
                          argc > 4 ? std::atof(argv[4]) : 0.45);
    int n = argc > 1 ? std::atoi(argv[1]) : 10;
    int runs = argc > 2 ? std::atoi(argv[2]) : 50;
    double radius = argc > 3 ? std::atof(argv[3]) : 0.45;
    const char* csv = argc > 4 ? argv[4] : nullptr;

    EMDCfg paper;                                   // Sec. 8.2 parameters (M=15, T=5)
    EMDCfg m100 = paper; m100.M = 100; m100.max_iter = 1000;
    EMDCfg m100x = m100; m100x.exact_proj = true;   // idealized: exact column sums

    std::vector<std::string> names = {
        "Greedy nearest-task", "CBAA", "Distributed auction (eps=0.01)",
        "Convex entropic OT (eps=1)", "Proposed, paper (M=15,T=5)",
        "Proposed, M=100, T=5", "Proposed, M=100, exact col. sums"};
    std::vector<Acc> acc(names.size());
    std::vector<double> opts;
    FILE* f = csv ? std::fopen(csv, "w") : nullptr;
    if (f) std::fprintf(f, "seed,method,conflict_free,converged,cost,opt,gap_pct,iters,alloc_bc,aux_bc,kbytes,max_dW\n");

    for (int s = 1; s <= runs; ++s) {
        std::mt19937 rng(1000 + s);
        Mat C = gen_cost(n, rng);
        Graph g = gen_graph(n, radius, rng);
        double opt = hungarian(C);
        opts.push_back(opt);
        std::vector<Result> res = {
            run_greedy(C, g), run_cbaa(C, g), run_auction(C, g, 0.01),
            run_entropic_ot(C, g, 1.0, paper.T, 1e-6, 2000),
            run_emd(C, g, paper), run_emd(C, g, m100), run_emd(C, g, m100x)};
        for (size_t m = 0; m < res.size(); ++m) {
            acc[m].add(res[m], opt);
            if (f) std::fprintf(f, "%d,\"%s\",%d,%d,%.6f,%.6f,%.6f,%d,%ld,%ld,%.4f,%.3e\n", s, names[m].c_str(),
                                res[m].conflict_free, res[m].converged, res[m].cost, opt,
                                100.0 * (res[m].cost - opt) / opt, res[m].iters, res[m].alloc_bc, res[m].aux_bc,
                                res[m].bytes / 1000.0, res[m].max_dW);
        }
    }
    if (f) std::fclose(f);

    auto o = ms(opts);
    std::printf("n=%d, runs=%d, radius=%.2f; Hungarian optimum %.1f +/- %.1f\n\n", n, runs, radius, o.first, o.second);
    std::printf("%-36s %5s %5s %13s %11s %11s %13s %13s %9s\n", "Method", "CF%", "Conv%", "Cost", "Gap%",
                "Iter/rounds", "Alloc bc", "Total bc", "kB");
    for (size_t m = 0; m < names.size(); ++m) {
        auto c = ms(acc[m].cost), gp = ms(acc[m].gap), it = ms(acc[m].iters), al = ms(acc[m].alloc),
             tt = ms(acc[m].total), kb = ms(acc[m].kb);
        std::printf("%-36s %5.0f %5.0f %6.1f+/-%-5.1f %5.2f+/-%-4.2f %5.0f+/-%-5.0f %6.0f+/-%-5.0f %6.0f+/-%-6.0f %8.1f\n",
                    names[m].c_str(), 100.0 * acc[m].cf / acc[m].runs, 100.0 * acc[m].conv / acc[m].runs,
                    c.first, c.second, gp.first, gp.second, it.first, it.second, al.first, al.second,
                    tt.first, tt.second, kb.first);
    }
    std::printf("\nLargest relative W_k increase (EMD variants, max over runs):\n");
    for (size_t m = 4; m < names.size(); ++m)
        std::printf("  %-36s %.2e\n", names[m].c_str(), *std::max_element(acc[m].maxdw.begin(), acc[m].maxdw.end()));
    return 0;
}
