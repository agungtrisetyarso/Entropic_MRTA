/* standalone EMD-Auto Table 6 — single file, no CMake required
 * Save as: ~/SNJint/emd_table6.cpp
 * Build:   c++ -std=c++17 -O3 -o emd_table6 emd_table6.cpp
 * Run:     ./emd_table6 --quick
 */

#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

// Square assignment: Hungarian (Kuhn–Munkres) minimizing sum C[i][perm[i]].
// Returns {min_cost, assignment} where assignment[i] = column assigned to row i.
struct AssignmentResult {
    double cost = 0.0;
    std::vector<int> row_to_col;
};

inline AssignmentResult hungarian(const std::vector<std::vector<double>>& C_in) {
    const int n = static_cast<int>(C_in.size());
    AssignmentResult out;
    out.row_to_col.assign(n, -1);
    if (n == 0) return out;

    const double INF = 1e100;
    std::vector<std::vector<double>> a(n + 1, std::vector<double>(n + 1, 0.0));
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
            a[i][j] = C_in[i - 1][j - 1];

    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<char> used(n + 1, 0);
        do {
            used[j0] = 1;
            int i0 = p[j0], j1 = 0;
            double delta = INF;
            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                double cur = a[i0][j] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    out.row_to_col.assign(n, -1);
    for (int j = 1; j <= n; ++j) {
        if (p[j] >= 1 && p[j] <= n)
            out.row_to_col[p[j] - 1] = j - 1;
    }
    out.cost = -v[0];
    // Recompute cost from original matrix for numerical safety.
    double chk = 0.0;
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (out.row_to_col[i] < 0) { ok = false; break; }
        chk += C_in[i][out.row_to_col[i]];
    }
    if (ok) out.cost = chk;
    return out;
}

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>
#include <iomanip>
#include <sstream>

// Entropic Mirror Descent with Autonomy reporting (EMD-Auto)
// Implements Algorithms 1–3 and the Table-6 protocol of:
// Trisetyarso, Yulianti, Surendro — "Entropic Certificates for Autonomous
// Multi-Robot Task Allocation".
//
// Square allocation after padding: n == d.

struct Params {
    double alpha     = 0.08;   // mirror step size
    double beta      = 0.35;   // dual-domain damping
    double lambda0   = 0.5;    // initial regularizer
    double lambda_inf= 0.02;   // frozen floor of the schedule
    double delta     = 0.05;   // event-trigger (not required for Table 6)
    int    tau       = 10;     // stall window
    double eta       = 0.02;   // stall tolerance
    int    M         = 15;     // Sinkhorn sweeps
    int    T         = 5;      // consensus rounds (centralized Sinkhorn uses M only)
    int    max_iter  = 400;    // safety cap per reduced problem
    double purity_th = 0.99;   // declare committed when Tr ρ_i^2 > this
    double v_tol     = 1e-6;   // |ΔV| stall of the certificate
    int    v_hold    = 20;     // consecutive tiny-ΔV iterations
    double eps_num   = 1e-16;  // floor for logs / divisions
    bool   use_anneal= true;
    bool   verbose   = false;
};

struct Scenario {
    std::string name;
    int n_real = 0;     // physical robots (before dummy padding)
    int d      = 0;     // tasks = Birkhoff dimension after padding
    unsigned seed_base = 1;
};

struct RunRecord {
    double A_design = 0.0;     // Theorem 10 floor (may be negative)
    double min_A    = 1.0;     // min_k A(k) over the mission
    double final_A  = 0.0;
    int    Nop      = 0;
    int    bound    = 0;       // d-1
    int    iterations = 0;
    double Jmin     = 0.0;
    double V0       = 0.0;
    double cost_final = 0.0;
};

inline double safe_log(double x, double eps) {
    return std::log(std::max(x, eps));
}

inline double shannon_row(const std::vector<double>& p, double eps) {
    double s = 0.0;
    for (double x : p) {
        if (x > eps) s -= x * std::log(x);
    }
    return s;
}

inline double purity_row(const std::vector<double>& p) {
    double g = 0.0;
    for (double x : p) g += x * x;
    return g;
}

inline double lambda_of(const Params& par, int k) {
    if (!par.use_anneal) return par.lambda0;
    return std::max(par.lambda_inf, par.lambda0 / std::log(static_cast<double>(k + 2)));
}

// Row-normalize in place.
inline void row_normalize(std::vector<double>& row, double eps) {
    double z = 0.0;
    for (double x : row) z += x;
    if (z < eps) {
        const double u = 1.0 / static_cast<double>(row.size());
        std::fill(row.begin(), row.end(), u);
        return;
    }
    for (double& x : row) x /= z;
}

// Centralized Sinkhorn onto the Birkhoff polytope (exact column sums).
// Equivalent to Algorithm 1 with perfect consensus (ε → 0).
inline void sinkhorn(std::vector<std::vector<double>>& P, int M, double eps) {
    const int n = static_cast<int>(P.size());
    if (n == 0) return;
    const int d = static_cast<int>(P[0].size());
    for (int sweep = 0; sweep < M; ++sweep) {
        for (int i = 0; i < n; ++i) {
            double rs = 0.0;
            for (int j = 0; j < d; ++j) rs += P[i][j];
            rs = std::max(rs, eps);
            for (int j = 0; j < d; ++j) P[i][j] /= rs;
        }
        std::vector<double> cs(d, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j) cs[j] += P[i][j];
        for (int j = 0; j < d; ++j) cs[j] = std::max(cs[j], eps);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j) P[i][j] /= cs[j];
    }
    // Final row normalize so each robot remains a density operator.
    for (int i = 0; i < n; ++i) row_normalize(P[i], eps);
}

inline double team_entropy(const std::vector<std::vector<double>>& P, double eps) {
    double S = 0.0;
    for (const auto& row : P) S += shannon_row(row, eps);
    return S;
}

inline double inner_product(const std::vector<std::vector<double>>& C,
                            const std::vector<std::vector<double>>& P) {
    double v = 0.0;
    const int n = static_cast<int>(P.size());
    const int d = n ? static_cast<int>(P[0].size()) : 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            v += C[i][j] * P[i][j];
    return v;
}

inline double autonomy_index(double S, int n, int d) {
    const double denom = static_cast<double>(n) * std::log(static_cast<double>(d));
    if (denom <= 0.0) return 1.0;
    return 1.0 - S / denom;
}

// Generate a scenario cost matrix. Heterogeneity, capability exclusions and
// a planted low-cost matching make Jmin well-defined and the landscape nontrivial.
std::vector<std::vector<double>> make_cost(const Scenario& sc, std::mt19937& rng);

// One Monte-Carlo mission with stall-triggered mixed-initiative (Algorithm 3).
RunRecord run_mission(const Scenario& sc, const Params& par, unsigned seed);


std::vector<std::vector<double>> make_cost(const Scenario& sc, std::mt19937& rng) {
    const int d = sc.d;
    const int n_real = std::max(1, std::min(sc.n_real, d));
    std::vector<std::vector<double>> C(d, std::vector<double>(d, 0.0));

    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);

    // Planted permutation: cheap "natural" assignment so Jmin is identifiable.
    std::vector<int> plant(d);
    std::iota(plant.begin(), plant.end(), 0);
    std::shuffle(plant.begin(), plant.end(), rng);

    for (int i = 0; i < d; ++i) {
        const bool dummy_robot = (i >= n_real);
        for (int j = 0; j < d; ++j) {
            double base = 8.0 + 6.0 * U(rng);          // travel / energy
            // Heterogeneous capability mismatch
            const int type_r = i % 4;
            const int type_t = j % 4;
            if (type_r != type_t) base += 4.0 + 3.0 * U(rng);
            // Time-window / deadline pressure
            base += 2.0 * std::abs(N01(rng));
            // Planted matching discount
            if (j == plant[i]) base *= 0.35;
            // Dummy robots: only cheap on "slack" (last) tasks, expensive otherwise
            if (dummy_robot) {
                if (j < n_real) base += 40.0;
                else            base *= 0.5;
            }
            // Dummy tasks for extra robots (none if n_real <= d)
            C[i][j] = std::max(0.05, base);
        }
    }

    // Scenario-specific structure
    if (sc.name.find("UAV") != std::string::npos) {
        // Sparse high-cost "no-fly" pairs
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                if ((i + 3 * j) % 7 == 0 && j != plant[i]) C[i][j] += 18.0;
    } else if (sc.name.find("Heterogeneous") != std::string::npos) {
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                if ((i % 3) == 0 && (j % 5) == 0 && j != plant[i]) C[i][j] += 25.0;
    } else if (sc.name.find("Dynamic") != std::string::npos) {
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                C[i][j] += 1.5 * ((i * 17 + j * 13) % 11);
    }
    return C;
}

// Restrict C to active rows/cols.
static std::vector<std::vector<double>> submatrix(
    const std::vector<std::vector<double>>& C,
    const std::vector<int>& rows,
    const std::vector<int>& cols)
{
    std::vector<std::vector<double>> S(rows.size(), std::vector<double>(cols.size()));
    for (size_t a = 0; a < rows.size(); ++a)
        for (size_t b = 0; b < cols.size(); ++b)
            S[a][b] = C[rows[a]][cols[b]];
    return S;
}

static void init_uniform(std::vector<std::vector<double>>& P) {
    const int n = static_cast<int>(P.size());
    if (!n) return;
    const int d = static_cast<int>(P[0].size());
    const double u = 1.0 / static_cast<double>(d);
    for (int i = 0; i < n; ++i)
        std::fill(P[i].begin(), P[i].end(), u);
}

static void emd_step(std::vector<std::vector<double>>& P,
                     const std::vector<std::vector<double>>& C,
                     const Params& par, double lam)
{
    const int n = static_cast<int>(P.size());
    const int d = n ? static_cast<int>(P[0].size()) : 0;
    const double expn = 1.0 + par.alpha * lam;
    std::vector<std::vector<double>> Ptil(n, std::vector<double>(d, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            const double p = std::max(P[i][j], par.eps_num);
            Ptil[i][j] = std::pow(p, expn) * std::exp(-par.alpha * C[i][j]);
        }
        row_normalize(Ptil[i], par.eps_num);
        // Dual-domain damping (12): geometric mean, then renormalize
        for (int j = 0; j < d; ++j) {
            const double a = std::max(Ptil[i][j], par.eps_num);
            const double b = std::max(P[i][j], par.eps_num);
            Ptil[i][j] = std::pow(a, 1.0 - par.beta) * std::pow(b, par.beta);
        }
        row_normalize(Ptil[i], par.eps_num);
    }
    sinkhorn(Ptil, par.M, par.eps_num);
    P.swap(Ptil);
}

RunRecord run_mission(const Scenario& sc, const Params& par, unsigned seed) {
    std::mt19937 rng(seed);
    const int D = sc.d;
    auto Cfull = make_cost(sc, rng);

    // Design-time quantities on the full problem, uniform start, λ = λ0.
    std::vector<std::vector<double>> P0(D, std::vector<double>(D, 1.0 / D));
    const double S0 = team_entropy(P0, par.eps_num);
    const double JC0 = inner_product(Cfull, P0);
    const double V0  = JC0 + par.lambda0 * S0;
    const auto lap   = hungarian(Cfull);
    const double Jmin = lap.cost;
    const double denom = par.lambda0 * static_cast<double>(D) * std::log(static_cast<double>(D));
    const double A_design = (denom > 0.0) ? 1.0 - (V0 - Jmin) / denom : 0.0;

    RunRecord rec;
    rec.A_design = A_design;
    rec.Jmin = Jmin;
    rec.V0 = V0;
    rec.bound = D - 1;
    rec.min_A = 1.0; // updated after every EMD step (post-update A(k))

    // Active sets (Algorithm 3 clamping)
    std::vector<int> arows(D), acols(D);
    std::iota(arows.begin(), arows.end(), 0);
    std::iota(acols.begin(), acols.end(), 0);
    std::vector<std::pair<int,int>> clamps; // (row, col) in original indices

    int global_k = 0;
    int Nop = 0;

    while (!arows.empty() && Nop <= D - 1) {
        const int n = static_cast<int>(arows.size());
        if (n == 1) {
            // Last pair is forced — no operator query (Theorem 11(i)).
            clamps.emplace_back(arows[0], acols[0]);
            break;
        }

        auto C = submatrix(Cfull, arows, acols);
        std::vector<std::vector<double>> P(n, std::vector<double>(n, 1.0 / n));
        init_uniform(P);

        std::vector<double> Ai_hist; // team-mean A per iteration of this block
        std::vector<std::vector<double>> Ai_win(n); // per-robot A history
        double V_prev = 0.0;
        int hold = 0;
        bool committed = false;
        int stall_robot = -1;

        for (int k = 0; k < par.max_iter; ++k) {
            const double lam = lambda_of(par, global_k);
            emd_step(P, C, par, lam);
            ++global_k;

            const double S_active = team_entropy(P, par.eps_num);
            const double JC = inner_product(C, P);
            const double V  = JC + lam * S_active;
            // Full-team index: clamped robots are pure (S=0), denominator is n=d
            // of the original problem so A is comparable to Theorem 10.
            const double A  = autonomy_index(S_active, D, D);
            rec.min_A = std::min(rec.min_A, A);
            Ai_hist.push_back(A);

            if (k == 0) V_prev = V;
            if (std::abs(V - V_prev) < par.v_tol) ++hold;
            else hold = 0;
            V_prev = V;

            bool all_pure = true;
            stall_robot = -1;
            double worst_Ai = 1.0;
            for (int i = 0; i < n; ++i) {
                const double Si = shannon_row(P[i], par.eps_num);
                const double Ai = 1.0 - Si / std::log(static_cast<double>(n));
                Ai_win[i].push_back(Ai);
                const double g = purity_row(P[i]);
                if (g < par.purity_th) all_pure = false;

                if (static_cast<int>(Ai_win[i].size()) > par.tau) {
                    const double Aold = Ai_win[i][Ai_win[i].size() - 1 - par.tau];
                    const bool stalled = (Ai - Aold < par.eta) && (Ai < 1.0 - par.eta);
                    if (stalled && Ai < worst_Ai) {
                        worst_Ai = Ai;
                        stall_robot = i;
                    }
                }
            }

            if (all_pure && hold >= par.v_hold) {
                committed = true;
                break;
            }
            if (stall_robot >= 0 && k >= par.tau) {
                break; // hand back to operator
            }
        }

        if (committed || stall_robot < 0) {
            // Round remaining block by Hungarian on current C (anytime rounding).
            auto asg = hungarian(C);
            for (int i = 0; i < n; ++i)
                clamps.emplace_back(arows[i], acols[asg.row_to_col[i]]);
            break;
        }

        // Algorithm 3: query the most stalled robot. Oracle = LAP-optimal
        // remaining task for that robot (paper §7.7).
        const int i_star = stall_robot;
        auto asg = hungarian(C);
        int j_star = asg.row_to_col[i_star];

        // Present top-2 by current P (logged only).
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return P[i_star][a] > P[i_star][b]; });

        if (par.verbose) {
            std::cout << "  QUERY robot " << arows[i_star]
                      << " top2=[" << acols[order[0]] << ","
                      << (n > 1 ? acols[order[1]] : -1)
                      << "] oracle=" << acols[j_star] << "\n";
        }

        clamps.emplace_back(arows[i_star], acols[j_star]);
        ++Nop;

        arows.erase(arows.begin() + i_star);
        acols.erase(acols.begin() + j_star);
    }

    rec.Nop = Nop;
    rec.iterations = global_k;

    // Reconstruct assignment cost
    std::vector<int> assign(D, -1);
    for (auto [r, c] : clamps) assign[r] = c;
    double cf = 0.0;
    bool complete = true;
    for (int i = 0; i < D; ++i) {
        if (assign[i] < 0) { complete = false; break; }
        cf += Cfull[i][assign[i]];
    }
    rec.cost_final = complete ? cf : std::numeric_limits<double>::quiet_NaN();

    // Final A is 1 for a permutation (S = 0).
    rec.final_A = complete ? 1.0 : rec.min_A;
    return rec;
}


#include <fstream>
#include <chrono>
#include <cstring>

static void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " [--runs N] [--max-iter K] [--verbose] [--quick]\n"
        << "  --runs N      Monte-Carlo runs per scenario (default 20; paper uses 50)\n"
        << "  --max-iter K  Safety cap per reduced block (default 400)\n"
        << "  --quick       Fast demo: 5 runs, smaller iteration cap\n"
        << "  --verbose     Print each operator query\n";
}

int main(int argc, char** argv) {
    int n_runs = 20;
    Params par;
    bool quick = false;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) {
            n_runs = std::max(1, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--max-iter") && i + 1 < argc) {
            par.max_iter = std::max(10, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--verbose")) {
            par.verbose = true;
        } else if (!std::strcmp(argv[i], "--quick")) {
            quick = true;
        } else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (quick) {
        n_runs = 5;
        par.max_iter = 120;
    }

    // Table 6 scenarios. d is the (padded) Birkhoff dimension.
    // n_real is the physical team size from §7.9.
    const std::vector<Scenario> scenarios = {
        {"Warehouse logistics", 12, 60, 101},
        {"UAV search swarm",     8, 24, 202},
        {"Heterogeneous team",  10, 32, 303},
        {"Dynamic online",      12, 48, 404},
    };

    std::cout << "EMD-Auto  |  Table 6 protocol  |  runs=" << n_runs
              << "  max_iter=" << par.max_iter << "\n";
    std::cout << "Params: α=" << par.alpha << "  β=" << par.beta
              << "  λ0=" << par.lambda0 << "  λ∞=" << par.lambda_inf
              << "  τ=" << par.tau << "  η=" << par.eta
              << "  M=" << par.M << "\n\n";

    std::cout << std::left
              << std::setw(24) << "Scenario"
              << std::setw(6)  << "d"
              << std::setw(18) << "A (design)"
              << std::setw(18) << "min A (meas)"
              << std::setw(16) << "Nop (meas)"
              << std::setw(12) << "bound d-1"
              << std::setw(12) << "Jmin"
              << "\n";
    std::cout << std::string(108, '-') << "\n";

    std::ofstream csv("table6.csv");
    csv << "scenario,d,A_design_mean,A_design_std,minA_mean,minA_std,"
           "Nop_mean,Nop_std,bound,Jmin_mean,iters_mean\n";

    const auto t0 = std::chrono::steady_clock::now();

    for (const auto& sc : scenarios) {
        std::vector<RunRecord> recs;
        recs.reserve(n_runs);
        for (int r = 0; r < n_runs; ++r) {
            const unsigned seed = sc.seed_base + 1000u * static_cast<unsigned>(r) + 7u;
            recs.push_back(run_mission(sc, par, seed));
            if ((r + 1) % 5 == 0 || r + 1 == n_runs) {
                std::cerr << "  " << sc.name << "  run " << (r + 1) << "/" << n_runs
                          << "  Nop=" << recs.back().Nop
                          << "  minA=" << std::fixed << std::setprecision(3)
                          << recs.back().min_A << "\n";
            }
        }

        auto mean_std = [](const std::vector<double>& x) {
            const double m = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
            double v = 0.0;
            for (double t : x) v += (t - m) * (t - m);
            v = (x.size() > 1) ? std::sqrt(v / (x.size() - 1)) : 0.0;
            return std::pair<double,double>{m, v};
        };

        std::vector<double> Ad, Am, Np, Jm, It;
        for (const auto& rec : recs) {
            Ad.push_back(rec.A_design);
            Am.push_back(rec.min_A);
            Np.push_back(static_cast<double>(rec.Nop));
            Jm.push_back(rec.Jmin);
            It.push_back(static_cast<double>(rec.iterations));
        }
        const auto [Adm, Ads] = mean_std(Ad);
        const auto [Amm, Ams] = mean_std(Am);
        const auto [Npm, Nps] = mean_std(Np);
        const auto [Jmm, Jms] = mean_std(Jm);
        const auto [Itm, Its] = mean_std(It);
        const int bound = sc.d - 1;

        std::ostringstream Ad_s, Am_s, Np_s;
        Ad_s << std::fixed << std::setprecision(3) << Adm << "±" << std::setprecision(3) << Ads;
        Am_s << std::fixed << std::setprecision(3) << Amm << "±" << std::setprecision(3) << Ams;
        Np_s << std::fixed << std::setprecision(2) << Npm << "±" << std::setprecision(2) << Nps;

        std::cout << std::left
                  << std::setw(24) << sc.name
                  << std::setw(6)  << sc.d
                  << std::setw(18) << Ad_s.str()
                  << std::setw(18) << Am_s.str()
                  << std::setw(16) << Np_s.str()
                  << std::setw(12) << bound
                  << std::setw(12) << std::fixed << std::setprecision(1) << Jmm
                  << "\n";

        csv << sc.name << "," << sc.d << ","
            << Adm << "," << Ads << ","
            << Amm << "," << Ams << ","
            << Npm << "," << Nps << ","
            << bound << ","
            << Jmm << "," << Itm << "\n";
    }

    csv.close();
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nWrote table6.csv  (" << std::fixed << std::setprecision(1)
              << sec << " s)\n";
    std::cout
        << "\nNotes\n"
        << "  A (design) is Theorem 10 evaluated at λ=λ0 from the uniform start.\n"
        << "  A floor may be negative when λ0 does not dominate the cost spread;\n"
        << "  that is the honest worst-case envelope (paper §6.1).\n"
        << "  Nop ≤ d−1 is Theorem 11(i); the measured mean should stay below the bound.\n"
        << "  Operator answers are an LAP oracle on the active subproblem (§7.7).\n";
    return 0;
}
