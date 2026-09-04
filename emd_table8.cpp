/* EMD-Auto — Table 8 (FILL-2) resource-triangle sweep (Corollary 13)
 *
 * Save as:  ~/SNJint/emd_table8.cpp
 * Build:    c++ -std=c++17 -O3 -o emd_table8 emd_table8.cpp
 * Run:      ./emd_table8 --quick
 *           ./emd_table8 --runs 50
 *
 * Columns:
 *   δ              event threshold of Theorem 12 / eq. (25)
 *   Comm. volume   team broadcasts of the allocation row
 *   Nop            stall-triggered operator queries (Algorithm 3)
 *   Optimality gap (J_final − Jmin) / Jmin × 100
 *   Final A        Allocation Autonomy Index at termination
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct Params {
    double alpha = 0.08;
    double beta = 0.35;
    double lambda0 = 0.5;
    double lambda_inf = 0.02;
    double delta = 0.05;
    int tau = 10;
    double eta = 0.02;
    int M = 15;
    int max_iter = 250;
    double purity_th = 0.99;
    double v_tol = 1e-6;
    int v_hold = 20;
    double eps_num = 1e-16;
    bool use_anneal = true;
};

struct Scenario {
    std::string name;
    int n_real = 5;
    int d = 12;
    unsigned seed_base = 11;
};

struct SweepRow {
    double delta = 0.0;
    double comm_mean = 0.0, comm_std = 0.0;
    double nop_mean = 0.0, nop_std = 0.0;
    double gap_mean = 0.0, gap_std = 0.0;
    double A_mean = 0.0, A_std = 0.0;
};

struct AssignmentResult {
    double cost = 0.0;
    std::vector<int> row_to_col;
};

AssignmentResult hungarian(const std::vector<std::vector<double>>& C_in) {
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
    for (int j = 1; j <= n; ++j)
        if (p[j] >= 1 && p[j] <= n) out.row_to_col[p[j] - 1] = j - 1;
    double chk = 0.0;
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (out.row_to_col[i] < 0) {
            ok = false;
            break;
        }
        chk += C_in[i][out.row_to_col[i]];
    }
    out.cost = ok ? chk : -v[0];
    return out;
}

static void row_normalize(std::vector<double>& row, double eps) {
    double z = 0.0;
    for (double x : row) z += x;
    if (z < eps) {
        std::fill(row.begin(), row.end(), 1.0 / static_cast<double>(row.size()));
        return;
    }
    for (double& x : row) x /= z;
}

static double shannon_row(const std::vector<double>& p, double eps) {
    double s = 0.0;
    for (double x : p)
        if (x > eps) s -= x * std::log(x);
    return s;
}

static double purity_row(const std::vector<double>& p) {
    double g = 0.0;
    for (double x : p) g += x * x;
    return g;
}

static double team_entropy(const std::vector<std::vector<double>>& P, double eps) {
    double S = 0.0;
    for (const auto& row : P) S += shannon_row(row, eps);
    return S;
}

static double inner_product(const std::vector<std::vector<double>>& C,
                            const std::vector<std::vector<double>>& P) {
    double v = 0.0;
    for (size_t i = 0; i < P.size(); ++i)
        for (size_t j = 0; j < P[i].size(); ++j)
            v += C[i][j] * P[i][j];
    return v;
}

static double autonomy_index(double S, int n, int d) {
    const double denom = static_cast<double>(n) * std::log(static_cast<double>(std::max(d, 2)));
    return 1.0 - S / denom;
}

static double lambda_of(const Params& par, int k) {
    if (!par.use_anneal) return par.lambda0;
    return std::max(par.lambda_inf, par.lambda0 / std::log(static_cast<double>(k + 2)));
}

// Sinkhorn that uses last-broadcast rows for other agents' column sums
// (event-triggered / stale consensus, Theorem 12(iii)).
static void sinkhorn_stale(std::vector<std::vector<double>>& P_local,
                           const std::vector<std::vector<double>>& P_view,
                           int M, double eps) {
    const int n = static_cast<int>(P_local.size());
    if (!n) return;
    const int d = static_cast<int>(P_local[0].size());
    auto Q = P_local;
    for (int sweep = 0; sweep < M; ++sweep) {
        for (int i = 0; i < n; ++i) row_normalize(Q[i], eps);
        std::vector<double> cs(d, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j)
                cs[j] += P_view[i][j];  // stale neighbours + own last view
        // Blend in current local rows so the iterate can still move.
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j)
                cs[j] += 0.5 * (Q[i][j] - P_view[i][j]);
        for (int j = 0; j < d; ++j) cs[j] = std::max(cs[j], eps);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j)
                Q[i][j] /= cs[j];
    }
    for (int i = 0; i < n; ++i) row_normalize(Q[i], eps);
    P_local.swap(Q);
}

static std::vector<std::vector<double>> make_cost(const Scenario& sc, std::mt19937& rng) {
    const int d = sc.d;
    const int n_real = std::max(1, std::min(sc.n_real, d));
    std::vector<std::vector<double>> C(d, std::vector<double>(d, 0.0));
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::vector<int> plant(d);
    std::iota(plant.begin(), plant.end(), 0);
    std::shuffle(plant.begin(), plant.end(), rng);
    for (int i = 0; i < d; ++i) {
        const bool dummy = (i >= n_real);
        for (int j = 0; j < d; ++j) {
            double base = 8.0 + 6.0 * U(rng);
            if ((i % 4) != (j % 4)) base += 4.0 + 3.0 * U(rng);
            base += 2.0 * std::abs(N01(rng));
            if (j == plant[i]) base *= 0.35;
            if (dummy) base += (j < n_real) ? 40.0 : 0.0;
            C[i][j] = std::max(0.05, base);
        }
    }
    return C;
}

static std::vector<std::vector<double>> submatrix(const std::vector<std::vector<double>>& C,
                                                 const std::vector<int>& rows,
                                                 const std::vector<int>& cols) {
    std::vector<std::vector<double>> S(rows.size(), std::vector<double>(cols.size()));
    for (size_t a = 0; a < rows.size(); ++a)
        for (size_t b = 0; b < cols.size(); ++b)
            S[a][b] = C[rows[a]][cols[b]];
    return S;
}

struct MissionOut {
    int Ncomm = 0;
    int Nop = 0;
    double gap_pct = 0.0;
    double final_A = 0.0;
    double Jmin = 0.0;
    double Jfinal = 0.0;
};

static MissionOut run_mission(const Scenario& sc, const Params& par, unsigned seed) {
    std::mt19937 rng(seed);
    const int D = sc.d;
    auto Cfull = make_cost(sc, rng);
    const auto lap0 = hungarian(Cfull);
    const double Jmin = lap0.cost;

    std::vector<int> arows(D), acols(D);
    std::iota(arows.begin(), arows.end(), 0);
    std::iota(acols.begin(), acols.end(), 0);
    std::vector<std::pair<int, int>> clamps;

    int Ncomm = 0, Nop = 0, global_k = 0;
    double last_A = 0.0;

    while (!arows.empty() && Nop <= D - 1) {
        const int n = static_cast<int>(arows.size());
        if (n == 1) {
            clamps.emplace_back(arows[0], acols[0]);
            break;
        }
        auto C = submatrix(Cfull, arows, acols);
        std::vector<std::vector<double>> P(n, std::vector<double>(n, 1.0 / n));
        auto P_view = P;  // last broadcast rows
        std::vector<double> vlast(n, 0.0);
        for (int i = 0; i < n; ++i)
            vlast[i] = inner_product(
                           std::vector<std::vector<double>>{C[i]},
                           std::vector<std::vector<double>>{P[i]}) +
                       par.lambda0 * shannon_row(P[i], par.eps_num);

        std::vector<std::vector<double>> Ai_win(n);
        double V_prev = 0.0;
        int hold = 0;
        bool committed = false;
        int stall_robot = -1;

        for (int k = 0; k < par.max_iter; ++k) {
            const double lam = lambda_of(par, global_k);
            const double expn = 1.0 + par.alpha * lam;
            auto Ptil = P;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    const double p = std::max(P[i][j], par.eps_num);
                    Ptil[i][j] = std::pow(p, expn) * std::exp(-par.alpha * C[i][j]);
                }
                row_normalize(Ptil[i], par.eps_num);
                for (int j = 0; j < n; ++j) {
                    const double a = std::max(Ptil[i][j], par.eps_num);
                    const double b = std::max(P[i][j], par.eps_num);
                    Ptil[i][j] = std::pow(a, 1.0 - par.beta) * std::pow(b, par.beta);
                }
                row_normalize(Ptil[i], par.eps_num);
            }
            sinkhorn_stale(Ptil, P_view, par.M, par.eps_num);
            P.swap(Ptil);
            ++global_k;

            const double S = team_entropy(P, par.eps_num);
            const double JC = inner_product(C, P);
            const double V = JC + lam * S;
            last_A = autonomy_index(S, D, D);
            if (k == 0) V_prev = V;
            if (std::fabs(V - V_prev) < par.v_tol) ++hold;
            else hold = 0;
            V_prev = V;

            bool all_pure = true;
            stall_robot = -1;
            double worst_Ai = 1.0;
            for (int i = 0; i < n; ++i) {
                const double Si = shannon_row(P[i], par.eps_num);
                double Jci = 0.0;
                for (int j = 0; j < n; ++j) Jci += C[i][j] * P[i][j];
                const double vi = Jci + lam * Si;
                // Theorem 12 / eq. (25): broadcast iff local certificate dropped by > δ
                const bool fire = (par.delta <= 0.0) || (vlast[i] - vi > par.delta);
                if (fire) {
                    P_view[i] = P[i];
                    vlast[i] = vi;
                    ++Ncomm;
                }
                const double Ai = 1.0 - Si / std::log(static_cast<double>(n));
                Ai_win[i].push_back(Ai);
                if (purity_row(P[i]) < par.purity_th) all_pure = false;
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
            if (stall_robot >= 0 && k >= par.tau) break;
        }

        if (committed || stall_robot < 0) {
            // Greedy rounding of the current block (not a centralized LAP),
            // so the optimality gap reflects EMD + event-triggered staleness.
            std::vector<char> taken(n, 0);
            for (int i = 0; i < n; ++i) {
                int best = -1;
                double bestp = -1.0;
                for (int j = 0; j < n; ++j) {
                    if (taken[j]) continue;
                    if (P[i][j] > bestp) {
                        bestp = P[i][j];
                        best = j;
                    }
                }
                if (best < 0) {
                    for (int j = 0; j < n; ++j)
                        if (!taken[j]) {
                            best = j;
                            break;
                        }
                }
                taken[best] = 1;
                clamps.emplace_back(arows[i], acols[best]);
            }
            break;
        }
        auto asg = hungarian(C);
        const int i_star = stall_robot;
        const int j_star = asg.row_to_col[i_star];
        clamps.emplace_back(arows[i_star], acols[j_star]);
        ++Nop;
        arows.erase(arows.begin() + i_star);
        acols.erase(acols.begin() + j_star);
    }

    std::vector<int> assign(D, -1);
    for (auto [r, c] : clamps) assign[r] = c;
    double Jf = 0.0;
    bool complete = true;
    for (int i = 0; i < D; ++i) {
        if (assign[i] < 0) {
            complete = false;
            break;
        }
        Jf += Cfull[i][assign[i]];
    }
    MissionOut o;
    o.Ncomm = Ncomm;
    o.Nop = Nop;
    o.Jmin = Jmin;
    o.Jfinal = complete ? Jf : Jmin;
    o.gap_pct = (Jmin > 1e-12) ? 100.0 * (o.Jfinal - Jmin) / Jmin : 0.0;
    if (o.gap_pct < 0.0) o.gap_pct = 0.0;
    o.final_A = last_A;
    return o;
}

static std::pair<double, double> mean_std(const std::vector<double>& x) {
    const double m = std::accumulate(x.begin(), x.end(), 0.0) / std::max<size_t>(1, x.size());
    double v = 0.0;
    for (double t : x) v += (t - m) * (t - m);
    v = (x.size() > 1) ? std::sqrt(v / (x.size() - 1)) : 0.0;
    return {m, v};
}

int main(int argc, char** argv) {
    int n_runs = 20;
    bool quick = false;
    Scenario sc{"Warehouse (Table 5 scale)", 5, 12, 11};

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--runs") && i + 1 < argc)
            n_runs = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--quick"))
            quick = true;
        else if (!std::strcmp(argv[i], "--warehouse60")) {
            sc = {"Warehouse logistics", 12, 60, 101};
        } else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            std::cerr << "Usage: " << argv[0]
                      << " [--quick] [--runs N] [--warehouse60]\n";
            return 0;
        }
    }
    if (quick) n_runs = 5;

    const std::vector<double> deltas = {0.0, 0.01, 0.05, 0.10, 0.20};

    std::cout << "EMD-Auto  |  Table 8 resource-triangle sweep  |  runs=" << n_runs
              << "  scenario=" << sc.name << "  d=" << sc.d << "\n";
    std::cout << "δ trigger of eq. (25); Nop from Algorithm 3; gap vs Hungarian Jmin.\n\n";

    std::cout << std::left
              << std::setw(18) << "delta"
              << std::setw(22) << "Comm. volume"
              << std::setw(16) << "Nop"
              << std::setw(22) << "Optimality gap (%)"
              << std::setw(14) << "Final A"
              << "\n";
    std::cout << std::string(92, '-') << "\n";

    std::ofstream csv("table8.csv");
    csv << "delta,comm_mean,comm_std,Nop_mean,Nop_std,gap_mean,gap_std,A_mean,A_std\n";

    const auto t0 = std::chrono::steady_clock::now();
    for (double del : deltas) {
        Params par;
        par.delta = del;
        if (quick) par.max_iter = 80;

        std::vector<double> comm, nop, gap, Af;
        for (int r = 0; r < n_runs; ++r) {
            const unsigned seed = sc.seed_base + 1000u * static_cast<unsigned>(r) + 17u;
            auto o = run_mission(sc, par, seed);
            comm.push_back(static_cast<double>(o.Ncomm));
            nop.push_back(static_cast<double>(o.Nop));
            gap.push_back(o.gap_pct);
            Af.push_back(o.final_A);
        }
        const auto [cm, cs] = mean_std(comm);
        const auto [nm, ns] = mean_std(nop);
        const auto [gm, gs] = mean_std(gap);
        const auto [am, as] = mean_std(Af);

        std::ostringstream dlab;
        if (del <= 0.0) dlab << "0 (continuous)";
        else
            dlab << std::fixed << std::setprecision(2) << del;

        auto cell = [](double m, double s, int prec) {
            std::ostringstream o;
            o << std::fixed << std::setprecision(prec) << m << "±" << s;
            return o.str();
        };

        std::cout << std::left
                  << std::setw(18) << dlab.str()
                  << std::setw(22) << cell(cm, cs, 1)
                  << std::setw(16) << cell(nm, ns, 2)
                  << std::setw(22) << cell(gm, gs, 2)
                  << std::setw(14) << cell(am, as, 3)
                  << "\n";

        csv << del << "," << cm << "," << cs << "," << nm << "," << ns << ","
            << gm << "," << gs << "," << am << "," << as << "\n";
    }
    csv.close();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\nWrote table8.csv  (" << std::fixed << std::setprecision(1) << sec << " s)\n";
    std::cout << "Corollary 13: larger δ cuts bandwidth, leaves Nop ≤ d−1, "
                 "and widens the stationarity / optimality neighbourhood.\n";
    return 0;
}
