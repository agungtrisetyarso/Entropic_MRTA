#include "emd.h"
#include <random>
#include <iomanip>
#include <sstream>

// Simple Hungarian algorithm for reference optimum (square, minimization)
// Implementation of Kuhn-Munkres (O(n^3))
double hungarian(const Matrix& cost) {
    int n = cost.size();
    const double INF = 1e9;
    Vector u(n+1, 0), v(n+1, 0), p(n+1, 0), way(n+1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        Vector minv(n+1, INF);
        std::vector<bool> used(n+1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = INF;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
                double cur = cost[i0-1][j-1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    double total = 0.0;
    for (int j = 1; j <= n; ++j) total += cost[p[j]-1][j-1];
    return total;
}

// Greedy nearest-task with conflict resolution by robot index
double greedy_cost(const Matrix& C) {
    int n = C.size();
    std::vector<bool> used(n, false);
    double cost = 0.0;
    const double INF = 1e9;
    for (int i = 0; i < n; ++i) {
        double best = INF;
        int bj = -1;
        for (int j = 0; j < n; ++j) if (!used[j] && C[i][j] < best) {
            best = C[i][j]; bj = j;
        }
        if (bj >= 0) { cost += best; used[bj] = true; }
    }
    return cost;
}

Matrix generate_cost_matrix(int n, std::mt19937& rng) {
    // Approximate TWPC-MRTA: heterogeneous costs, some high for capability mismatch, time-window pressure as additive
    std::uniform_real_distribution<double> base(5.0, 40.0);
    std::uniform_real_distribution<double> penalty(0.0, 30.0);
    Matrix C(n, Vector(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = base(rng);
            // Random capability mismatch ~20% chance of high cost
            if (std::uniform_real_distribution<double>(0,1)(rng) < 0.2)
                C[i][j] += 50.0 + penalty(rng);
            // Time window pressure: some tasks more urgent for some robots
            C[i][j] += penalty(rng) * 0.3;
        }
    }
    return C;
}

struct Stats {
    double success_rate = 0;
    double avg_cost = 0;
    double avg_msgs = 0;
    double avg_iters = 0;
    double avg_opt_gap = 0;
    int n_runs = 0;
};

void print_stats(const std::string& name, const Stats& s) {
    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::left << std::setw(28) << name
              << std::setw(12) << s.success_rate
              << std::setw(12) << s.avg_cost
              << std::setw(12) << s.avg_msgs
              << std::setw(10) << s.avg_iters
              << "\n";
}

int main() {
    const int N = 10;          // robots = tasks
    const int N_RUNS = 50;
    EMDParams params;          // paper defaults
    EMDAllocator emd(params);

    std::random_device rd;
    std::mt19937 rng(rd());

    Stats emd_stats, greedy_stats;
    double sum_opt = 0;

    std::cout << "Running " << N_RUNS << " Monte Carlo simulations for n=" << N << " ...\n";

    for (int r = 0; r < N_RUNS; ++r) {
        Matrix C = generate_cost_matrix(N, rng);
        double opt = hungarian(C);
        sum_opt += opt;

        // Proposed method
        RunResult res = emd.run(C);
        emd_stats.n_runs++;
        if (res.success) emd_stats.success_rate += 1.0;
        emd_stats.avg_cost += res.final_cost;
        emd_stats.avg_msgs += res.alloc_msgs;
        emd_stats.avg_iters += res.iterations;
        if (opt > 0) emd_stats.avg_opt_gap += (res.final_cost - opt) / opt * 100.0;

        // Greedy
        double gcost = greedy_cost(C);
        greedy_stats.n_runs++;
        // Greedy always "succeeds" as assignment, but may have higher cost; count as success if finite
        greedy_stats.success_rate += 1.0;
        greedy_stats.avg_cost += gcost;
        greedy_stats.avg_msgs += 0; // no messages in simple greedy
        greedy_stats.avg_iters += 1;
    }

    // Averages
    emd_stats.success_rate = 100.0 * emd_stats.success_rate / N_RUNS;
    emd_stats.avg_cost /= N_RUNS;
    emd_stats.avg_msgs /= N_RUNS;
    emd_stats.avg_iters /= N_RUNS;
    emd_stats.avg_opt_gap /= N_RUNS;

    greedy_stats.success_rate = 100.0 * greedy_stats.success_rate / N_RUNS;
    greedy_stats.avg_cost /= N_RUNS;
    greedy_stats.avg_msgs /= N_RUNS;
    greedy_stats.avg_iters /= N_RUNS;

    double avg_opt = sum_opt / N_RUNS;

    std::cout << "\n=== Table 4 style results (TWPC-MRTA approx, n=10, " << N_RUNS << " runs) ===\n";
    std::cout << std::left << std::setw(28) << "Method"
              << std::setw(12) << "Success %"
              << std::setw(12) << "Avg cost"
              << std::setw(12) << "Alloc msgs"
              << std::setw(10) << "Conv iter" << "\n";
    std::cout << std::string(74, '-') << "\n";
    print_stats("Greedy nearest-task", greedy_stats);
    print_stats("Proposed (Alg. 2)", emd_stats);
    std::cout << std::string(74, '-') << "\n";
    std::cout << "Centralized Hungarian optimum (reference avg cost): " << std::fixed << std::setprecision(1) << avg_opt << "\n";
    std::cout << "Proposed avg optimality gap: " << std::setprecision(1) << emd_stats.avg_opt_gap << "%\n";
    std::cout << "\nNote: Message counts are allocation-layer only (event-triggered row broadcasts).\n";
    std::cout << "Projection-consensus traffic is additional (see paper Table 5/8).\n";
    std::cout << "Exact paper numbers require the full Gazebo/ROS2 scenario; this reproduces the algorithmic behavior.\n";

    return 0;
}
