#include "emd.h"
#include <random>
#include <cassert>

double log_safe(double x) {
    return std::log(std::max(x, 1e-12));
}

void normalize_row(Vector& row) {
    double s = std::accumulate(row.begin(), row.end(), 0.0);
    if (s > 0) {
        for (auto& v : row) v /= s;
    }
}

Matrix make_uniform(int n) {
    Matrix P(n, Vector(n, 1.0 / n));
    return P;
}

double EMDAllocator::get_lambda(int k) const {
    if (k >= params_.Kf) return params_.lambda_inf;
    return std::max(params_.lambda_inf, params_.lambda0 / std::log(k + 2.0));
}

double EMDAllocator::compute_local_v(const Vector& row, const Vector& cost_row, double lambda) const {
    double cost = 0.0;
    double ent = 0.0;
    for (size_t j = 0; j < row.size(); ++j) {
        cost += row[j] * cost_row[j];
        if (row[j] > 0) ent -= row[j] * log_safe(row[j]);
    }
    return cost + lambda * ent;
}

double EMDAllocator::team_entropy(const Matrix& P) const {
    double S = 0.0;
    for (const auto& row : P) {
        for (double p : row) {
            if (p > 0) S -= p * log_safe(p);
        }
    }
    return S;
}

double EMDAllocator::purity(const Matrix& P) const {
    double min_purity = 1.0;
    for (const auto& row : P) {
        double gamma = 0.0;
        for (double p : row) gamma += p * p;
        min_purity = std::min(min_purity, gamma);
    }
    return min_purity;
}

double EMDAllocator::assignment_cost(const Matrix& P, const Matrix& C) const {
    // Use rounded assignment for reported cost
    Matrix Pi = round_to_permutation(P);
    double cost = 0.0;
    for (size_t i = 0; i < Pi.size(); ++i)
        for (size_t j = 0; j < Pi[i].size(); ++j)
            cost += Pi[i][j] * C[i][j];
    return cost;
}

Matrix EMDAllocator::round_to_permutation(const Matrix& P) const {
    int n = P.size();
    Matrix Pi(n, Vector(n, 0.0));
    std::vector<bool> used(n, false);
    // Greedy by highest probability, resolve conflicts by order
    std::vector<std::pair<double, std::pair<int,int>>> cands;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            cands.emplace_back(P[i][j], std::make_pair(i, j));
    std::sort(cands.rbegin(), cands.rend()); // descending
    for (const auto& cand : cands) {
        int i = cand.second.first, j = cand.second.second;
        if (!used[j] && Pi[i][j] == 0.0) { // row not yet assigned? Wait, better check row sum
            bool row_assigned = false;
            for (int jj = 0; jj < n; ++jj) if (Pi[i][jj] > 0.5) row_assigned = true;
            if (!row_assigned) {
                Pi[i][j] = 1.0;
                used[j] = true;
            }
        }
    }
    // Fill remaining if any (rare)
    for (int i = 0; i < n; ++i) {
        bool assigned = false;
        for (int j = 0; j < n; ++j) if (Pi[i][j] > 0.5) assigned = true;
        if (!assigned) {
            for (int j = 0; j < n; ++j) if (!used[j]) {
                Pi[i][j] = 1.0;
                used[j] = true;
                break;
            }
        }
    }
    return Pi;
}

void EMDAllocator::mirror_step(Matrix& P, const Matrix& C, double alpha, double lambda, double beta) {
    int n = P.size();
    Matrix Pnew = P;
    for (int i = 0; i < n; ++i) {
        Vector tilde(n);
        for (int j = 0; j < n; ++j) {
            double p = std::max(P[i][j], 1e-12);
            tilde[j] = std::pow(p, 1.0 + alpha * lambda) * std::exp(-alpha * C[i][j]);
        }
        normalize_row(tilde);
        // Dual damping
        Vector hat(n);
        for (int j = 0; j < n; ++j) {
            double tp = std::max(tilde[j], 1e-12);
            double pp = std::max(P[i][j], 1e-12);
            hat[j] = std::pow(tp, 1.0 - beta) * std::pow(pp, beta);
        }
        normalize_row(hat);
        Pnew[i] = hat;
    }
    P = Pnew;
}

void EMDAllocator::sinkhorn_project(Matrix& P, int M, int T, int& msg_count) {
    int n = P.size();
    // Simulate distributed Sinkhorn: row scaling local, column by consensus
    // For correctness we do exact Sinkhorn; for message count we add hypothetical traffic
    // From paper: 2 * M * T * d * |E| approx, but for complete graph or degree, we count per robot broadcasts of d-vectors
    // Simple: each of M sweeps, T consensus rounds, each robot sends its current estimate to neighbors.
    // Approximate msg_count += M * T * n * (n-1); // all-to-all worst case, but paper uses graph.
    // For fair count of allocation layer only later; here we track projection messages separately if needed.
    // For this sim, we focus on allocation-layer msgs from event trigger; projection is auxiliary.

    for (int m = 0; m < M; ++m) {
        // Row scaling (local)
        for (int i = 0; i < n; ++i) {
            normalize_row(P[i]);
        }
        // Column scaling: compute column sums exactly (simulates perfect consensus)
        Vector col_sums(n, 0.0);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                col_sums[j] += P[i][j];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (col_sums[j] > 0) P[i][j] /= col_sums[j];
        // Hypothetical consensus messages: each of T rounds, each robot sends d-vector to neighbors
        // Assume average degree ~ log n or fixed; paper uses time-varying graph. We add n * T per sweep for rough count.
        msg_count += n * T; // rough: one "broadcast" equivalent per robot per round
    }
    // Final row normalize
    for (int i = 0; i < n; ++i) normalize_row(P[i]);
}

RunResult EMDAllocator::run(const Matrix& C) {
    int n = C.size();
    assert(n > 0 && C[0].size() == (size_t)n);
    Matrix P = make_uniform(n);
    RunResult res;
    std::vector<double> v_last(n, std::numeric_limits<double>::infinity());
    int alloc_msgs = 0;
    int proj_msgs = 0;

    for (int k = 0; k < params_.max_iter; ++k) {
        double lambda = get_lambda(k);
        double alpha = params_.alpha;

        // (1-2) Mirror step + damping
        mirror_step(P, C, alpha, lambda, params_.beta);

        // (3) Distributed Sinkhorn (exact in sim, count msgs)
        sinkhorn_project(P, params_.M, params_.T, proj_msgs);

        // (4) Local certificates
        std::vector<double> v_now(n);
        for (int i = 0; i < n; ++i) {
            v_now[i] = compute_local_v(P[i], C[i], lambda);
        }

        // (5) Event-triggered broadcast: if decrease > delta
        for (int i = 0; i < n; ++i) {
            if (v_last[i] - v_now[i] > params_.delta) {
                alloc_msgs++; // one allocation-layer broadcast of the row
                v_last[i] = v_now[i];
            }
        }

        // Check convergence
        double min_pur = purity(P);
        double W = 0.0;
        for (int i = 0; i < n; ++i) W += v_now[i];
        bool converged = (min_pur > params_.purity_thresh);
        // Also check small change, but purity is main for success in paper
        if (converged || k == params_.max_iter - 1) {
            res.iterations = k + 1;
            res.alloc_msgs = alloc_msgs;
            res.final_cost = assignment_cost(P, C);
            res.final_purity = min_pur;
            res.success = (min_pur > params_.purity_thresh && res.final_cost < 1e9);
            res.final_P = P;
            break;
        }
    }
    return res;
}
