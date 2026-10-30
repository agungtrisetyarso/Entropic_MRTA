#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct EMDParams {
    double alpha = 0.08;
    double beta = 0.35;
    double lambda0 = 0.5;
    double lambda_inf = 0.02;
    int Kf = 200;
    int M = 15;          // Sinkhorn sweeps
    int T = 5;           // consensus rounds per sweep
    double delta = 0.05; // event trigger threshold
    double eps_conv = 1e-6;
    double purity_thresh = 0.99;
    int max_iter = 500;
};

struct RunResult {
    bool success = false;
    double final_cost = 0.0;
    int alloc_msgs = 0;
    int iterations = 0;
    double final_purity = 0.0;
    Matrix final_P;
};

class EMDAllocator {
public:
    EMDAllocator(const EMDParams& p = EMDParams()) : params_(p) {}

    RunResult run(const Matrix& C);

private:
    EMDParams params_;

    double get_lambda(int k) const;
    void mirror_step(Matrix& P, const Matrix& C, double alpha, double lambda, double beta);
    void sinkhorn_project(Matrix& P, int M, int T, int& msg_count); // counts hypothetical consensus msgs
    double compute_local_v(const Vector& row, const Vector& cost_row, double lambda) const;
    double team_entropy(const Matrix& P) const;
    double purity(const Matrix& P) const;
    double assignment_cost(const Matrix& P, const Matrix& C) const;
    Matrix round_to_permutation(const Matrix& P) const; // simple argmax per row with conflict resolution
};

// Utility
Matrix make_uniform(int n);
void normalize_row(Vector& row);
double log_safe(double x);
