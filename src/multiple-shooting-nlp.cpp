#include "coin-or/IpIpoptApplication.hpp"

#include "multiple-shooting-nlp.hpp"

#include <cassert>
#include <cmath>

#include <iostream>

using namespace Ipopt;

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

MultipleShootingNlp::MultipleShootingNlp(const boost::shared_ptr<crocoddyl::ShootingProblem> &problem,
                                         bool                                                 printiterate)

    : problem_(problem),
      nx_(problem_->get_nx()),
      nu_(problem_->get_nu_max()),
      T_(problem_->get_T()),
      nvar_(T_ * (nx_ + nu_) + nx_),  // Multiple shooting, states and controls
      nconst_((T_ + 1) * nx_),        // T*nx eq. constraints for dynamics , nx eq constraints for initial conditions
      printiterate_(printiterate)
{
}

MultipleShootingNlp::~MultipleShootingNlp() {}

bool MultipleShootingNlp::get_nlp_info(Index          &n,
                                       Index          &m,
                                       Index          &nnz_jac_g,
                                       Index          &nnz_h_lag,
                                       IndexStyleEnum &index_style)
{
    n = nvar_;

    m = nconst_;

    // Jacobian nonzeros for dynamic constraints
    nnz_jac_g = T_ * nx_ * (2 * nx_ + nu_);

    // Jacobian nonzeros for initial conditiona
    nnz_jac_g += nx_;

    // Hessian is only affected by costs
    // Running Costs
    std::size_t nonzero = 0;
    for (size_t i = 1; i <= (nx_ + nu_); i++) {
        nonzero += i;
    }
    nnz_h_lag = T_ * nonzero;

    // Terminal Costs
    nonzero = 0;
    for (size_t i = 1; i <= nx_; i++) {
        nonzero += i;
    }
    nnz_h_lag += nonzero;

    // use the C style indexing (0-based)
    index_style = TNLP::C_STYLE;

    return true;
}
// [TNLP_get_nlp_info]

// [TNLP_get_bounds_info]
// returns the variable bounds
bool MultipleShootingNlp::get_bounds_info(Index n, Number *x_l, Number *x_u, Index m, Number *g_l, Number *g_u)
{
    // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
    // If desired, we could assert to make sure they are what we think they are.
    assert(n == nvar_);
    assert(m == nconst_);

    // the variables have lower bounds of 1
    for (Index i = 0; i < n; i++) {
        x_l[i] = -2e19;
        x_u[i] = 2e19;
    }

    // Dynamics
    for (Index i = 0; i < nconst_ - nx_; i++) {
        g_l[i] = 0;
        g_u[i] = 0;
    }

    for (Index i = 0; i < nx_; i++) {
        g_l[nconst_ - nx_ + i] = problem_->get_x0()[i];
        g_u[nconst_ - nx_ + i] = problem_->get_x0()[i];
    }

    return true;
}
// [TNLP_get_bounds_info]

// [TNLP_get_starting_point]
// returns the initial point for the problem
bool MultipleShootingNlp::get_starting_point(Index   n,
                                             bool    init_x,
                                             Number *x,
                                             bool    init_z,
                                             Number *z_L,
                                             Number *z_U,
                                             Index   m,
                                             bool    init_lambda,
                                             Number *lambda)
{
    // Here, we assume we only have starting values for x, if you code
    // your own NLP, you can provide starting values for the dual variables
    // if you wish
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);

    // initialize to the given starting point
    for (size_t i = 0; i < nvar_; i++) {
        x[i] = 0;
    }

    return true;
}
// [TNLP_get_starting_point]

// [TNLP_eval_f]
// returns the value of the objective function
bool MultipleShootingNlp::eval_f(Index n, const Number *x, bool new_x, Number &obj_value)
{
    assert(n == nvar_);

    for (size_t i = 0; i < T_; i++) {
        Eigen::VectorXd state   = Eigen::VectorXd::Map(x + i * (nx_ + nu_), nx_);
        Eigen::VectorXd control = Eigen::VectorXd::Map(x + i * (nx_ + nu_) + nx_, nu_);

        problem_->get_runningModels()[i]->calc(problem_->get_runningDatas()[i], state, control);

        obj_value += problem_->get_runningDatas()[i]->cost;
    }

    Eigen::VectorXd state = Eigen::VectorXd::Map(x + T_ * (nx_ + nu_), nx_);
    problem_->get_terminalModel()->calc(problem_->get_terminalData(), state);
    obj_value += problem_->get_terminalData()->cost;

    return true;
}
// [TNLP_eval_f]

// [TNLP_eval_grad_f]
// return the gradient of the objective function grad_{x} f(x)
bool MultipleShootingNlp::eval_grad_f(Index n, const Number *x, bool new_x, Number *grad_f)
{
    assert(n == nvar_);

    for (size_t i = 0; i < T_; i++) {
        Eigen::VectorXd state   = Eigen::VectorXd::Map(x + i * (nx_ + nu_), nx_);
        Eigen::VectorXd control = Eigen::VectorXd::Map(x + i * (nx_ + nu_) + nx_, nu_);

        // problem_->get_runningModels()[i]->calc(problem_->get_runningDatas()[i], state, control);
        problem_->get_runningModels()[i]->calcDiff(problem_->get_runningDatas()[i], state, control);

        for (size_t j = 0; j < nx_; j++) {
            grad_f[i * (nx_ + nu_) + j] = problem_->get_runningDatas()[i]->Lx[j];
        }

        for (size_t j = 0; j < nu_; j++) {
            grad_f[i * (nx_ + nu_) + nx_ + j] = problem_->get_runningDatas()[i]->Lu[j];
        }
    }

    Eigen::VectorXd state = Eigen::VectorXd::Map(x + T_ * (nx_ + nu_), nx_);
    problem_->get_terminalModel()->calcDiff(problem_->get_terminalData(), state);

    for (size_t j = 0; j < nx_; j++) {
        grad_f[T_ * (nx_ + nu_) + j] = problem_->get_terminalData()->Lx[j];
    }

    return true;
}
// [TNLP_eval_grad_f]

// [TNLP_eval_g]
// return the value of the constraints: g(x)
bool MultipleShootingNlp::eval_g(Index n, const Number *x, bool new_x, Index m, Number *g)
{
    assert(n == nvar_);
    assert(m == nconst_);

    for (size_t i = 0; i < T_; i++) {
        Eigen::VectorXd state      = Eigen::VectorXd::Map(x + i * (nx_ + nu_), nx_);
        Eigen::VectorXd state_next = Eigen::VectorXd::Map(x + (i + 1) * (nx_ + nu_), nx_);
        Eigen::VectorXd control    = Eigen::VectorXd::Map(x + i * (nx_ + nu_) + nx_, nu_);

        problem_->get_runningModels()[i]->calc(problem_->get_runningDatas()[i], state, control);
        // problem_->get_runningModels()[i]->calcDiff(problem_->get_runningDatas()[i], state, control);

        Eigen::VectorXd state_diff = state_next - problem_->get_runningDatas()[i]->xnext;

        for (size_t j = 0; j < nx_; j++) {
            g[i * nx_ + j] = state_diff[j];
        }
    }

    for (size_t j = 0; j < nx_; j++) {
        g[T_ * nx_ + j] = x[j];
    }

    return true;
}
// [TNLP_eval_g]

// [TNLP_eval_jac_g]
// return the structure or values of the Jacobian
bool MultipleShootingNlp::eval_jac_g(Index         n,
                                     const Number *x,
                                     bool          new_x,
                                     Index         m,
                                     Index         nele_jac,
                                     Index        *iRow,
                                     Index        *jCol,
                                     Number       *values)
{
    assert(n == nvar_);
    assert(m == nconst_);

    if (values == NULL) {
        std::size_t idx_value = 0;
        // Dynamic constraints
        for (size_t idx_block = 0; idx_block < T_; idx_block++) {
            for (size_t idx_row = 0; idx_row < nx_; idx_row++) {
                for (size_t idx_col = 0; idx_col < (2 * nx_ + nu_); idx_col++) {
                    iRow[idx_value] = idx_block * nx_ + idx_row;
                    jCol[idx_value] = idx_block * (nx_ + nu_) + idx_col;
                    // x_k+1 could be more optimized since it is a diagonal
                    idx_value++;
                }
            }
        }

        // Initial condition
        for (size_t idx_row = 0; idx_row < nx_; idx_row++) {
            for (size_t idx_col = 0; idx_col < nx_; idx_col++) {
                if (idx_row == idx_col) {
                    iRow[idx_value] = T_ * nx_ + idx_row;
                    jCol[idx_value] = idx_col;
                    idx_value++;
                    break;
                }
            }
        }

        assert(nele_jac == idx_value);

    } else {
        std::size_t idx_value = 0;

        // Dynamic constraints
        for (size_t idx_block = 0; idx_block < T_; idx_block++) {
            Eigen::VectorXd state      = Eigen::VectorXd::Map(x + idx_block * (nx_ + nu_), nx_);
            Eigen::VectorXd state_next = Eigen::VectorXd::Map(x + (idx_block + 1) * (nx_ + nu_), nx_);
            Eigen::VectorXd control    = Eigen::VectorXd::Map(x + idx_block * (nx_ + nu_) + nx_, nu_);

            problem_->get_runningModels()[idx_block]->calc(problem_->get_runningDatas()[idx_block], state, control);
            problem_->get_runningModels()[idx_block]->calcDiff(problem_->get_runningDatas()[idx_block], state,
                                                               control);

            for (size_t idx_row = 0; idx_row < nx_; idx_row++) {
                for (size_t idx_col = 0; idx_col < nx_; idx_col++) {
                    values[idx_value] = -problem_->get_runningDatas()[idx_block]->Fx(idx_row, idx_col);
                    idx_value++;
                }

                for (size_t idx_col = 0; idx_col < nu_; idx_col++) {
                    values[idx_value] = -problem_->get_runningDatas()[idx_block]->Fu(idx_row, idx_col);
                    idx_value++;
                }

                // This could be more optimized since there are a lot of zeros!
                for (size_t idx_col = 0; idx_col < nx_; idx_col++) {
                    values[idx_value] = idx_col == idx_row ? 1 : 0;
                    idx_value++;
                }
            }
        }

        for (size_t idx_row = 0; idx_row < nx_; idx_row++) {
            for (size_t idx_col = 0; idx_col < nx_; idx_col++) {
                if (idx_row == idx_col) {
                    values[idx_value] = 1;
                    idx_value++;
                    break;
                }
            }
        }

        // Initial condition
    }

    return true;
}
// [TNLP_eval_jac_g]

// [TNLP_eval_h]
// return the structure or values of the Hessian
bool MultipleShootingNlp::eval_h(Index         n,
                                 const Number *x,
                                 bool          new_x,
                                 Number        obj_factor,
                                 Index         m,
                                 const Number *lambda,
                                 bool          new_lambda,
                                 Index         nele_hess,
                                 Index        *iRow,
                                 Index        *jCol,
                                 Number       *values)
{
    assert(n == nvar_);
    assert(m == nconst_);

    if (values == NULL) {
        // return the structure. This is a symmetric matrix, fill the lower left
        // triangle only

        // Running Costs
        std::size_t idx_value = 0;
        for (size_t idx_block = 0; idx_block < T_; idx_block++) {
            for (size_t idx_row = 0; idx_row < nx_ + nu_; idx_row++) {
                for (size_t idx_col = 0; idx_col < nx_ + nu_; idx_col++) {
                    // We need the lower triangular matrix
                    if (idx_col > idx_row) {
                        break;
                    }
                    iRow[idx_value] = idx_block * (nx_ + nu_) + idx_row;
                    jCol[idx_value] = idx_block * (nx_ + nu_) + idx_col;
                    idx_value++;
                }
            }
        }

        // Terminal costs
        for (size_t idx_row = 0; idx_row < nx_; idx_row++) {
            for (size_t idx_col = 0; idx_col < nx_; idx_col++) {
                // We need the lower triangular matrix
                if (idx_col > idx_row) {
                    break;
                }
                iRow[idx_value] = T_ * (nx_ + nu_) + idx_row;
                jCol[idx_value] = T_ * (nx_ + nu_) + idx_col;
                idx_value++;
            }
        }

        assert(idx_value == nele_hess);
    } else {
        // return the values. This is a symmetric matrix, fill the lower left
        // triangle only
        std::size_t idx_value = 0;

        // Running Costs
        for (size_t idx_block = 0; idx_block < T_; idx_block++) {
            Eigen::VectorXd state      = Eigen::VectorXd::Map(x + idx_block * (nx_ + nu_), nx_);
            Eigen::VectorXd state_next = Eigen::VectorXd::Map(x + (idx_block + 1) * (nx_ + nu_), nx_);
            Eigen::VectorXd control    = Eigen::VectorXd::Map(x + idx_block * (nx_ + nu_) + nx_, nu_);

            problem_->get_runningModels()[idx_block]->calcDiff(problem_->get_runningDatas()[idx_block], state,
                                                               control);
            for (size_t idx_row = 0; idx_row < nx_; idx_row++) {
                for (size_t idx_col = 0; idx_col < nx_; idx_col++) {
                    // We need the lower triangular matrix
                    if (idx_col > idx_row) {
                        break;
                    }
                    values[idx_value] = problem_->get_runningDatas()[idx_block]->Lxx(idx_row, idx_col);
                    idx_value++;
                }
            }

            for (size_t idx_row = 0; idx_row < nu_; idx_row++) {
                for (size_t idx_col = 0; idx_col < nx_; idx_col++) {
                    values[idx_value] = problem_->get_runningDatas()[idx_block]->Lxu(idx_col, idx_row);
                    idx_value++;
                }

                for (size_t idx_col = 0; idx_col < nu_; idx_col++) {
                    if (idx_col > idx_row) {
                        break;
                    }
                    values[idx_value] = problem_->get_runningDatas()[idx_block]->Luu(idx_row, idx_col);
                    idx_value++;
                }
            }
        }

        // Terminal costs
        Eigen::VectorXd state = Eigen::VectorXd::Map(x + T_ * (nx_ + nu_), nx_);
        problem_->get_terminalModel()->calcDiff(problem_->get_terminalData(), state);
        for (size_t idx_row = 0; idx_row < nx_; idx_row++) {
            for (size_t idx_col = 0; idx_col < nx_; idx_col++) {
                // We need the lower triangular matrix
                if (idx_col > idx_row) {
                    break;
                }
                values[idx_value] = problem_->get_terminalData()->Lxx(idx_row, idx_col);
                idx_value++;
            }
        }
    }

    return true;
}
// [TNLP_eval_h]

// [TNLP_finalize_solution]
void MultipleShootingNlp::finalize_solution(SolverReturn               status,
                                            Index                      n,
                                            const Number              *x,
                                            const Number              *z_L,
                                            const Number              *z_U,
                                            Index                      m,
                                            const Number              *g,
                                            const Number              *lambda,
                                            Number                     obj_value,
                                            const IpoptData           *ip_data,
                                            IpoptCalculatedQuantities *ip_cq)
{
    // here is where we would store the solution to variables, or write to a file, etc
    // so we could use the solution.
}
// [TNLP_finalize_solution]

// [TNLP_intermediate_callback]
bool MultipleShootingNlp::intermediate_callback(AlgorithmMode              mode,
                                                Index                      iter,
                                                Number                     obj_value,
                                                Number                     inf_pr,
                                                Number                     inf_du,
                                                Number                     mu,
                                                Number                     d_norm,
                                                Number                     regularization_size,
                                                Number                     alpha_du,
                                                Number                     alpha_pr,
                                                Index                      ls_trials,
                                                const IpoptData           *ip_data,
                                                IpoptCalculatedQuantities *ip_cq)
{
    if (!printiterate_) {
        return true;
    }

    Number x[8];
    Number x_L_viol[8];
    Number x_U_viol[8];
    Number z_L[8];
    Number z_U[8];
    Number compl_x_L[8];
    Number compl_x_U[8];
    Number grad_lag_x[8];

    Number g[6];
    Number lambda[6];
    Number constraint_violation[6];
    Number compl_g[6];

    bool have_iter = get_curr_iterate(ip_data, ip_cq, false, 8, x, z_L, z_U, 6, g, lambda);
    bool have_viol = get_curr_violations(ip_data, ip_cq, false, 8, x_L_viol, x_U_viol, compl_x_L, compl_x_U,
                                         grad_lag_x, 6, constraint_violation, compl_g);

    printf("Current iterate:\n");
    printf("  %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n", "x", "z_L", "z_U", "bound_viol", "compl_x_L", "compl_x_U",
           "grad_lag_x");
    for (int i = 0; i < 8; ++i) {
        if (have_iter) {
            printf("  %-12g %-12g %-12g", x[i], z_L[i], z_U[i]);
        } else {
            printf("  %-12s %-12s %-12s", "n/a", "n/a", "n/a");
        }
        if (have_viol) {
            printf(" %-12g %-12g %-12g %-12g\n", x_L_viol[i] > x_U_viol[i] ? x_L_viol[i] : x_U_viol[i], compl_x_L[i],
                   compl_x_U[i], grad_lag_x[i]);
        } else {
            printf(" %-12s %-12s %-12s %-12s\n", "n/a", "n/a", "n/a", "n/a");
        }
    }

    printf("  %-12s %-12s %-12s %-12s\n", "g(x)", "lambda", "constr_viol", "compl_g");
    for (int i = 0; i < 6; ++i) {
        if (have_iter) {
            printf("  %-12g %-12g", g[i], lambda[i]);
        } else {
            printf("  %-12s %-12s", "n/a", "n/a");
        }
        if (have_viol) {
            printf(" %-12g %-12g\n", constraint_violation[i], compl_g[i]);
        } else {
            printf(" %-12s %-12s\n", "n/a", "n/a");
        }
    }

    return true;
}