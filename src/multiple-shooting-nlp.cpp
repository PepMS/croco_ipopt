#include "coin-or/IpIpoptApplication.hpp"

#include "multiple-shooting-nlp.hpp"

#include <cassert>
#include <cmath>

#include <iostream>

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

MultipleShootingNlp::MultipleShootingNlp(const boost::shared_ptr<crocoddyl::ShootingProblem> &problem)
    : problem_(problem),
      nx_(problem_->get_nx()),
      nu_(problem_->get_nu_max()),
      T_(problem_->get_T()),
      nvar_(T_ * (nx_ + nu_) + nx_),  // Multiple shooting, states and controls
      nconst_((T_ + 1) * nx_)         // T*nx eq. constraints for dynamics , nx eq constraints for initial conditions
{
    xs_.resize(T_ + 1);
    us_.resize(T_);
}

MultipleShootingNlp::~MultipleShootingNlp() {}

bool MultipleShootingNlp::get_nlp_info(Ipopt::Index   &n,
                                       Ipopt::Index   &m,
                                       Ipopt::Index   &nnz_jac_g,
                                       Ipopt::Index   &nnz_h_lag,
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
    index_style = Ipopt::TNLP::C_STYLE;

    return true;
}
// [TNLP_get_nlp_info]

// [TNLP_get_bounds_info]
// returns the variable bounds
bool MultipleShootingNlp::get_bounds_info(Ipopt::Index   n,
                                          Ipopt::Number *x_l,
                                          Ipopt::Number *x_u,
                                          Ipopt::Index   m,
                                          Ipopt::Number *g_l,
                                          Ipopt::Number *g_u)
{
    // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
    // If desired, we could assert to make sure they are what we think they are.
    assert(n == nvar_);
    assert(m == nconst_);

    // the variables have lower bounds of 1
    for (Ipopt::Index i = 0; i < n; i++) {
        x_l[i] = -2e19;
        x_u[i] = 2e19;
    }

    // Dynamics
    for (Ipopt::Index i = 0; i < nconst_ - nx_; i++) {
        g_l[i] = 0;
        g_u[i] = 0;
    }

    for (Ipopt::Index i = 0; i < nx_; i++) {
        g_l[nconst_ - nx_ + i] = problem_->get_x0()[i];
        g_u[nconst_ - nx_ + i] = problem_->get_x0()[i];
    }

    return true;
}
// [TNLP_get_bounds_info]

// [TNLP_get_starting_point]
// returns the initial point for the problem
bool MultipleShootingNlp::get_starting_point(Ipopt::Index   n,
                                             bool           init_x,
                                             Ipopt::Number *x,
                                             bool           init_z,
                                             Ipopt::Number *z_L,
                                             Ipopt::Number *z_U,
                                             Ipopt::Index   m,
                                             bool           init_lambda,
                                             Ipopt::Number *lambda)
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
bool MultipleShootingNlp::eval_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number &obj_value)
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
bool MultipleShootingNlp::eval_grad_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number *grad_f)
{
    assert(n == nvar_);

    for (size_t i = 0; i < T_; i++) {
        Eigen::VectorXd state   = Eigen::VectorXd::Map(x + i * (nx_ + nu_), nx_);
        Eigen::VectorXd control = Eigen::VectorXd::Map(x + i * (nx_ + nu_) + nx_, nu_);

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
bool MultipleShootingNlp::eval_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Index m, Ipopt::Number *g)
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
bool MultipleShootingNlp::eval_jac_g(Ipopt::Index         n,
                                     const Ipopt::Number *x,
                                     bool                 new_x,
                                     Ipopt::Index         m,
                                     Ipopt::Index         nele_jac,
                                     Ipopt::Index        *iRow,
                                     Ipopt::Index        *jCol,
                                     Ipopt::Number       *values)
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
bool MultipleShootingNlp::eval_h(Ipopt::Index         n,
                                 const Ipopt::Number *x,
                                 bool                 new_x,
                                 Ipopt::Number        obj_factor,
                                 Ipopt::Index         m,
                                 const Ipopt::Number *lambda,
                                 bool                 new_lambda,
                                 Ipopt::Index         nele_hess,
                                 Ipopt::Index        *iRow,
                                 Ipopt::Index        *jCol,
                                 Ipopt::Number       *values)
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
void MultipleShootingNlp::finalize_solution(Ipopt::SolverReturn               status,
                                            Ipopt::Index                      n,
                                            const Ipopt::Number              *x,
                                            const Ipopt::Number              *z_L,
                                            const Ipopt::Number              *z_U,
                                            Ipopt::Index                      m,
                                            const Ipopt::Number              *g,
                                            const Ipopt::Number              *lambda,
                                            Ipopt::Number                     obj_value,
                                            const Ipopt::IpoptData           *ip_data,
                                            Ipopt::IpoptCalculatedQuantities *ip_cq)
{
    for (size_t i = 0; i < T_; i++) {
        Eigen::VectorXd state   = Eigen::VectorXd::Map(x + i * (nx_ + nu_), nx_);
        Eigen::VectorXd control = Eigen::VectorXd::Map(x + i * (nx_ + nu_) + nx_, nu_);

        xs_[i] = state;
        us_[i] = control;
    }

    Eigen::VectorXd state = Eigen::VectorXd::Map(x + T_ * (nx_ + nu_), nx_);
    xs_[T_]               = state;
}
// [TNLP_finalize_solution]

// [TNLP_intermediate_callback]
bool MultipleShootingNlp::intermediate_callback(Ipopt::AlgorithmMode              mode,
                                                Ipopt::Index                      iter,
                                                Ipopt::Number                     obj_value,
                                                Ipopt::Number                     inf_pr,
                                                Ipopt::Number                     inf_du,
                                                Ipopt::Number                     mu,
                                                Ipopt::Number                     d_norm,
                                                Ipopt::Number                     regularization_size,
                                                Ipopt::Number                     alpha_du,
                                                Ipopt::Number                     alpha_pr,
                                                Ipopt::Index                      ls_trials,
                                                const Ipopt::IpoptData           *ip_data,
                                                Ipopt::IpoptCalculatedQuantities *ip_cq)
{
    return true;
}

const std::size_t &MultipleShootingNlp::get_nvar() const { return nvar_; }

const std::vector<Eigen::VectorXd> &MultipleShootingNlp::get_xs() const { return xs_; }

const std::vector<Eigen::VectorXd> &MultipleShootingNlp::get_us() const { return us_; }
