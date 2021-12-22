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
      state_(problem_->get_runningModels()[0]->get_state()),
      nx_(problem_->get_nx()),
      ndx_(problem_->get_ndx()),
      nu_(problem_->get_nu_max()),
      T_(problem_->get_T()),
      nconst_(T_ * ndx_ + nx_),         // T*nx eq. constraints for dynamics , nx eq constraints for initial conditions
      nvar_(T_ * (ndx_ + nu_) + ndx_),  // Multiple shooting, states and controls
      is_manifold_(nx_ != ndx_)
{
    xs_.resize(T_ + 1);
    us_.resize(T_);

    for (size_t i = 0; i < T_; i++) {
        xs_[i] = problem_->get_runningModels()[0]->get_state()->zero();
        us_[i] = Eigen::VectorXd::Zero(nu_);
    }
    xs_[T_] = xs_[0];
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
    nnz_jac_g = T_ * ndx_ * (2 * ndx_ + nu_);

    // Jacobian nonzeros for initial condition
    nnz_jac_g += ndx_;

    // Hessian is only affected by costs
    // Running Costs
    std::size_t nonzero = 0;
    for (size_t i = 1; i <= (ndx_ + nu_); i++) {
        nonzero += i;
    }
    nnz_h_lag = T_ * nonzero;

    // Terminal Costs
    nonzero = 0;
    for (size_t i = 1; i <= ndx_; i++) {
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
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);

    // initialize to the given starting point
    // State variable are always at 0 since they represent increments from the given initial point
    for (size_t i = 0; i < T_; i++) {
        for (size_t j = 0; j < ndx_; j++) {
            x[i * (ndx_ + nu_) + j] = 0;
        }

        for (size_t j = 0; j < nu_; j++) {
            x[i * (ndx_ + nu_) + ndx_ + j] = us_[i](j);
        }
    }

    for (size_t j = 0; j < ndx_; j++) {
        x[T_ * (ndx_ + nu_) + j] = 0;
    }

    return true;
}
// [TNLP_get_starting_point]

// [TNLP_eval_f]
// returns the value of the objective function
bool MultipleShootingNlp::eval_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number &obj_value)
{
    assert(n == nvar_);

    // Running costs
    for (size_t i = 0; i < T_; i++) {
        Eigen::VectorXd state   = state_->zero();
        Eigen::VectorXd dstate  = Eigen::VectorXd::Map(x + i * (ndx_ + nu_), ndx_);
        Eigen::VectorXd control = Eigen::VectorXd::Map(x + i * (ndx_ + nu_) + ndx_, nu_);

        state_->integrate(xs_[i], dstate, state);

        problem_->get_runningModels()[i]->calc(problem_->get_runningDatas()[i], state, control);

        obj_value += problem_->get_runningDatas()[i]->cost;
    }

    // Terminal costs
    Eigen::VectorXd state  = state_->zero();
    Eigen::VectorXd dstate = Eigen::VectorXd::Map(x + T_ * (ndx_ + nu_), ndx_);

    state_->integrate(xs_[T_], dstate, state);

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
        Eigen::VectorXd state   = state_->zero();
        Eigen::VectorXd dstate  = Eigen::VectorXd::Map(x + i * (ndx_ + nu_), ndx_);
        Eigen::VectorXd control = Eigen::VectorXd::Map(x + i * (ndx_ + nu_) + ndx_, nu_);

        Eigen::MatrixXd Jx  = Eigen::MatrixXd::Zero(ndx_, ndx_);
        Eigen::MatrixXd Jdx = Eigen::MatrixXd::Zero(ndx_, ndx_);

        state_->integrate(xs_[i], dstate, state);
        problem_->get_runningModels()[i]->calcDiff(problem_->get_runningDatas()[i], state, control);

        state_->Jintegrate(xs_[i], dstate, Jx, Jdx, crocoddyl::second, crocoddyl::setto);
        Eigen::VectorXd Lx = problem_->get_runningDatas()[i]->Lx.transpose() * Jdx;

        for (size_t j = 0; j < ndx_; j++) {
            // grad_f[i * (nx_ + nu_) + j] = problem_->get_runningDatas()[i]->Lx[j];
            grad_f[i * (ndx_ + nu_) + j] = Lx(j);
        }

        for (size_t j = 0; j < nu_; j++) {
            grad_f[i * (ndx_ + nu_) + ndx_ + j] = problem_->get_runningDatas()[i]->Lu[j];
        }
    }

    Eigen::VectorXd state  = state_->zero();
    Eigen::VectorXd dstate = Eigen::VectorXd::Map(x + T_ * (ndx_ + nu_), ndx_);
    state_->integrate(xs_[T_], dstate, state);

    problem_->get_terminalModel()->calcDiff(problem_->get_terminalData(), state);

    Eigen::MatrixXd Jx  = Eigen::MatrixXd::Zero(ndx_, ndx_);
    Eigen::MatrixXd Jdx = Eigen::MatrixXd::Zero(ndx_, ndx_);
    state_->Jintegrate(xs_[T_], dstate, Jx, Jdx, crocoddyl::second, crocoddyl::setto);
    Eigen::VectorXd Lx = problem_->get_terminalData()->Lx.transpose() * Jdx;
    for (size_t j = 0; j < ndx_; j++) {
        grad_f[T_ * (ndx_ + nu_) + j] = Lx(j);
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
        Eigen::VectorXd state       = state_->zero();
        Eigen::VectorXd state_next  = state_->zero();
        Eigen::VectorXd dstate      = Eigen::VectorXd::Map(x + i * (ndx_ + nu_), ndx_);
        Eigen::VectorXd dstate_next = Eigen::VectorXd::Map(x + (i + 1) * (ndx_ + nu_), ndx_);
        Eigen::VectorXd control     = Eigen::VectorXd::Map(x + i * (ndx_ + nu_) + ndx_, nu_);

        state_->integrate(xs_[i], dstate, state);
        state_->integrate(xs_[i + 1], dstate_next, state_next);

        problem_->get_runningModels()[i]->calc(problem_->get_runningDatas()[i], state, control);

        Eigen::VectorXd state_diff = Eigen::VectorXd::Zero(ndx_);
        state_->diff(problem_->get_runningDatas()[i]->xnext, state_next, state_diff);
        // state_next - problem_->get_runningDatas()[i]->xnext;

        for (size_t j = 0; j < ndx_; j++) {
            g[i * nx_ + j] = state_diff[j];
        }
    }

    Eigen::VectorXd state  = state_->zero();
    Eigen::VectorXd dstate = Eigen::VectorXd::Map(x, ndx_);
    state_->integrate(xs_[0], dstate, state);

    for (size_t j = 0; j < nx_; j++) {
        g[T_ * ndx_ + j] = state[j];
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
            for (size_t idx_row = 0; idx_row < ndx_; idx_row++) {
                for (size_t idx_col = 0; idx_col < (2 * ndx_ + nu_); idx_col++) {
                    iRow[idx_value] = idx_block * ndx_ + idx_row;
                    jCol[idx_value] = idx_block * (ndx_ + nu_) + idx_col;
                    // x_k+1 could be more optimized since it is a diagonal
                    idx_value++;
                }
            }
        }

        // Initial condition
        for (size_t idx_row = 0; idx_row < nx_; idx_row++) {
            for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
                if (idx_row == idx_col) {
                    iRow[idx_value] = T_ * ndx_ + idx_row;
                    jCol[idx_value] = idx_col;
                    idx_value++;
                }
            }
        }

        assert(nele_jac == idx_value);

    } else {
        std::size_t idx_value = 0;

        // Dynamic constraints
        for (size_t idx_block = 0; idx_block < T_; idx_block++) {
            Eigen::VectorXd state       = state_->zero();
            Eigen::VectorXd state_next  = state_->zero();
            Eigen::VectorXd dstate      = Eigen::VectorXd::Map(x + idx_block * (ndx_ + nu_), ndx_);
            Eigen::VectorXd dstate_next = Eigen::VectorXd::Map(x + (idx_block + 1) * (ndx_ + nu_), ndx_);
            Eigen::VectorXd control     = Eigen::VectorXd::Map(x + idx_block * (ndx_ + nu_) + ndx_, nu_);

            state_->integrate(xs_[idx_block], dstate, state);
            state_->integrate(xs_[idx_block + 1], dstate_next, state_next);

            problem_->get_runningModels()[idx_block]->calc(problem_->get_runningDatas()[idx_block], state, control);
            problem_->get_runningModels()[idx_block]->calcDiff(problem_->get_runningDatas()[idx_block], state,
                                                               control);

            Eigen::MatrixXd Jsum_x  = Eigen::MatrixXd::Zero(ndx_, ndx_);
            Eigen::MatrixXd Jsum_dx = Eigen::MatrixXd::Zero(ndx_, ndx_);

            Eigen::MatrixXd Jsum_xnext  = Eigen::MatrixXd::Zero(ndx_, ndx_);
            Eigen::MatrixXd Jsum_dxnext = Eigen::MatrixXd::Zero(ndx_, ndx_);

            Eigen::MatrixXd Jdiff_xnext = Eigen::MatrixXd::Zero(ndx_, ndx_);
            Eigen::MatrixXd Jdiff_x     = Eigen::MatrixXd::Zero(ndx_, ndx_);

            state_->Jintegrate(xs_[idx_block], dstate, Jsum_x, Jsum_dx, crocoddyl::second, crocoddyl::setto);
            state_->Jintegrate(xs_[idx_block + 1], dstate_next, Jsum_xnext, Jsum_dxnext, crocoddyl::second,
                               crocoddyl::setto);
            state_->Jdiff(problem_->get_runningDatas()[idx_block]->xnext, state_next, Jdiff_x, Jdiff_xnext,
                          crocoddyl::both);

            Eigen::MatrixXd Jg_dx     = Jdiff_x * problem_->get_runningDatas()[idx_block]->Fx * Jsum_dx;
            Eigen::MatrixXd Jg_u      = Jdiff_x * problem_->get_runningDatas()[idx_block]->Fu;
            Eigen::MatrixXd Jg_dxnext = Jdiff_xnext * Jsum_dxnext;

            for (size_t idx_row = 0; idx_row < ndx_; idx_row++) {
                for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
                    values[idx_value] = Jg_dx(idx_row, idx_col);
                    idx_value++;
                }

                for (size_t idx_col = 0; idx_col < nu_; idx_col++) {
                    values[idx_value] = Jg_u(idx_row, idx_col);
                    idx_value++;
                }

                // This could be more optimized since there are a lot of zeros!
                for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
                    values[idx_value] = Jg_dxnext(idx_row, idx_col);
                    idx_value++;
                }
            }
        }

        Eigen::VectorXd dstate  = Eigen::VectorXd::Map(x, ndx_);
        Eigen::MatrixXd Jsum_x  = Eigen::MatrixXd::Zero(ndx_, ndx_);
        Eigen::MatrixXd Jsum_dx = Eigen::MatrixXd::Zero(ndx_, ndx_);

        state_->Jintegrate(xs_[0], dstate, Jsum_x, Jsum_dx, crocoddyl::second, crocoddyl::setto);

        for (size_t idx_row = 0; idx_row < nx_; idx_row++) {
            for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
                if (idx_row == idx_col) {
                    values[idx_value] = Jsum_dx(idx_row, idx_col);
                    idx_value++;
                }
            }
        }
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
            for (size_t idx_row = 0; idx_row < ndx_ + nu_; idx_row++) {
                for (size_t idx_col = 0; idx_col < ndx_ + nu_; idx_col++) {
                    // We need the lower triangular matrix
                    if (idx_col > idx_row) {
                        break;
                    }
                    iRow[idx_value] = idx_block * (ndx_ + nu_) + idx_row;
                    jCol[idx_value] = idx_block * (ndx_ + nu_) + idx_col;
                    idx_value++;
                }
            }
        }

        // Terminal costs
        for (size_t idx_row = 0; idx_row < ndx_; idx_row++) {
            for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
                // We need the lower triangular matrix
                if (idx_col > idx_row) {
                    break;
                }
                iRow[idx_value] = T_ * (ndx_ + nu_) + idx_row;
                jCol[idx_value] = T_ * (ndx_ + nu_) + idx_col;
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
            Eigen::VectorXd state   = state_->zero();
            Eigen::VectorXd dstate  = Eigen::VectorXd::Map(x + idx_block * (ndx_ + nu_), ndx_);
            Eigen::VectorXd control = Eigen::VectorXd::Map(x + idx_block * (ndx_ + nu_) + ndx_, nu_);

            state_->integrate(xs_[idx_block], dstate, state);

            problem_->get_runningModels()[idx_block]->calcDiff(problem_->get_runningDatas()[idx_block], state,
                                                               control);
            Eigen::MatrixXd Jsum_x  = Eigen::MatrixXd::Zero(ndx_, ndx_);
            Eigen::MatrixXd Jsum_dx = Eigen::MatrixXd::Zero(ndx_, ndx_);

            state_->Jintegrate(xs_[idx_block], dstate, Jsum_x, Jsum_dx, crocoddyl::second, crocoddyl::setto);
            Eigen::MatrixXd Lxx = Jsum_dx.transpose() * problem_->get_runningDatas()[idx_block]->Lxx * Jsum_dx;
            Eigen::MatrixXd Lxu = Jsum_dx.transpose() * problem_->get_runningDatas()[idx_block]->Lxu;

            for (size_t idx_row = 0; idx_row < ndx_; idx_row++) {
                for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
                    // We need the lower triangular matrix
                    if (idx_col > idx_row) {
                        break;
                    }
                    values[idx_value] = Lxx(idx_row, idx_col);
                    idx_value++;
                }
            }

            for (size_t idx_row = 0; idx_row < nu_; idx_row++) {
                for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
                    values[idx_value] = Lxu(idx_col, idx_row);
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
        Eigen::VectorXd state  = state_->zero();
        Eigen::VectorXd dstate = Eigen::VectorXd::Map(x + T_ * (ndx_ + nu_), ndx_);

        state_->integrate(xs_[T_], dstate, state);

        problem_->get_terminalModel()->calcDiff(problem_->get_terminalData(), state);

        Eigen::MatrixXd Jsum_x  = Eigen::MatrixXd::Zero(ndx_, ndx_);
        Eigen::MatrixXd Jsum_dx = Eigen::MatrixXd::Zero(ndx_, ndx_);
        state_->Jintegrate(xs_[T_], dstate, Jsum_x, Jsum_dx, crocoddyl::second, crocoddyl::setto);

        Eigen::MatrixXd Lxx = Jsum_dx.transpose() * problem_->get_terminalData()->Lxx * Jsum_dx;

        // Eigen::VectorXd state = Eigen::VectorXd::Map(x + T_ * (nx_ + nu_), nx_);
        // problem_->get_terminalModel()->calcDiff(problem_->get_terminalData(), state);
        for (size_t idx_row = 0; idx_row < ndx_; idx_row++) {
            for (size_t idx_col = 0; idx_col < ndx_; idx_col++) {
                // We need the lower triangular matrix
                if (idx_col > idx_row) {
                    break;
                }
                values[idx_value] = Lxx(idx_row, idx_col);
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
    // Copy the solution to vector once solver is finished
    for (size_t i = 0; i < T_; i++) {
        Eigen::VectorXd state   = state_->zero();
        Eigen::VectorXd dstate  = Eigen::VectorXd::Map(x + i * (ndx_ + nu_), ndx_);
        Eigen::VectorXd control = Eigen::VectorXd::Map(x + i * (ndx_ + nu_) + ndx_, nu_);

        state_->integrate(xs_[i], dstate, state);
        xs_[i] = state;
        us_[i] = control;
    }

    Eigen::VectorXd state  = state_->zero();
    Eigen::VectorXd dstate = Eigen::VectorXd::Map(x + T_ * (ndx_ + nu_), ndx_);

    state_->integrate(xs_[T_], dstate, state);

    xs_[T_] = state;
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

void MultipleShootingNlp::set_xs(const std::vector<Eigen::VectorXd> &xs) { xs_ = xs; }

void MultipleShootingNlp::set_us(const std::vector<Eigen::VectorXd> &us) { us_ = us; }

const std::size_t &MultipleShootingNlp::get_nvar() const { return nvar_; }

const std::vector<Eigen::VectorXd> &MultipleShootingNlp::get_xs() const { return xs_; }

const std::vector<Eigen::VectorXd> &MultipleShootingNlp::get_us() const { return us_; }

const boost::shared_ptr<crocoddyl::ShootingProblem> &MultipleShootingNlp::get_problem() const { return problem_; }
