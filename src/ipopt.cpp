#include "ipopt.hpp"

SolverIpOpt::SolverIpOpt(const boost::shared_ptr<crocoddyl::ShootingProblem>& problem)
    : ms_nlp_(new MultipleShootingNlp(problem)), app_(IpoptApplicationFactory())
{
    app_->Options()->SetNumericValue("tol", 3.82e-6);
    app_->Options()->SetStringValue("mu_strategy", "adaptive");
    // app_->Options()->SetStringValue("max_iter", 100);
    // app->Options()->SetStringValue("output_file", "ipopt.out");

    status_ = app_->Initialize();

    if (status_ != Ipopt::Solve_Succeeded) {
        // Throw an exception here
        std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
    }
}

bool SolverIpOpt::solve(const std::vector<Eigen::VectorXd>& init_xs,
                        const std::vector<Eigen::VectorXd>& init_us,
                        const std::size_t                   maxiter)
{
    assert(init_xs.size() == ms_nlp_->get_problem()->get_T() + 1);
    assert(init_us.size() == ms_nlp_->get_problem()->get_T());

    ms_nlp_->set_us(init_us);
    ms_nlp_->set_xs(init_xs);

    app_->Options()->SetIntegerValue("max_iter", maxiter);
    bool suceed = solve();
    return suceed;
}

bool SolverIpOpt::solve()
{
    Ipopt::ApplicationReturnStatus status = app_->OptimizeTNLP(ms_nlp_);

    return status == Ipopt::Solve_Succeeded;
}

SolverIpOpt::~SolverIpOpt() {}

const std::vector<Eigen::VectorXd>& SolverIpOpt::get_xs() const { return ms_nlp_->get_xs(); }

const std::vector<Eigen::VectorXd>& SolverIpOpt::get_us() const { return ms_nlp_->get_us(); }
