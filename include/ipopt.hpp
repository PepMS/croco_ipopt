#ifndef __IPOPT_HPP__
#define __IPOPT_HPP__

#include "coin-or/IpIpoptApplication.hpp"

#include "multiple-shooting-nlp.hpp"

class SolverIpOpt
{
    public:
    SolverIpOpt(const boost::shared_ptr<crocoddyl::ShootingProblem>& problem);
    ~SolverIpOpt();

    bool solve(const std::vector<Eigen::VectorXd>& init_xs,
               const std::vector<Eigen::VectorXd>& init_us,
               const std::size_t                   maxiter);

    bool solve();

    const std::vector<Eigen::VectorXd>& get_xs() const;
    const std::vector<Eigen::VectorXd>& get_us() const;

    private:
    Ipopt::SmartPtr<MultipleShootingNlp>     ms_nlp_;
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app_;
    Ipopt::ApplicationReturnStatus           status_;
};

#endif
