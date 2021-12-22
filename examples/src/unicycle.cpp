#include "coin-or/IpIpoptApplication.hpp"

#include "ipopt.hpp"

#include <cassert>
#include <cmath>

#include <iostream>

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

int main(int /*argv*/, char** /*argc*/)
{
    // Create the Croco problem
    boost::shared_ptr<crocoddyl::ActionModelUnicycle> model = boost::make_shared<crocoddyl::ActionModelUnicycle>();

    Eigen::Vector2d weights;
    weights << 10, 1;
    model->set_cost_weights(weights);

    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> models(20, model);

    Eigen::Vector3d x0;
    x0 << -1, -1, 1;
    boost::shared_ptr<crocoddyl::ShootingProblem> problem =
        boost::make_shared<crocoddyl::ShootingProblem>(x0, models, model);

    // Instantiate Ipopt solver wrapper
    SolverIpOpt solver(problem);

    // Solve the OCP problem
    int solved = solver.solve();

    return solved;
}