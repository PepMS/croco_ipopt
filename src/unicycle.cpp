#include "coin-or/IpIpoptApplication.hpp"

#include "multiple-shooting-nlp.hpp"

#include <cassert>
#include <cmath>

#include <iostream>

using namespace Ipopt;

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

    // Create a new instance of your nlp
    //  (use a SmartPtr, not raw)
    SmartPtr<TNLP> mynlp = new MultipleShootingNlp(problem, false);

    // Create a new instance of IpoptApplication
    //  (use a SmartPtr, not raw)
    // We are using the factory, since this allows us to compile this
    // example with an Ipopt Windows DLL
    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

    // Change some options
    // Note: The following choices are only examples, they might not be
    //       suitable for your optimization problem.
    app->Options()->SetNumericValue("tol", 3.82e-6);
    app->Options()->SetStringValue("mu_strategy", "adaptive");
    app->Options()->SetStringValue("output_file", "ipopt.out");
    // The following overwrites the default name (ipopt.opt) of the options file
    // app->Options()->SetStringValue("option_file_name", "hs071.opt");

    // Initialize the IpoptApplication and process the options
    ApplicationReturnStatus status;
    status = app->Initialize();
    if (status != Solve_Succeeded) {
        std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
        return (int)status;
    }

    // Ask Ipopt to solve the problem
    status = app->OptimizeTNLP(mynlp);

    if (status == Solve_Succeeded) {
        std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
    } else {
        std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
    }

    // As the SmartPtrs go out of scope, the reference count
    // will be decremented and the objects will automatically
    // be deleted.

    return (int)status;
}