
#include <boost/python.hpp>
#include "ipopt.hpp"

namespace bp = boost::python;

BOOST_PYTHON_MODULE(libipopt_croco_pywrap)
{
    bool (SolverIpOpt::*solve_no_args)() = &SolverIpOpt::solve;
    bp::class_<SolverIpOpt>("SolverIpOpt", bp::init<const boost::shared_ptr<crocoddyl::ShootingProblem>&>(
                                               bp::args("self", "problem"), "Initialize multicopter params"))
        .def("solve", solve_no_args, bp::args("self"), "Solves using ipopt")
        .add_property("us",
                      bp::make_function(&SolverIpOpt::get_us, bp::return_value_policy<bp::copy_const_reference>()))
        .add_property("xs",
                      bp::make_function(&SolverIpOpt::get_xs, bp::return_value_policy<bp::copy_const_reference>()));
}