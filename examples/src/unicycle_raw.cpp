#include "coin-or/IpIpoptApplication.hpp"

#include "unicycle_raw.hpp"

#include <cassert>
#include <cmath>

#include <iostream>

using namespace Ipopt;

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// constructor
Unicycle::Unicycle(
    bool printiterate) : printiterate_(printiterate)
{
}

// destructor
Unicycle::~Unicycle()
{
}

// [TNLP_get_nlp_info]
// returns the size of the problem
bool Unicycle::get_nlp_info(
    Index &n,
    Index &m,
    Index &nnz_jac_g,
    Index &nnz_h_lag,
    IndexStyleEnum &index_style)
{
   // The problem described in Unicycle.hpp has 4 variables, x[0] through x[3]
   n = 8;

   // one equality constraint and one inequality constraint
   m = 6;

   // in this example the jacobian is dense and contains 8 nonzeros
   nnz_jac_g = 14; // 11 dyn., 3 i. cond.

   // the Hessian is also dense and has 16 total nonzeros, but we
   // only need the lower left corner (since it is symmetric)
   nnz_h_lag = 8;

   // use the C style indexing (0-based)
   index_style = TNLP::C_STYLE;

   return true;
}
// [TNLP_get_nlp_info]

// [TNLP_get_bounds_info]
// returns the variable bounds
bool Unicycle::get_bounds_info(
    Index n,
    Number *x_l,
    Number *x_u,
    Index m,
    Number *g_l,
    Number *g_u)
{
   // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
   // If desired, we could assert to make sure they are what we think they are.
   assert(n == 8);
   assert(m == 6);

   // the variables have lower bounds of 1
   for (Index i = 0; i < n; i++)
   {
      x_l[i] = -2e19;
      x_u[i] = 2e19;
   }

   // Dynamics
   for (Index i = 0; i < 3; i++)
   {
      g_l[i] = 0;
      g_u[i] = 0;
   }

   // Initial conditions
   g_l[3] = -1;
   g_l[4] = -1;
   g_l[5] = 1;

   g_u[3] = -1;
   g_u[4] = -1;
   g_u[5] = 1;

   return true;
}
// [TNLP_get_bounds_info]

// [TNLP_get_starting_point]
// returns the initial point for the problem
bool Unicycle::get_starting_point(
    Index n,
    bool init_x,
    Number *x,
    bool init_z,
    Number *z_L,
    Number *z_U,
    Index m,
    bool init_lambda,
    Number *lambda)
{
   // Here, we assume we only have starting values for x, if you code
   // your own NLP, you can provide starting values for the dual variables
   // if you wish
   assert(init_x == true);
   assert(init_z == false);
   assert(init_lambda == false);

   // initialize to the given starting point
   x[0] = 0.0;
   x[1] = 0.0;
   x[2] = 0.0;
   x[3] = 0.0;
   x[4] = 0.0;
   x[5] = 0.0;
   x[6] = 0.0;
   x[7] = 0.0;

   return true;
}
// [TNLP_get_starting_point]

// [TNLP_eval_f]
// returns the value of the objective function
bool Unicycle::eval_f(
    Index n,
    const Number *x,
    bool new_x,
    Number &obj_value)
{
   assert(n == 8);

   double dt = 0.1;

   obj_value = 0.5 * (100 * x[0] * x[0] +
                      100 * x[1] * x[1] +
                      100 * x[2] * x[2] +
                      x[3] * x[3] +
                      x[4] * x[4] +
                      100 * x[5] * x[5] +
                      100 * x[6] * x[6] +
                      100 * x[7] * x[7]);

   return true;
}
// [TNLP_eval_f]

// [TNLP_eval_grad_f]
// return the gradient of the objective function grad_{x} f(x)
bool Unicycle::eval_grad_f(
    Index n,
    const Number *x,
    bool new_x,
    Number *grad_f)
{
   assert(n == 8);

   double dt = 0.1;

   grad_f[0] = 100 * x[0];
   grad_f[1] = 100 * x[1];
   grad_f[2] = 100 * x[2];
   grad_f[3] = x[3];
   grad_f[4] = x[4];
   grad_f[5] = 100 * x[5];
   grad_f[6] = 100 * x[6];
   grad_f[7] = 100 * x[7];

   return true;
}
// [TNLP_eval_grad_f]

// [TNLP_eval_g]
// return the value of the constraints: g(x)
bool Unicycle::eval_g(
    Index n,
    const Number *x,
    bool new_x,
    Index m,
    Number *g)
{
   assert(n == 8);
   assert(m == 6);

   double dt = 0.1;

   g[0] = x[5] - x[0] - cos(x[2]) * x[3] * dt;
   g[1] = x[6] - x[1] - sin(x[2]) * x[3] * dt;
   g[2] = x[7] - x[2] - x[4] * dt;

   g[3] = x[0];
   g[4] = x[1];
   g[5] = x[2];

   return true;
}
// [TNLP_eval_g]

// [TNLP_eval_jac_g]
// return the structure or values of the Jacobian
bool Unicycle::eval_jac_g(
    Index n,
    const Number *x,
    bool new_x,
    Index m,
    Index nele_jac,
    Index *iRow,
    Index *jCol,
    Number *values)
{
   assert(n == 8);
   assert(m == 6);

   double dt = 0.1;

   if (values == NULL)
   {
      // return the structure of the Jacobian

      // this particular Jacobian is dense
      iRow[0] = 0;
      jCol[0] = 0;

      iRow[1] = 0;
      jCol[1] = 2;

      iRow[2] = 0;
      jCol[2] = 3;

      iRow[3] = 0;
      jCol[3] = 5;

      iRow[4] = 1;
      jCol[4] = 1;

      iRow[5] = 1;
      jCol[5] = 2;

      iRow[6] = 1;
      jCol[6] = 3;

      iRow[7] = 1;
      jCol[7] = 6;

      iRow[8] = 2;
      jCol[8] = 2;

      iRow[9] = 2;
      jCol[9] = 4;

      iRow[10] = 2;
      jCol[10] = 7;

      iRow[11] = 3;
      jCol[11] = 0;

      iRow[12] = 4;
      jCol[12] = 1;

      iRow[13] = 5;
      jCol[13] = 2;
   }
   else
   {
      // return the values of the Jacobian of the constraints

      values[0] = -1;                    // 0,0
      values[1] = sin(x[2]) * x[3] * dt; // 0,2
      values[2] = -cos(x[2]) * dt;       // 0,3
      values[3] = 1;                     // 0,5

      values[4] = -1;                     // 1,1
      values[5] = -cos(x[2]) * x[3] * dt; // 1,2
      values[6] = -sin(x[2]) * dt;        // 1,3
      values[7] = 1;                      // 1,6

      values[8] = -1;  // 2,2
      values[9] = -dt; // 2,4
      values[10] = 1;  // 2,7

      values[11] = 1; // 3,0
      values[12] = 1; // 4,1
      values[13] = 1; // 5,2
   }

   return true;
}
// [TNLP_eval_jac_g]

// [TNLP_eval_h]
// return the structure or values of the Hessian
bool Unicycle::eval_h(
    Index n,
    const Number *x,
    bool new_x,
    Number obj_factor,
    Index m,
    const Number *lambda,
    bool new_lambda,
    Index nele_hess,
    Index *iRow,
    Index *jCol,
    Number *values)
{
   assert(n == 8);
   assert(m == 6);

   if (values == NULL)
   {
      // return the structure. This is a symmetric matrix, fill the lower left
      // triangle only.

      // the hessian for this problem is actually dense
      Index idx = 0;
      for (Index row = 0; row < n; row++)
      {
         iRow[idx] = row;
         jCol[idx] = row;
         idx++;
      }

      assert(idx == nele_hess);
   }
   else
   {
      // return the values. This is a symmetric matrix, fill the lower left
      // triangle only

      // fill the objective portion
      values[0] = obj_factor * 100; // 0,0

      values[1] = obj_factor * 100; // 1,0
      values[2] = obj_factor * 100; // 1,1

      values[3] = obj_factor * 1; // 2,0
      values[4] = obj_factor * 1; // 2,0

      values[5] = obj_factor * 100; // 3,0
      values[6] = obj_factor * 100; // 3,1
      values[7] = obj_factor * 100; // 3,2
   }

   return true;
}
// [TNLP_eval_h]

// [TNLP_finalize_solution]
void Unicycle::finalize_solution(
    SolverReturn status,
    Index n,
    const Number *x,
    const Number *z_L,
    const Number *z_U,
    Index m,
    const Number *g,
    const Number *lambda,
    Number obj_value,
    const IpoptData *ip_data,
    IpoptCalculatedQuantities *ip_cq)
{
   // here is where we would store the solution to variables, or write to a file, etc
   // so we could use the solution.

   // For this example, we write the solution to the console
   std::cout << std::endl
             << std::endl
             << "Solution of the primal variables, x" << std::endl;
   for (Index i = 0; i < n; i++)
   {
      std::cout << "x[" << i << "] = " << x[i] << std::endl;
   }

   std::cout << std::endl
             << std::endl
             << "Solution of the bound multipliers, z_L and z_U" << std::endl;
   for (Index i = 0; i < n; i++)
   {
      std::cout << "z_L[" << i << "] = " << z_L[i] << std::endl;
   }
   for (Index i = 0; i < n; i++)
   {
      std::cout << "z_U[" << i << "] = " << z_U[i] << std::endl;
   }

   std::cout << std::endl
             << std::endl
             << "Objective value" << std::endl;
   std::cout << "f(x*) = " << obj_value << std::endl;

   std::cout << std::endl
             << "Final value of the constraints:" << std::endl;
   for (Index i = 0; i < m; i++)
   {
      std::cout << "g(" << i << ") = " << g[i] << std::endl;
   }
}
// [TNLP_finalize_solution]

// [TNLP_intermediate_callback]
bool Unicycle::intermediate_callback(
    AlgorithmMode mode,
    Index iter,
    Number obj_value,
    Number inf_pr,
    Number inf_du,
    Number mu,
    Number d_norm,
    Number regularization_size,
    Number alpha_du,
    Number alpha_pr,
    Index ls_trials,
    const IpoptData *ip_data,
    IpoptCalculatedQuantities *ip_cq)
{
   if (!printiterate_)
   {
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
   bool have_viol = get_curr_violations(ip_data, ip_cq, false, 8, x_L_viol, x_U_viol, compl_x_L, compl_x_U, grad_lag_x, 6, constraint_violation, compl_g);

   printf("Current iterate:\n");
   printf("  %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n", "x", "z_L", "z_U", "bound_viol", "compl_x_L", "compl_x_U", "grad_lag_x");
   for (int i = 0; i < 8; ++i)
   {
      if (have_iter)
      {
         printf("  %-12g %-12g %-12g", x[i], z_L[i], z_U[i]);
      }
      else
      {
         printf("  %-12s %-12s %-12s", "n/a", "n/a", "n/a");
      }
      if (have_viol)
      {
         printf(" %-12g %-12g %-12g %-12g\n", x_L_viol[i] > x_U_viol[i] ? x_L_viol[i] : x_U_viol[i], compl_x_L[i], compl_x_U[i], grad_lag_x[i]);
      }
      else
      {
         printf(" %-12s %-12s %-12s %-12s\n", "n/a", "n/a", "n/a", "n/a");
      }
   }

   printf("  %-12s %-12s %-12s %-12s\n", "g(x)", "lambda", "constr_viol", "compl_g");
   for (int i = 0; i < 6; ++i)
   {
      if (have_iter)
      {
         printf("  %-12g %-12g", g[i], lambda[i]);
      }
      else
      {
         printf("  %-12s %-12s", "n/a", "n/a");
      }
      if (have_viol)
      {
         printf(" %-12g %-12g\n", constraint_violation[i], compl_g[i]);
      }
      else
      {
         printf(" %-12s %-12s\n", "n/a", "n/a");
      }
   }

   return true;
}

int main(
    int /*argv*/,
    char ** /*argc*/
)
{
   // Create a new instance of your nlp
   //  (use a SmartPtr, not raw)
   SmartPtr<TNLP> mynlp = new Unicycle(false);

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
   if (status != Solve_Succeeded)
   {
      std::cout << std::endl
                << std::endl
                << "*** Error during initialization!" << std::endl;
      return (int)status;
   }

   // Ask Ipopt to solve the problem
   status = app->OptimizeTNLP(mynlp);

   if (status == Solve_Succeeded)
   {
      std::cout << std::endl
                << std::endl
                << "*** The problem solved!" << std::endl;
   }
   else
   {
      std::cout << std::endl
                << std::endl
                << "*** The problem FAILED!" << std::endl;
   }

   // As the SmartPtrs go out of scope, the reference count
   // will be decremented and the objects will automatically
   // be deleted.

   return (int)status;
}