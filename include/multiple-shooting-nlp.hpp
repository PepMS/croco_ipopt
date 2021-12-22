#ifndef __MULTIPLE_SHOOTING_NLP_HPP__
#define __MULTIPLE_SHOOTING_NLP_HPP__

#include "coin-or/IpTNLP.hpp"

#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"

class MultipleShootingNlp : public Ipopt::TNLP
{
    public:
    /** Constructor */
    MultipleShootingNlp(const boost::shared_ptr<crocoddyl::ShootingProblem> &problem);

    /** Destructor */
    virtual ~MultipleShootingNlp();

    /**@name Overloaded from TNLP */
    //@{
    /** Method to return some info about the NLP */
    virtual bool get_nlp_info(Ipopt::Index   &n,
                              Ipopt::Index   &m,
                              Ipopt::Index   &nnz_jac_g,
                              Ipopt::Index   &nnz_h_lag,
                              IndexStyleEnum &index_style);

    /** Method to return the bounds for my problem */
    virtual bool get_bounds_info(Ipopt::Index   n,
                                 Ipopt::Number *x_l,
                                 Ipopt::Number *x_u,
                                 Ipopt::Index   m,
                                 Ipopt::Number *g_l,
                                 Ipopt::Number *g_u);

    /** Method to return the starting point for the algorithm */
    virtual bool get_starting_point(Ipopt::Index   n,
                                    bool           init_x,
                                    Ipopt::Number *x,
                                    bool           init_z,
                                    Ipopt::Number *z_L,
                                    Ipopt::Number *z_U,
                                    Ipopt::Index   m,
                                    bool           init_lambda,
                                    Ipopt::Number *lambda);

    /** Method to return the objective value */
    virtual bool eval_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number &obj_value);

    /** Method to return the gradient of the objective */
    virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number *grad_f);

    /** Method to return the constraint residuals */
    virtual bool eval_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Index m, Ipopt::Number *g);

    /** Method to return:
     *   1) The structure of the jacobian (if "values" is NULL)
     *   2) The values of the jacobian (if "values" is not NULL)
     */
    virtual bool eval_jac_g(Ipopt::Index         n,
                            const Ipopt::Number *x,
                            bool                 new_x,
                            Ipopt::Index         m,
                            Ipopt::Index         nele_jac,
                            Ipopt::Index        *iRow,
                            Ipopt::Index        *jCol,
                            Ipopt::Number       *values);

    /** Method to return:
     *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
     *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
     */
    virtual bool eval_h(Ipopt::Index         n,
                        const Ipopt::Number *x,
                        bool                 new_x,
                        Ipopt::Number        obj_factor,
                        Ipopt::Index         m,
                        const Ipopt::Number *lambda,
                        bool                 new_lambda,
                        Ipopt::Index         nele_hess,
                        Ipopt::Index        *iRow,
                        Ipopt::Index        *jCol,
                        Ipopt::Number       *values);

    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(Ipopt::SolverReturn               status,
                                   Ipopt::Index                      n,
                                   const Ipopt::Number              *x,
                                   const Ipopt::Number              *z_L,
                                   const Ipopt::Number              *z_U,
                                   Ipopt::Index                      m,
                                   const Ipopt::Number              *g,
                                   const Ipopt::Number              *lambda,
                                   Ipopt::Number                     obj_value,
                                   const Ipopt::IpoptData           *ip_data,
                                   Ipopt::IpoptCalculatedQuantities *ip_cq);
    //@}

    bool intermediate_callback(Ipopt::AlgorithmMode              mode,
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
                               Ipopt::IpoptCalculatedQuantities *ip_cq);

    const std::size_t &get_nvar() const;

    const std::vector<Eigen::VectorXd>                  &get_xs() const;
    const std::vector<Eigen::VectorXd>                  &get_us() const;
    const boost::shared_ptr<crocoddyl::ShootingProblem> &get_problem() const;

      void set_xs(const std::vector<Eigen::VectorXd>& xs);
      void set_us(const std::vector<Eigen::VectorXd>& us);


    private:
    boost::shared_ptr<crocoddyl::ShootingProblem> problem_;
    boost::shared_ptr<crocoddyl::StateAbstract>   state_;

    std::vector<Eigen::VectorXd> xs_;
    std::vector<Eigen::VectorXd> us_;

    std::size_t nx_;
    std::size_t ndx_;
    std::size_t nu_;
    std::size_t T_;

    std::size_t nconst_;
    std::size_t nvar_;

    bool is_manifold_;

    /**@name Methods to block default compiler methods.
     *
     * The compiler automatically generates the following three methods.
     *  Since the default compiler implementation is generally not what
     *  you want (for all but the most simple classes), we usually
     *  put the declarations of these methods in the private section
     *  and never implement them. This prevents the compiler from
     *  implementing an incorrect "default" behavior without us
     *  knowing. (See Scott Meyers book, "Effective C++")
     */
    //@{
    MultipleShootingNlp(const MultipleShootingNlp &);

    MultipleShootingNlp &operator=(const MultipleShootingNlp &);
    //@}
};

#endif
