#ifndef _MULTIGRIDSOLVER_HEADER
#define _MULTIGRIDSOLVER_HEADER
#include <bitset>
#include <assert.h>
#include <iosfwd>
#include <vector>
#include <climits>
#include <iomanip>
#include <complex>
#include "grid.h"
#include "multigrid.h"

    //=============================================
    //                                           //
    // A general multigrid solver to solve       //
    // PDEs in a domain with periodic BC         //
    // in any dimension.                         //
    //                                           //
    // Implement: l_operator, dl_operator        //
    // and (optional) check_convergence          //
    //                                           //
    // The standard equation that is implemented // 
    // is Poissons eq.: D^2phi  = rho, see       //
    // poisson_solver.h for this and how one can //
    // use this class to solve general equations //
    //                                           //
    // Any external fields needed to define the  //
    // PDE can be added to a grid-list           //
    // _ext_fields by running add_external_field // 
    //                                           //
    // _MAXSTEPS defines how many V-cycles we    //
    // do before giving up if convergence is not //
    // reached. Change by running [set_maxsteps] //
    //                                           //
    // _EPS_CONVERGE is a parameters defining    //
    // convergence. In standard implementation   //
    // we define convergence as _rms_res < eps   //
    // Change by running [set_epsilon]           //
    //                                           //
    // _NGRIDCOLOURS defines in which order we   //
    // sweep through the grid: sum of int-coord  //
    // mod _NGRIDCOLOURS. For 2 we have standard //
    // chess-board ordering                      //
    //                                           //
    //===========================================//

template<size_t NDIM, typename T>
class MultiGridSolver {
private:

    size_t _N;      // The number of cells per dim in the main grid
    size_t _Ntot;   // Total number of cells in the main grid
    size_t _Nmin;   // The number of cells per dim in the smallest grid
    size_t _Nlevel; // Number of levels

    // All the grids needed
    MultiGrid<NDIM, T> _f;      // The solution
    MultiGrid<NDIM, T> _res;    // The residual
    MultiGrid<NDIM, T> _source; // The multigrid source (restriction of residual)

    // If the source of the equation depends on fields external to the solver they can
    // be added by running add_ext_field and then used in l_operator etc.
    std::vector<MultiGrid<NDIM,T> * > _ext_field;
    
    // Solver parameters
    size_t _ngs_coarse      = 2;     // Number of NGS sweeps on coarse grid
    size_t _ngs_fine        = 2;     // Number of NGS sweeps on the main grid
    size_t _ngridcolours    = 2;     // The order we go through the grid: 
                                           // [Sum_i coord[i] % ngridcolour == j for j = 0,1,..,ngridcolour-1]
    
    // Book-keeping variables
    size_t _tot_sweeps_domain_grid = 0; // Total number of sweeps on the domaingrid (level = 0)

    // Internal methods:
    double calculate_residual(size_t level, Grid<NDIM,T>& res);
    void   prolonge_up_array(size_t to_level, Grid<NDIM,T>& BottomGrid, Grid<NDIM,T>& TopGrid);
    void   make_prolongation_array(Grid<NDIM,T>& f, Grid<NDIM,T>& Rf, Grid<NDIM,T>& df);
    void   GaussSeidelSweep(size_t level, size_t curcolor, T *f);
    void   solve_current_level(size_t level);
    void   recursive_go_up(size_t to_level);
    void   recursive_go_down(size_t from_level);
    void   make_new_source(size_t level);

protected:

    // Turn on verbose while solving
    bool _verbose;

    // Internal methods:
    void get_neighbor_gridindex(std::vector<size_t>& index_list, size_t i, size_t ngrid);

    // Convergence criterion (if the convergence check is not overwritten)
    bool _conv_criterion_residual = true;  // [True]: residual < eps [False]: residual/residual_i < eps
    double _eps_converge          = 1e-5;  // Convergence criterion
    size_t _maxsteps        = 1000;  // Maximum number of V-cycles
    size_t _istep_vcycle = 0;        // The number of V-cycles we are currenlty at

    // Residual information
    double _rms_res;                       // Residual
    double _rms_res_i;                     // The initial residual
    double _rms_res_old;                   // The residual at the old step

public:
    // exit status of solver
    enum class Exit_Status {SUCCESS, FAILURE, SLOW, ITERATE, MAX_STEPS };

    bool _store_all_residual = false;         // Store the residual after every sweep
    std::vector<double> _res_domain_array;    // Array with the residuals after each step
 
    // Constructors
    MultiGridSolver() {}
    MultiGridSolver(size_t N) : MultiGridSolver(N, true) {}
    MultiGridSolver(size_t N, bool verbose) : MultiGridSolver(N, 2, verbose) {}
    MultiGridSolver(size_t N, size_t Nmin, bool verbose);

    size_t get_istep() const { return _istep_vcycle; };

    // Get a pointer to the solution array / grid
    T *get_y(size_t level = 0);
    T const * get_y(size_t level = 0) const;

    Grid<NDIM, T> &get_grid(size_t level = 0){ return _f.get_grid(level); };
    const Grid<NDIM, T> &get_grid(size_t level = 0) const { return _f.get_grid(level); };

    MultiGrid<NDIM, T> &get_mlt_grid(size_t level = 0){ return _f; };
    const MultiGrid<NDIM, T> &get_mlt_grid(size_t level = 0) const { return _f; };

    // Fetch values in externally added fields
    T* get_external_field(size_t level, size_t field) { return _ext_field[field]->get_y(level); };
    T const * get_external_field(size_t level, size_t field) const { return _ext_field[field]->get_y(level); };

    Grid<NDIM, T> &get_external_grid(size_t level, size_t field) { return _ext_field[field]->get_grid(level); };
    const Grid<NDIM, T> &get_external_grid(size_t level, size_t field) const { return _ext_field[field]->get_grid(level); };

    size_t get_external_field_size() const { return _ext_field.size(); };
    
    // Get values of the multigrid-source used to store the restricted residual during the solve step
    T get_multigrid_source(size_t level, size_t i) const { return _source[level][i]; };

    // Set precision parameters
    void set_epsilon(double eps_converge);
    void set_maxsteps(size_t maxsteps);
    void set_ngs_sweeps(size_t ngs_fine, size_t ngs_coarse);
    void set_convergence_criterion_residual(bool use_residual);
    void set_Nlevel(size_t N);
    
    // Fetch info about the grids
    size_t get_N(size_t level = 0) const;
    size_t get_Ntot(size_t level = 0) const;
    size_t get_Nlevel() const;

    // Add a pointer to an external grid if this grid is needed to define the PDE
    void add_external_grid(MultiGrid<NDIM,T> *field);

    // Set the initial guess (uniform or from a grid)
    void set_initial_guess(T  guess);
    void set_initial_guess(T *guess);
    void set_initial_guess(Grid<NDIM,T>& guess);

    // Solve the PDE
    Exit_Status solve();

    // Free up all memory
    void clear();

    // Methods that may be implemented by user
    virtual T upd_operator(const T f, const size_t level, const std::vector<size_t>& index_list, const T h) const;
    virtual T l_operator(const size_t level, const std::vector<size_t>& index_list, bool addsource, const T h) const;
    virtual T dl_operator(const size_t level, const std::vector<size_t>& index_list, const T h) const;
    virtual void correct_sol(Grid<NDIM,T>& f, const Grid<NDIM,T>& corr, const size_t level);
    virtual Exit_Status check_convergence();
    virtual void check_solution(size_t level, Grid<NDIM,T>& sol);
    void check_solution(size_t level); //< automatically retrieves reference to solution
};

#endif
