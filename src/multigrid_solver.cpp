#include "multigrid_solver.h"
#include <iostream>

#define BOOST_LOG_DYN_LINK 1
#include <boost/log/trivial.hpp>

// Simple int-int a^b power-function
inline size_t power(size_t a, size_t b){
  size_t res = 1;
  for(size_t i = 0; i < b; i++) {
    res *= a;
  }
#ifdef _BOUNDSCHECK
  assert( (pow(1.0*a,1.0*b) - double(res)) < 0.5);
#endif
  return res;
}

//================================================
// The L operator ( EOM is written as L(f) = 0 )
//================================================

template<size_t NDIM, typename T>
T MultiGridSolver<NDIM,T>::l_operator(const size_t level, const std::vector<size_t>& index_list, const bool addsource, const T h) const
{
  T l, source, kinetic;
  size_t i = index_list[0];

  // Compute the standard kinetic term [D^2 f]
  kinetic = -2.0 * NDIM * _f[level][ i ];
  for(size_t k = 1; k < 2*NDIM + 1; k++){
    kinetic += _f[level][ index_list[k] ];
  }

  // The right hand side of the PDE
  T *rho = _ext_field[0]->get_y(level);
  source = rho[i];

  // Add the source term arising from restricting the equation down to the lower level
  if( level > 0 && addsource ){
    source += _source[level][i];
  }

  // The full equation
  l = kinetic/(h*h) - source;

  return l;
}

//================================================
// The derivative of the L-operator dL/df
// or more accurately dL_{ijk..} / df_{ijk..}
//================================================

template<size_t NDIM, typename T>
T MultiGridSolver<NDIM,T>::dl_operator(const size_t level, const std::vector<size_t>& index_list, const T h) const
{
  // The derivtive dL/df
  T dl = -2.0*NDIM/(h*h);

  // Sanity check
  if(fabs(dl) < 1e-10){
    BOOST_LOG_TRIVIAL(error) << "Error: dl close to 0" << std::endl;
    exit(1);
  }

  return dl;
}

//================================================
// override for different method of updating,
// default solver use Newton`s method:
// f_new = f_old - L / dL/df
//================================================

template<size_t NDIM, typename T>
T MultiGridSolver<NDIM,T>::upd_operator(const T f, const size_t level, const std::vector<size_t>& index_list, const T h) const
{
    T l  =  l_operator(level, index_list, true, h);
    T dl = dl_operator(level, index_list, h);
    return f - l/dl;
}

//================================================
// The driver routine for solving the PDE
//================================================

template<size_t NDIM, typename T>
typename MultiGridSolver<NDIM,T>::Exit_Status MultiGridSolver<NDIM,T>::solve(){
  // Init some variables
  _istep_vcycle = 0;
  _tot_sweeps_domain_grid = 0;
  _res_domain_array.clear();

  if(_verbose){
    BOOST_LOG_TRIVIAL(debug) << std::endl;
    BOOST_LOG_TRIVIAL(debug) << "===============================================================" << std::endl;
    BOOST_LOG_TRIVIAL(debug) << "==> Starting multigrid solver" << std::endl;
    BOOST_LOG_TRIVIAL(debug) << "===============================================================\n" << std::endl;
  }

  // Pre-solve on domaingrid
  solve_current_level(0);

  // Set the initial residual 
  _rms_res_i = _rms_res;
  _rms_res_old = 0.0;

  // Check if we already have convergence
  Exit_Status status = check_convergence();

  // The V-cycle
  while(status == Exit_Status::ITERATE) {
    ++_istep_vcycle;
    
    if(_verbose){
      BOOST_LOG_TRIVIAL(debug) << std::endl;
      BOOST_LOG_TRIVIAL(debug) << "===============================================================" << std::endl;
      BOOST_LOG_TRIVIAL(debug) << "==> Starting V-cycle istep = " << _istep_vcycle << " Res = " << _rms_res << std::endl;
      BOOST_LOG_TRIVIAL(debug) << "===============================================================\n" << std::endl;
    }

    if (_Nlevel == 1)
        // no V-cycle, solve on domaingrid
        solve_current_level(0);
    else
    {// Go down to the bottom (from finest grid [0] to coarsest grid [_Nlevel-1])
        recursive_go_down(0);

        // Go up to the top
        recursive_go_up(_Nlevel-2);
    }

    // Check for errors in the computation (NaN) and exit if true
    _f.get_grid(0).check_for_nan(true);
    
    // Check for convergence
    status = check_convergence();
  }

  return status;
}

template<size_t NDIM, typename T>
T* MultiGridSolver<NDIM,T>::get_y(size_t level){ 
  return _f.get_y(level);
}

template<size_t NDIM, typename T>
T const * MultiGridSolver<NDIM,T>::get_y(size_t level) const{
    return _f.get_y(level);
} 

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::set_epsilon(double eps_converge){ 
  _eps_converge = eps_converge; 
}

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::add_external_grid(MultiGrid<NDIM,T> *field){ 
  assert(field->get_Nlevel() >= _Nlevel);
  _ext_field.push_back(field); 
}

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::set_maxsteps(size_t maxsteps){ 
  _maxsteps = maxsteps; 
}

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::set_convergence_criterion_residual(bool use_residual){
  _conv_criterion_residual = use_residual;
}

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::set_ngs_sweeps(size_t ngs_fine, size_t ngs_coarse){
  _ngs_fine = ngs_fine; 
  _ngs_coarse = ngs_coarse; 
}

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::set_Nlevel(size_t N){
    // Check that N is divisible by 2^{Nlevel - 1} which is required for the restriction to make sense
    assert( ( _N / power(2, N - 1) ) * power(2, N - 1) == _N); 

    // We should have atleast 1 level
    assert(N >= 1); 

    // set new _Nlevel
    _Nlevel = N;
}

template<size_t NDIM, typename T>
size_t  MultiGridSolver<NDIM,T>::get_N(size_t level) const{ 
  return _f.get_N(level); 
}

template<size_t NDIM, typename T>
size_t  MultiGridSolver<NDIM,T>::get_Ntot(size_t level) const{ 
  return _f.get_Ntot(level); 
}

template<size_t NDIM, typename T>
size_t  MultiGridSolver<NDIM,T>::get_Nlevel() const{ 
  return _Nlevel;
}

template<size_t NDIM, typename T>
MultiGridSolver<NDIM,T>::MultiGridSolver(size_t N, size_t Nmin, bool verbose) :
  _N(N), _Ntot(power(N, NDIM)), _Nmin(Nmin), _Nlevel(int(log2(N / _Nmin) + 1)), _verbose(verbose), 
  _rms_res(0.0), _rms_res_i(0.0), _rms_res_old(0.0) {

    // Check that N is divisible by 2^{Nlevel - 1} which is required for the restriction to make sense
    assert( ( _N / power(2, _Nlevel - 1) ) * power(2, _Nlevel - 1) == _N); 

    // We should have atleast 1 level
    assert(_Nlevel >= 1);                 

    // Allocate memory
    _f      = MultiGrid<NDIM,T>(_N, _Nlevel);
    _source = MultiGrid<NDIM,T>(_N, _Nlevel);
    _res    = MultiGrid<NDIM,T>(_N, _Nlevel);
  }

//================================================
// The initial guess for the solver at the 
// domain level (level = 0)
//================================================

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::set_initial_guess(T guess){
  T *f = _f.get_y(0);
  std::fill_n(f, _Ntot, guess);
}

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::set_initial_guess(T *guess){
  T *f = _f.get_y(0);
  std::copy( &guess[0], &guess[0] + _Ntot, &f[0] );
}

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::set_initial_guess(Grid<NDIM,T>& guessgrid){
  T *f = _f.get_y(0);
  T *guess = guessgrid.get_y();
  std::copy( &guess[0], &guess[0] + _Ntot, &f[0] );
}

//================================================
// Given a cell i = (ix,iy,iz, ...) it computes
// the grid-index of the 2NDIM neighboring cells
// 0: (ix  ,iy  , iz, ...)
// 1: (ix-1,iy  , iz, ...)
// 2: (ix+1,iy  , iz, ...)
// 3: (ix,  iy-1, iz, ...)
// 4: (ix,  iy+1, iz, ...)
// ...
//
// Assuming periodic boundary conditions
//================================================

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::get_neighbor_gridindex(std::vector<size_t>& index_list, size_t i, size_t N){
  index_list = std::vector<size_t>(2*NDIM+1);
  index_list[0] = i;
  for(size_t j = 0, n = 1; j < NDIM; j++, n *= N){
    size_t ii = i/n % N;
    size_t iminus = ii >= 1   ? ii - 1 : N - 1;
    size_t iplus  = ii <= N-2 ? ii + 1 : 0;
    index_list[2*j+1] = i + (iminus - ii) * n;
    index_list[2*j+2] = i + (iplus - ii) * n;
  }
}

//================================================
// Calculates the residual in each cell at
// a given level and stores it in [res]. Returns 
// the rms-residual over the whole grid
//================================================

template<size_t NDIM, typename T>
double MultiGridSolver<NDIM,T>::calculate_residual(size_t level, Grid<NDIM,T> &res){
  size_t N    = get_N(level);
  size_t Ntot = get_Ntot(level);

  // Gridspacing
  const T h = 1.0/T( get_N(level) );

  // Calculate and store (minus) the residual in each cell
#ifdef OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < Ntot; i++) {
    std::vector<size_t> index_list;
    get_neighbor_gridindex(index_list, i, N);
    res[i] = -l_operator(level, index_list, true, h);
  }

  // Calculate and return RMS residual
  return res.rms_norm();
}

//================================================
// Criterion for defining convergence.
// Standard ways are: based on residual or
// the ratio of the residual to the initial
// residual (err).
//================================================

template<size_t NDIM, typename T>
typename MultiGridSolver<NDIM,T>::Exit_Status MultiGridSolver<NDIM,T>::check_convergence(){
  // Compute ratio of residual to initial residual
  double err = _rms_res_i != 0.0 ? _rms_res/_rms_res_i : 1.0;
  Exit_Status converged = Exit_Status::ITERATE;

  // Print out some information
  if(_verbose){
    BOOST_LOG_TRIVIAL(debug) << "    Checking for convergence at step = " << _istep_vcycle << std::endl;
    BOOST_LOG_TRIVIAL(debug) << "        Residual = " << _rms_res << "  Residual_old = " <<  _rms_res_old << std::endl;
    BOOST_LOG_TRIVIAL(debug) << "        Residual_i = " << _rms_res_i << "  Err = " << err << std::endl;
  }

  // Convergence criterion
  if(_conv_criterion_residual) {

    // Convergence criterion based on the residual
    if(_rms_res < _eps_converge){
      if(_verbose || true){
        BOOST_LOG_TRIVIAL(debug) << std::endl;
        BOOST_LOG_TRIVIAL(debug) << "    The solution has converged res = " << _rms_res << " < " << _eps_converge << " istep = " << _istep_vcycle << "\n" << std::endl;
      }
      converged = Exit_Status::SUCCESS;
    } else {
      if(_verbose){
        BOOST_LOG_TRIVIAL(debug) << "    The solution has not yet converged res = " << _rms_res << " !< " << _eps_converge << std::endl; 
      }
    }

  } else {

    // Convergence criterion based on the ratio of the residual
    if(err < _eps_converge){
      if(_verbose || true){
        BOOST_LOG_TRIVIAL(debug) << std::endl;
        BOOST_LOG_TRIVIAL(debug) << "    The solution has converged err = " << err << " < " << _eps_converge << " ( res = " << _rms_res << " ) istep = " << _istep_vcycle << "\n" << std::endl;
      }
      converged = Exit_Status::SUCCESS;
    } else {
      if(_verbose){
       BOOST_LOG_TRIVIAL(debug) << "    The solution has not yet converged err = " << err << " !< " << _eps_converge << std::endl;
      }
    }
  }

  if(_verbose && (_rms_res > _rms_res_old && _istep_vcycle > 1) ){
    BOOST_LOG_TRIVIAL(debug) << "    Warning: Residual_old > Residual" << std::endl;
  }

  // Define converged if istep exceeds maxsteps to avoid infinite loop...
  if(_istep_vcycle >= _maxsteps){
    BOOST_LOG_TRIVIAL(debug) << "    WARNING: MultigridSolver failed to converge! Reached istep = maxsteps = " << _maxsteps << std::endl;
    BOOST_LOG_TRIVIAL(debug) << "    res = " << _rms_res << " res_old = " << _rms_res_old << " res_i = " << _rms_res_i << std::endl;
    converged  = Exit_Status::MAX_STEPS;
  }

  return converged;
}

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::check_solution(size_t, Grid<NDIM,T>&) { }

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::check_solution(size_t level) { check_solution(level, get_grid(level)); }

//================================================
// Prolonge up solution phi from course grid
// to fine grid. Using trilinear prolongation
//================================================

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::prolonge_up_array(size_t to_level, Grid<NDIM,T>& Bottom, Grid<NDIM,T>& Top){
  size_t twotondim = 1 << NDIM;
  size_t NTop      = get_N(to_level);
  size_t NtotTop   = get_Ntot(to_level);
  size_t NBottom   = NTop/2;
  
  // Compute NTop, Ntop^2, ... , Ntop^{Ndim-1} and similar for Nbottom
  std::vector<size_t> nBottomPow(NDIM, 1);
  std::vector<size_t> nTopPow(NDIM, 1);
  for(size_t j = 0, n = 1, m = 1; j < NDIM; j++, n *= NBottom, m *= NTop){
    nBottomPow[j] = n;
    nTopPow[j] = m;
  }

  // Trilinear prolongation
#ifdef OPENMP
#pragma omp parallel for  
#endif
  for (size_t i = 0; i < NtotTop; i++) {
    std::vector<double> fac(NDIM, 0.0);
    std::vector<int> iplus(NDIM, 0);

    double norm = 1.0;
    int iBottom = 0;

    // Compute the shift in index from iBottom to the cells corresponding to ix -> ix+1
    // The fac is the weight for the trilinear interpolation
    for(size_t j = 0; j < NDIM; j++){
      size_t ii = i/nTopPow[j] % NTop;
      size_t iiBottom = ii/2;
      iplus[j] = (iiBottom == NBottom-1 ? 1 - NBottom : 1) * nBottomPow[j];
      fac[j]   = ii % 2 == 0 ? 0.0 : 1.0;
      iBottom += iiBottom * nBottomPow[j];
      norm    *= (1.0 + fac[j]);
    }

    // Compute the sum Top[i] = Sum fac_i             * Top[iBottom + d_i] 
    //                        + Sum fac_i fac_j       * Top[iBottom + d_i + d_j] 
    //                        + Sum fac_i fac_j fac_k * Top[iBottom + d_i + d_j + d_k]
    //                        + ... +
    //                        + fac_1 ... fac_NDIM * Top[iBottom + d_1 + ... + d_NDIM]
    Top[i] = Bottom[iBottom];
    for(size_t k = 1; k < twotondim; k++){
      double termfac = 1.0;
      int iAdd = 0;
      std::bitset<NDIM> bits = std::bitset<NDIM>(k);
      for(size_t j = 0; j < NDIM; j++){
        iAdd = bits[j] * iplus[j];
        termfac *= 1.0 + bits[j] * (fac[j] - 1.0) ;
      }
      Top[i] += T(termfac) * Bottom[iBottom + iAdd];
    }
    Top[i] *= 1.0/T(norm);
  }
}

//================================================
// The Gauss-Seidel Sweeps with standard chess-
// board (first black then white) ordering of 
// gridnodes if _ngridcolours = 2 
//================================================

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::GaussSeidelSweep(size_t level, size_t curcolor, T *f){
  size_t N    = get_N(level);
  size_t Ntot = get_Ntot(level);

  // Gridspacing
  const T h = 1.0/T( get_N(level) );

#ifdef OPENMP
#pragma omp parallel for 
#endif
  for (size_t i = 0; i < Ntot; i++) {
    // Compute cell-color
    size_t color = 0;
    for(size_t j = 0, n = 1; j < NDIM; j++, n *= N){
      color += ( i / n % N );
    }
    color = color % _ngridcolours;

    // Only select cells with right color
    if( color == curcolor ){

      // Update the solution f = f - L / (dL/df)
      std::vector<size_t> index_list;
      get_neighbor_gridindex(index_list, i, N);
      f[i] = upd_operator(f[i], level, index_list, h);
    }
  }
}

//================================================
// Solve the equation on the current level
//================================================

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::solve_current_level(size_t level){
  size_t ngs_sweeps;
 
  if(_verbose)
    BOOST_LOG_TRIVIAL(debug) << "    Performing Newton-Gauss-Seidel sweeps at level " << level << std::endl;

  // Number of sweeps we do
  if(level == 0)
    ngs_sweeps = _ngs_fine;
  else
    ngs_sweeps = _ngs_coarse;

  // Do N Gauss-Seidel Sweeps
  for (size_t i = 0; i < ngs_sweeps; i++) {
    if(level == 0)
      ++_tot_sweeps_domain_grid;

    // Sweep through grid according to sum of coord's mod _ngridcolours
    // Standard is _ngridcolours = 2 -> chess-board ordering
    for(size_t j = 0; j < _ngridcolours; j++)
      GaussSeidelSweep(level, j, _f[level]);

    // Calculate residual and output quite often.
    // For debug, but this is quite useful so keep it for now
    if(_verbose){
      if( (level > 0 && (i == 1 || i == ngs_sweeps-1) ) || (level == 0) ){
        BOOST_LOG_TRIVIAL(debug) << "        level = " << std::setw(5) << level << " NGS Sweep = " << std::setw(5) << i;
        BOOST_LOG_TRIVIAL(debug) << " Residual = " << std::setw(10) << calculate_residual(level, _res.get_grid(level)) << std::endl;
      }
    }

    // Compute and store the residual after every sweep on the domaingrid
    if(level == 0 && _store_all_residual){
      _res_domain_array.push_back( calculate_residual(level, _res.get_grid(level)) );
    }
  }
  if(_verbose) BOOST_LOG_TRIVIAL(debug) << std::endl;

  // Compute the residual
  double curres = calculate_residual(level, _res.get_grid(level));

  // Store domaingrid residual
  if (level == 0){
    _rms_res_old = _rms_res;
    _rms_res = curres;
  }

}

//================================================
// V-cycle go all the way up
//================================================

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::recursive_go_up(size_t to_level){
  size_t from_level = to_level + 1;

  // Restrict down R[f] and store in _res (used as temp-array)
  _f.restrict_down(to_level, _res.get_grid(from_level));
  check_solution(from_level, _res.get_grid(from_level));

  // Make prolongation array ready at from_level
  make_prolongation_array(_f.get_grid(from_level), _res.get_grid(from_level), _res.get_grid(from_level));

  // Prolonge up solution from-level to to-level and store in _res (used as temp array)
  if(_verbose)
    BOOST_LOG_TRIVIAL(debug) << "    Prolonge solution from level: " << to_level+1 << " -> " << to_level << std::endl;
  prolonge_up_array(to_level, _res.get_grid(from_level), _res.get_grid(to_level));

  // Correct solution at to_level (temp array _res contains the correction P[f-R[f]])
  correct_sol(_f.get_grid(to_level), _res.get_grid(to_level), to_level);

  // Calculate new residual
  calculate_residual(to_level, _res.get_grid(to_level));

  // Solve on the level we just went up to
  solve_current_level(to_level);

  // Continue going up
  if(to_level > 0)
    recursive_go_up(to_level-1);
  else {
    return;
  }
}

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::correct_sol(Grid<NDIM,T>& f, const Grid<NDIM,T>& corr, const size_t level)
{
    f += corr;
}

//================================================
// Make the array we are going to prolonge up
// Assumes [Rf] contains the restiction of f
// from the upper level and returns [df]
// containing df = f - R[f]
//================================================

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::make_prolongation_array(Grid<NDIM,T>& f, Grid<NDIM,T>& Rf, Grid<NDIM,T>& df){
  size_t Ntot = f.get_Ntot();
  df = f - Rf;
}

//================================================
// Make new source
//================================================

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::make_new_source(size_t level){
  size_t N    = get_N(level);
  size_t Ntot = get_Ntot(level);

  // Gridspacing
  const T h = 1.0/T( get_N(level) );

  // Calculate the new source
#ifdef OPENMP
#pragma omp parallel for
#endif
  for(size_t i = 0; i < Ntot; i++){
    std::vector<size_t> index_list;
    get_neighbor_gridindex(index_list, i, N);
    T res = l_operator(level, index_list, false, h);
    _source[level][i] = _res[level][i] + res;
  }
}

//================================================
// V-cycle go all the way down
//================================================

template<size_t NDIM, typename T>
void MultiGridSolver<NDIM,T>::recursive_go_down(size_t from_level){
  size_t to_level = from_level + 1;

  // Check if we are at the bottom
  if(to_level >= _Nlevel) {
    if(_verbose) {
      BOOST_LOG_TRIVIAL(debug) << "    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
      BOOST_LOG_TRIVIAL(debug) << "    We have reached the bottom level = " << from_level << " Start going up." << std::endl;
      BOOST_LOG_TRIVIAL(debug) << "    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n" << std::endl;
    }
    return;
  }
  
  if(_verbose)
    BOOST_LOG_TRIVIAL(debug) << "    Going down from level " << from_level << " -> " << to_level << std::endl;

  // Restrict residual and solution
  _res.restrict_down(from_level, _res.get_grid(to_level));
  _f.restrict_down(from_level, _f.get_grid(to_level));
  check_solution(to_level);

  // Make new source
  make_new_source(to_level);

  // Solve on current level
  solve_current_level(to_level);

  // Recursive call
  recursive_go_down(to_level);
}

template<size_t NDIM, typename T>
void  MultiGridSolver<NDIM,T>::clear() {
  _N = _Ntot = _Nlevel = 0;
  _f.clear();
  _res.clear();
  _source.clear();
  _ext_field.clear();
  _res_domain_array.clear();
}

// Explicit template specialisation
template class MultiGridSolver<3,long double>;

template class MultiGridSolver<3,double>;
template class MultiGridSolver<2,double>;
template class MultiGridSolver<1,double>;
template class MultiGridSolver<3,float>;
template class MultiGridSolver<2,float>;
template class MultiGridSolver<1,float>;
template class MultiGridSolver<1,std::complex<double> >;
