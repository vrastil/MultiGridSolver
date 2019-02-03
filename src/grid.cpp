#include "grid.h"
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
    
template<size_t NDIM, typename T>
void Grid<NDIM,T>::check_for_nan(bool exitifnan){
  bool nanfound = false;
  for(size_t i = 0; i < _Ntot; i++){
    if(_y[i] != _y[i]){
      nanfound = true;
      break;
    }
  }

  if(nanfound){
    BOOST_LOG_TRIVIAL(error) << "Warning: NaN found in grid" << (exitifnan ? " ...aborting!" : "" ) << std::endl;
    if(exitifnan) exit(1);
  }
}

// Constructor with intial value
template<size_t NDIM, typename T>
Grid<NDIM,T>::Grid(size_t N, T yini) : _N(N), _Ntot(power(_N, NDIM)), _y(std::vector<T>(_Ntot, yini)) {}

// Fetch pointer to grid
template<size_t NDIM, typename T>
T* Grid<NDIM,T>::get_y() { 
  return &_y[0]; 
}

template<size_t NDIM, typename T>
T const * Grid<NDIM,T>::get_y() const { 
  return &_y[0]; 
}

template<size_t NDIM, typename T>
const std::vector<T>& Grid<NDIM,T>::get_vec() const
{
    return _y;
}

// Allow to fetch value using f[i] syntax
template<size_t NDIM, typename T>
T& Grid<NDIM,T>::operator[](size_t i){ 
#ifdef _BOUNDSCHECK
  assert(i < _Ntot);
#endif
  return _y[i]; 
}

template<size_t NDIM, typename T>
const T& Grid<NDIM,T>::operator[](size_t i) const { 
#ifdef _BOUNDSCHECK
  assert(i < _Ntot);
#endif
  return _y[i]; 
}

// Fetch value of grid-cell [i]
template<size_t NDIM, typename T>
T Grid<NDIM,T>::get_y(size_t i) { 
#ifdef _BOUNDSCHECK
  assert(i < _Ntot);
#endif
  return _y[i]; 
}

// Assign whole grid from vector
template<size_t NDIM, typename T>
void Grid<NDIM,T>::set_y(std::vector<T> &y){
#ifdef _BOUNDSCHECK
  assert(y.size() == _Ntot);
#endif
  _y = y;
}

// Assign the gridcell [i] with [value]
template<size_t NDIM, typename T>
void Grid<NDIM,T>::set_y(size_t i, T &value){
#ifdef _BOUNDSCHECK
  assert(i < _Ntot);
#endif
  _y[i] = value;
}

// Compute coordinates given a gridindex
template<size_t NDIM, typename T>
std::vector<size_t> Grid<NDIM,T>::index_list(size_t i){
  std::vector<size_t> ii(NDIM, 0);
  for(size_t j = 0, n = 1; j < NDIM; j++, n *= _N){
    ii[j] = i / n % _N;
  }
  return ii;
}

// Coordinates -> grid-index (index in the 1D _y vector)
template<size_t NDIM, typename T>
size_t Grid<NDIM,T>::grid_index(std::vector<size_t> &index_list){
  size_t index = 0;
  for(size_t j = 0, n = 1; j < NDIM; j++, n *= _N)
    index += index_list[j] * n;
#ifdef _BOUNDSCHECK
  assert(index < _Ntot);
#endif
  return index;
}
    
// Coordinate -> gridindex for 3D grid
template<size_t NDIM, typename T>
size_t Grid<NDIM,T>::grid_index_3d(size_t ix, size_t iy, size_t iz){
  return ix + _N*(iy + _N*iz);
}

// Coordinate -> gridindex for 2D grid
template<size_t NDIM, typename T>
size_t Grid<NDIM,T>::grid_index_2d(size_t ix, size_t iy){
  return ix + _N*iy;
}
// Returns number of cells per dim
template<size_t NDIM, typename T>
size_t Grid<NDIM,T>::get_N() const{
  return _N;
}

// Return total number of cells
template<size_t NDIM, typename T>
size_t Grid<NDIM,T>::get_Ntot() const{
  return _Ntot;
}

// Write a grid to file
template<size_t NDIM, typename T>
void Grid<NDIM,T>::dump_to_file(std::string filename){
  size_t ndim = NDIM;
  
  // Verbose
  BOOST_LOG_TRIVIAL(debug) << "==> Dumping grid to file [" << filename << "]" << std::endl;
  BOOST_LOG_TRIVIAL(debug) << "    Ndim: " << NDIM << " N: " << _N << " Ntot: " << _Ntot << std::endl;

  // Write header
  std::ofstream fout(filename.c_str(), std::ios::out | std::ios::binary);
  fout.write((char*)&_N, sizeof(size_t));
  fout.write((char*)&ndim, sizeof(size_t));

  // Write the grid-data
  fout.write((char*)&_y[0], _y.size() * sizeof(T));
}

// Read a grid from file (assumes specific format)
template<size_t NDIM, typename T>
void Grid<NDIM,T>::read_from_file(std::string filename){
  size_t ninfile = 0, ntot = 0, ndim = 0, size = 0;
  
  // Read header: N and NDIM
  std::ifstream input(filename.c_str(), std::ios::in | std::ifstream::binary);
  
  input.read((char *) &ninfile, sizeof(size_t));
  input.read((char *) &ndim, sizeof(size_t));

  ntot = power(ninfile, NDIM);
  size = sizeof(T) * ntot;
  
  // Checks
  assert(ndim == NDIM);
  assert(ninfile > 0 && ntot < INT_MAX);

  // Verbose
  BOOST_LOG_TRIVIAL(debug) << "==> Reading file into grid [" << filename << "]" << std::endl;
  BOOST_LOG_TRIVIAL(debug) << "    Ndim: " << ndim << " Nfile: " << ninfile << " Ntot: " << ntot << std::endl;

  // Read the data
  std::vector<char> tempvec(size, 0);
  input.read(&tempvec[0], tempvec.size());
 
  // Copy the grid-data and set parameters
  _N = ninfile;
  _Ntot = ntot;
  _y = std::vector<T>(ntot, 0.0);
  std::memcpy(&_y[0], &tempvec[0], size);
}

template<size_t NDIM, typename T>
void Grid<NDIM,T>::clear(){
  _N = _Ntot = 0;
  _y.clear();
}

// Explicit template specialisation
template class Grid<3,long double>;

template class Grid<3,double>;
template class Grid<2,double>;
template class Grid<1,double>;
template class Grid<3,float>;
template class Grid<2,float>;
template class Grid<1,float>;
template class Grid<1,std::complex<double> >;

