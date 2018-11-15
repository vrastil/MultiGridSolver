#include "multigrid.h"
#include <iostream>

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
Grid<NDIM,T>& MultiGrid<NDIM,T>::get_grid(size_t level){
#ifdef _BOUNDSCHECK
  assert(level < _Nlevel);
#endif
  return _y[level];
}

template<size_t NDIM, typename T>
const Grid<NDIM,T>& MultiGrid<NDIM,T>::get_grid(size_t level) const{
#ifdef _BOUNDSCHECK
  assert(level < _Nlevel);
#endif
  return _y[level];
}

template<size_t NDIM, typename T>
T* MultiGrid<NDIM,T>::operator[](size_t level){ 
#ifdef _BOUNDSCHECK
  assert(level < _Nlevel);
#endif
  return _y[level].get_y(); 
}

template<size_t NDIM, typename T>
const T* MultiGrid<NDIM,T>::operator[](size_t level) const{ 
#ifdef _BOUNDSCHECK
  assert(level < _Nlevel);
#endif
  return _y[level].get_y(); 
}

template<size_t NDIM, typename T>
T* MultiGrid<NDIM,T>::get_y(size_t level){ 
#ifdef _BOUNDSCHECK
  assert(level < _Nlevel);
#endif
  return _y[level].get_y(); 
}

template<size_t NDIM, typename T>
T const * const MultiGrid<NDIM,T>::get_y(size_t level) const{ 
#ifdef _BOUNDSCHECK
  assert(level < _Nlevel);
#endif
  return _y[level].get_y(); 
}

template<size_t NDIM, typename T>
T MultiGrid<NDIM,T>::get_y(size_t level, size_t i){
#ifdef _BOUNDSCHECK
  assert(level < _Nlevel);
#endif
  return _y[level].get_y(i);
}

template<size_t NDIM, typename T>
T MultiGrid<NDIM,T>::get_y(size_t level, std::vector<size_t>& coord_list){
  size_t ind = gridindex_from_coord(level, coord_list);
#ifdef _BOUNDSCHECK
  assert(level < _Nlevel);
#endif
  return _y[level].get_y(ind);
}

template<size_t NDIM, typename T>
void MultiGrid<NDIM,T>::set_y(size_t level, size_t i, T value){
#ifdef _BOUNDSCHECK
  assert(level < _Nlevel);
#endif
  _y[level].set_y(i, value);
}

template<size_t NDIM, typename T>
size_t MultiGrid<NDIM,T>::get_N(size_t level) const{
#ifdef _BOUNDSCHECK
  assert(level < _Nlevel);
#endif
  return _NinLevel[level];
}

template<size_t NDIM, typename T>
size_t MultiGrid<NDIM,T>::get_Ntot(size_t level) const{
#ifdef _BOUNDSCHECK
  assert(level < _Nlevel);
#endif
  return _NtotinLevel[level];
}

template<size_t NDIM, typename T>
size_t MultiGrid<NDIM,T>::get_Ndim() const{ 
  return NDIM; 
}

template<size_t NDIM, typename T>
size_t MultiGrid<NDIM,T>::get_Nlevel() const{ 
  return _Nlevel; 
}

template<size_t NDIM, typename T>
size_t MultiGrid<NDIM,T>::get_Nmin() const{ 
  return _NinLevel[_Nlevel-1]; 
}

template<size_t NDIM, typename T>
MultiGrid<NDIM,T>::MultiGrid(Grid<NDIM, T> &y) : MultiGrid(y.get_N(), int(log2(y.get_N())+1)){
  assert( power(2, _Nlevel - 1 ) == y.get_N()  );
  _y[0] = y; 
}

template<size_t NDIM, typename T>
MultiGrid<NDIM,T>::MultiGrid(Grid<NDIM, T> &y, size_t Nlevel) : MultiGrid(y.get_N(), Nlevel) {
  _y[0] = y; 
}

template<size_t NDIM, typename T>
MultiGrid<NDIM,T>::MultiGrid(size_t N, size_t Nlevel) : _N(N), _Ntot(power(_N, NDIM)), _Nlevel(Nlevel), _NinLevel(std::vector<size_t>(_Nlevel, _N)), _NtotinLevel(std::vector<size_t>(_Nlevel, _Ntot)) {

  // Check that N is positive and divisible by 2^{Nlevel - 1}
  assert( ( _N / power(2, _Nlevel - 1) ) * power(2, _Nlevel - 1) == _N && _N > 0); 

  // We need atleast 1 level
  assert( _Nlevel > 0 ); 

  // Total number of cells in finest level should not be too large (otherwise we cannot use [int])
  assert( log2(INT_MAX) / log2(_N) > NDIM );             

  // Allocate memory
  _y = std::vector<Grid<NDIM, T> >(_Nlevel);
  _y[0] = Grid<NDIM, T> (_N, 0.0);
  for(size_t level = 1; level < _Nlevel; level++){
    _NinLevel[level] = _NinLevel[level-1] / 2;
    _NtotinLevel[level] = power(_NinLevel[level], NDIM);
    assert(_NinLevel[level] > 0);
    _y[level] = Grid<NDIM, T>(_NinLevel[level], 0.0);
  }
}

template<size_t NDIM, typename T>
void MultiGrid<NDIM,T>::restrict_down(size_t from_level, Grid<NDIM,T> &to_grid){

  // Cannot restict down if we are at the bottom
  if(from_level + 1 >= _Nlevel) return;
  
  // Sanity check
  assert( to_grid.get_N() == _y[from_level + 1].get_N() );

  // One over number of cells averaged over  [ = 1 / 2^Ndim ]
  T oneovernumcells = 1.0/T(1 << NDIM);

  // Pointers to Top and Bottom grid
  T *Top = _y[from_level].get_y();
  T *Bottom = to_grid.get_y();//_y[from_level+1].get_y();

  // Nodes on top and bottom level
  size_t NTop = _NinLevel[from_level];
  size_t NtotTop = _NtotinLevel[from_level];
  size_t NBottom = _NinLevel[from_level+1];
  size_t NtotBottom = _NtotinLevel[from_level+1];
  
  // Clear bottom array
  std::fill_n(Bottom, NtotBottom, 0.0);

  // Compute N^j
  std::vector<size_t> NpowTop(NDIM, 1);
  std::vector<size_t> NpowBottom(NDIM, 1);
  for(size_t j = 1; j < NDIM; j++){
    NpowTop[j] = NpowTop[j-1] * NTop;
    NpowBottom[j] = NpowBottom[j-1] * NBottom;
  }

  // Loop over top grid
  for (size_t i = 0; i < NtotTop; i++) {

    // Compute bottom array index the top cell 'belongs to'
    size_t i_bottom = 0;
    for(size_t j = 0; j < NDIM; j++){
      size_t ii = i / NpowTop[j] % NTop; 
      i_bottom += (ii/2) * NpowBottom[j];
    }

    // Add up to restricted grid
    Bottom[i_bottom] += Top[i] * oneovernumcells;
  }

}

template<size_t NDIM, typename T>
void MultiGrid<NDIM,T>::restrict_down(size_t from_level){
  restrict_down(from_level, _y[from_level+1]);
} 

template<size_t NDIM, typename T>
void MultiGrid<NDIM,T>::restrict_down_all(){
  for(size_t i = 0; i < _Nlevel-1; i++)
    restrict_down(i);
}

template<size_t NDIM, typename T>
std::vector<size_t> MultiGrid<NDIM,T>::coord_from_gridindex(size_t level, size_t i){
#ifdef _BOUNDSCHECK
  assert(i < _NtotinLevel[level]);
#endif
  std::vector<size_t> index(NDIM, 0);
  size_t N = _NinLevel[level];
  for(size_t idim = 0, n = 1; idim < NDIM; idim++, n *= N){
    index[idim] = i / n % N;
  }
  return index;
}

template<size_t NDIM, typename T>
size_t MultiGrid<NDIM,T>::gridindex_from_coord(size_t level, std::vector<size_t>& coord_list){
  size_t index = 0;
  for(size_t j = 0, N = 1; j < NDIM; j++, N *= _NinLevel[level])
    index += coord_list[j] * N;
  return index;
}

template<size_t NDIM, typename T>
void  MultiGrid<NDIM,T>::clear(){
  _N = _Ntot = _Nlevel = 0;
  _NinLevel.clear();
  _NtotinLevel.clear();
  _y.clear();
}

// Explicit template specialisation
template class MultiGrid<3,long double>;

template class MultiGrid<3,double>;
template class MultiGrid<2,double>;
template class MultiGrid<1,double>;
template class MultiGrid<3,float>;
template class MultiGrid<2,float>;
template class MultiGrid<1,float>;
template class MultiGrid<1,std::complex<double> >;
