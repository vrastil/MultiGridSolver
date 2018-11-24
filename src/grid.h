#ifndef _GRID_HEADER
#define _GRID_HEADER
#include <assert.h>
#include <cstring>  
#include <iosfwd> 
#include <fstream>
#include <complex>
#include <vector>
#include <climits>

    //=========================================
    //                                       // 
    // A simple multidimensional grid-class  //
    //                                       //
    // Bounds-check for array lookups        //
    // #define _BOUNDSCHECK                  //
    //                                       //
    //=========================================

template<size_t NDIM, typename T>
class Grid {

  private:

    size_t _N;      // Number of cells per dim in the grid
    size_t _Ntot;   // Total number of cells in the grid
    std::vector<T> _y;    // The grid data

  public:

    // Constructors
    Grid() : Grid(0, 0.0) {}
    Grid(size_t N) : Grid(N, 0.0) {}
    Grid(size_t N, T yini);

    // Get a pointer to the T-array
    T* get_y();
    T const * get_y() const;
    const std::vector<T>& get_vec() const;

    // Allow syntax grid[i] to get/set the index = i'th element
    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    // Fetch the index = i element in the grid
    T get_y(size_t i);

    // Assign value in the grid
    void set_y(std::vector<T> &y);
    void set_y(size_t i, T &value);

    // Grid-index -> coordinate list [ i = ix1 + N * ix2 + N^2 * ix3 + ... ]
    std::vector<size_t> index_list(size_t i);

    // Get some info about the grid
    size_t get_N() const;
    size_t get_Ntot() const;

    // Convert coordiates -> index in the grid
    size_t grid_index(std::vector<size_t> &index_list);
    size_t grid_index_3d(size_t ix, size_t iy, size_t iz);
    size_t grid_index_2d(size_t ix, size_t iy);

    // Dump a grid to file
    void dump_to_file(std::string filename);

    // Read a grid from file into the object
    void read_from_file(std::string filename);

    // Maximum (in norm)
    double max(){ 
      double maxval = std::norm(_y[0]);
#ifdef OPENMP
#pragma omp parallel for reduction(max: maxval)
#endif
      for(size_t i = 0; i < _Ntot; i++){
        double curval = std::norm(_y[i]);
        if(curval > maxval) maxval = curval;
      }
      return std::sqrt( maxval );
    }
    
    // Maximum (in norm)
    double min(){ 
      double minval = std::norm(_y[0]);
#ifdef OPENMP
#pragma omp parallel for reduction(min: minval)
#endif
      for(size_t i = 0; i < _Ntot; i++){
        double curval = std::norm(_y[i]);
        if(curval < minval) minval = curval;
      }
      return std::sqrt( minval );
    }

    // Free up all memory and reset all variables
    void clear();

    // Operator overloading: add two grids element by element
    template<size_t NNDIM, typename TT>
    Grid<NNDIM,TT>& operator+=(const Grid<NNDIM,TT>& rhs){
#ifdef _BOUNDSCHECK
      assert(this->_N == rhs._N);
#endif
#ifdef OPENMP
#pragma omp parallel for
#endif
      for(size_t i = 0; i < _Ntot; i++)
        this->_y[i] += rhs._y[i];
      return *this;      
    }
    
    // Operator overloading: subtract two grids element by element
    template<size_t NNDIM, typename TT>
    Grid<NNDIM,TT>& operator-=(const Grid<NNDIM,TT>& rhs){
#ifdef _BOUNDSCHECK
      assert(this->_N == rhs._N);
#endif
#ifdef OPENMP
#pragma omp parallel for
#endif
      for(size_t i = 0; i < _Ntot; i++)
        this->_y[i] -= rhs._y[i];
      return *this;      
    }

    // Operator overloading: multiply two grids element by element
    template<size_t NNDIM, typename TT>
    Grid<NNDIM,TT>& operator*=(const Grid<NNDIM,TT>& rhs){
#ifdef _BOUNDSCHECK
      assert(this->_N == rhs._N);
#endif
#ifdef OPENMP
#pragma omp parallel for
#endif
      for(size_t i = 0; i < _Ntot; i++)
        this->_y[i] *= rhs._y[i];
      return *this;      
    }
    
    // Operator overloading: multiply two grids element by element
    template<size_t NNDIM, typename TT>
    Grid<NNDIM,TT>& operator/=(const Grid<NNDIM,TT>& rhs){
#ifdef _BOUNDSCHECK
      assert(this->_N == rhs._N);
#endif
#ifdef OPENMP
#pragma omp parallel for
#endif
      for(size_t i = 0; i < _Ntot; i++)
        this->_y[i] /= rhs._y[i];
      return *this;      
    }
    
    // Operator overloading: multiply every element in grid by scalar
    Grid<NDIM,T>& operator *=(const T & rhs){ 
#ifdef OPENMP
#pragma omp parallel for
#endif
      for(size_t i = 0; i < _Ntot; i++) 
        this->_y[i] *= rhs; 
      return *this;
    } 
    
    // Operator overloading: divide every element in grid by scalar
    Grid<NDIM,T>& operator /=(const T & rhs){
#ifdef OPENMP
#pragma omp parallel for
#endif
      for(size_t i = 0; i < _Ntot; i++) 
        this->_y[i] /= rhs; 
      return *this;
    }

    // The rms-norm, sqrt[ Sum y[i]^2 / Ntot ], of the grid
    double rms_norm();

    // Check for NaN and exit if true
    void check_for_nan(bool exitifnan);
};
 
template<size_t NDIM, typename T>
Grid<NDIM,T> operator+(Grid<NDIM,T> lhs, const Grid<NDIM,T>& rhs){
  lhs += rhs;
  return lhs;
}

template<size_t NDIM, typename T>
Grid<NDIM,T> operator-(Grid<NDIM,T> lhs, const Grid<NDIM,T>& rhs){
  lhs -= rhs;
  return lhs;
}

template<size_t NDIM, typename T>
Grid<NDIM,T> operator*(Grid<NDIM,T> lhs, const Grid<NDIM,T>& rhs){
  lhs *= rhs;
  return lhs;
}

template<size_t NDIM, typename T>
Grid<NDIM,T> operator/(Grid<NDIM,T> lhs, const Grid<NDIM,T>& rhs){
  lhs /= rhs;
  return lhs;
}

template<size_t NDIM, typename T>
Grid<NDIM,T> operator*(Grid<NDIM,T> lhs, const T& rhs){
  lhs *= rhs;
  return lhs;
}

template<size_t NDIM, typename T>
Grid<NDIM,T> operator/(Grid<NDIM,T> lhs, const T& rhs){
  lhs /= rhs;
  return lhs;
}

template<size_t NDIM, typename T>
Grid<NDIM,T> operator+(Grid<NDIM,T> lhs, const T& rhs){
  lhs += rhs;
  return lhs;
}

template<size_t NDIM, typename T>
Grid<NDIM,T> operator-(Grid<NDIM,T> lhs, const T& rhs){
  lhs -= rhs;
  return lhs;
}

template<size_t NDIM, typename T>
Grid<NDIM,T> sqrt(Grid<NDIM,T> lhs){
  for(size_t i = 0; i < lhs.get_Ntot(); i++)
    lhs[i] = sqrt(fabs(lhs[i]));
  return lhs;
}

template<size_t NDIM, typename T>
double Grid<NDIM,T>::rms_norm(){
  double rms = 0.0;
#ifdef OPENMP
#pragma omp parallel for reduction(+:rms)
#endif
  for(size_t i = 0; i < _Ntot; i++){
    rms += std::norm(_y[i]);
  }
  rms = std::sqrt(rms / double(_Ntot));
  return rms;
}

#endif
