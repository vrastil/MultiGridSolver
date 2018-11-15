#ifndef _MULTIGRID_HEADER
#define _MULTIGRID_HEADER
#include <assert.h>
#include <iosfwd>
#include <vector>
#include <climits>
#include <complex>
#include "grid.h"

    //=========================================
    //                                       // 
    // A stack of _Nlevel grids with         //
    // N^NDIM / 2^Level cells in each level  //
    //                                       //
    // Bounds-check for array lookups:       //
    // #define _BOUNDSCHECK                  //
    //                                       //
    //=========================================

template<size_t NDIM, typename T>
class MultiGrid {
  private:

    size_t _N;                        // Number of cells per dim in domain-grid [0]
    size_t _Ntot;                     // Total number of cells in domain-grid [0]
    size_t _Nlevel;                   // Number of levels
    std::vector<size_t> _NinLevel;    // Number of cells per dim in each level
    std::vector<size_t> _NtotinLevel; // Total number of cells in each level
    std::vector<Grid<NDIM,T> > _y;          // The grid data

  public:

    // Constructors
    MultiGrid() {}
    MultiGrid(size_t N):  MultiGrid(N, int(log2(N)+1)) {}
    MultiGrid(size_t N, size_t Nlevel);
    MultiGrid(Grid<NDIM, T> &y, size_t Nlevel);
    MultiGrid(Grid<NDIM, T> &y);
    
    // Fetch a reference to the solution grid at a given level
    Grid<NDIM,T>& get_grid(size_t level = 0);
    const Grid<NDIM,T>& get_grid(size_t level = 0) const;

    // Fetch a pointer to the underlying array at each level
    T* operator[](size_t level);
    const T* operator[](size_t level) const;
    T* get_y(size_t level = 0);
    T const* const get_y(size_t level = 0) const;

    // Fetch the value in the grid at a given level and index
    T get_y(size_t level, size_t i);
    
    // Fetch the value in the grid at a given level and coordinates (ix,iy...)
    T get_y(size_t level, std::vector<size_t>& coord_list);

    // Fetch info about the grid
    size_t get_N(size_t level = 0) const;
    size_t get_Ntot(size_t level = 0) const;
    size_t get_Ndim() const;
    size_t get_Nlevel() const;
    size_t get_Nmin() const;
  
    // Set the value of y at given level and index (save way to define value)
    void set_y(size_t level, size_t i, T value);
    
    // Gridindex from coordinate and vice versa
    size_t gridindex_from_coord(size_t level, std::vector<size_t>& coord_list);
    std::vector<size_t> coord_from_gridindex(size_t level, size_t i);

    // Free up all memory and reset all variables
    void clear();

    // Restrict down a grid 
    void restrict_down(size_t from_level); 
    void restrict_down(size_t from_level, Grid<NDIM, T> &to_grid); 
    void restrict_down_all();
};

#endif
