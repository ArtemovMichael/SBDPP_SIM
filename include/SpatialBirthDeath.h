#pragma once

/**
 * @file SpatialBirthDeath.h
 * @brief Header for a spatial birth-death point process simulator (refactored).
 *
 * @details
 * This simulator models spatially explicit population dynamics with the ability
 * to spawn and kill individuals in a grid of cells.
 *
 * It supports:
 *  - One or more species
 *  - Configurable birth and competition kernels
 *  - Flexible boundary conditions (periodic/non-periodic)
 *  - Directed inter-species interactions (dd[s1][s2] can differ from dd[s2][s1])
 */

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <chrono>
#include <stdexcept>

/**
 * @brief Helper function for linear interpolation of tabulated data.
 *
 * @param xdat Vector of x values (must be sorted in ascending order)
 * @param ydat Vector of corresponding y values
 * @param point_x Point at which to calculate the interpolated value
 * @return Interpolated function value at point x
 */
double linearInterpolate(const std::vector<double> &xdat, const std::vector<double> &ydat,
                         double point_x);

/**
 * @brief Iterates over all neighbor cell indices within the specified cull range.
 *
 * @tparam DIM  The dimension of the domain (1, 2, or 3).
 * @tparam FUNC A callable like `[](const std::array<int,DIM> &nIdx){ ... }`.
 *
 * @param centerIdx Index of the center cell.
 * @param range The maximum offset in each dimension to search.
 * @param callback Function called for each neighboring cell index.
 */
template <int DIM, typename FUNC>
void forNeighbors(const std::array<int, DIM> &centerIdx, const std::array<int, DIM> &range,
                  const FUNC &callback);

/**
 * @brief Computes Euclidean distance between two points in DIM-dimensional space
 * with optional periodic boundary conditions.
 *
 * @details
 * For each dimension, if periodic=true and the difference exceeds half the domain length,
 * the distance is adjusted using periodic wrapping.
 *
 * @tparam DIM Spatial dimension (1, 2, or 3)
 * @param point1 First point coordinates
 * @param point2 Second point coordinates
 * @param length Domain size in each dimension
 * @param periodic Whether to use periodic boundary conditions
 * @return Euclidean distance between points
 */
template <int DIM>
double distancePeriodic(const std::array<double, DIM> &point1,
                        const std::array<double, DIM> &point2,
                        const std::array<double, DIM> &length, bool periodic);

/**
 * @brief Cell stores multi-species data within a single grid cell.
 *
 * @details
 * For each species s:
 *  - coords[s][i] - coordinates of i-th particle (in real space)
 *  - deathRates[s][i] - death rate for i-th particle
 *  - population[s] - total count of species s in this cell
 *
 * Cached sums:
 *  - cellBirthRateBySpecies[s] - total birth rate for species s
 *  - cellDeathRateBySpecies[s] - total death rate for species s
 *  - cellBirthRate - overall birth rate (sum over all species)
 *  - cellDeathRate - overall death rate (sum over all species)
 *
 * @tparam DIM Spatial dimension (1, 2, or 3)
 */
template <int DIM>
struct Cell {
    // private:
    //     template<int D>
    //     friend class Grid;

    std::vector<std::vector<std::array<double, DIM>>> coords;
    std::vector<std::vector<double>> deathRates;
    std::vector<int> population;
    std::vector<double> cellBirthRateBySpecies;
    std::vector<double> cellDeathRateBySpecies;
    double cellBirthRate{0.0};
    double cellDeathRate{0.0};

    // public:
    Cell() = default;

    /**
     * @brief Initializes data structures for M species
     * @param M Number of species in simulation
     * @throws std::invalid_argument if M <= 0
     */
    void initSpecies(int M) {
        if (M <= 0) {
            throw std::invalid_argument("Number of species must be positive");
        }
        coords.clear();
        deathRates.clear();
        population.clear();

        coords.resize(M);
        deathRates.resize(M);
        population.resize(M, 0);
        cellBirthRateBySpecies.resize(M, 0.0);
        cellDeathRateBySpecies.resize(M, 0.0);
        cellBirthRate = 0.0;
        cellDeathRate = 0.0;
    }
};

/**
 * @brief Main simulation Grid class that partitions domain into cells.
 *
 * @details
 * Physical Parameters:
 * @param area_length[dim] Domain size along dimension dim
 * @param cell_count[dim] Number of cells along dimension dim
 * @param periodic Whether to use periodic boundary conditions
 *
 * Species Parameters:
 * @param M Number of species
 * @param b[s], d[s] Per-species baseline birth/death rates
 * @param dd[s1][s2] Inter-species interaction strength
 * @param birth_x[s], birth_y[s] Birth kernels
 * @param death_x[s1][s2], death_y[s1][s2] Death kernels 
 * @param cutoff[s1][s2] Maximum interaction distance
 * @param cull[s1][s2][dim] Number of neighbor cells to search
 *
 * System State:
 * @param cells Grid of size total_num_cells
 * @param total_population Total particle count
 * @param total_birth_rate Total birth rate
 * @param total_death_rate Total death rate
 *
 * Runtime Parameters:
 * @param rng Random number generator
 * @param time Current simulation time
 * @param event_count Number of processed events
 * @param init_time Simulation start time
 * @param realtime_limit Real time limit (seconds)
 * @param realtime_limit_reached Time limit flag
 *
 * @note Simulation proceeds by repeated make_event() calls, choosing
 *       between birth/death based on rate ratios
 *
 * @tparam DIM Spatial dimension (1, 2 or 3)
 */
template <int DIM>
class Grid {
public:
    std::array<double, DIM> area_length;
    std::array<int, DIM> cell_count;
    bool periodic;

    int M;
    std::vector<double> b;
    std::vector<double> d;
    std::vector<std::vector<double>> dd;
    std::vector<std::vector<double>> birth_x;
    std::vector<std::vector<double>> birth_y;
    std::vector<std::vector<std::vector<double>>> death_x;
    std::vector<std::vector<std::vector<double>>> death_y;
    std::vector<std::vector<double>> cutoff;
    std::vector<std::vector<std::array<int, DIM>>> cull;

    std::vector<Cell<DIM>> cells;
    int total_num_cells;
    double total_birth_rate{0.0};
    double total_death_rate{0.0};
    int total_population{0};
    
    std::mt19937 rng;
    double time{0.0};
    int event_count{0};
    std::chrono::system_clock::time_point init_time;
    double realtime_limit;
    bool realtime_limit_reached{false};

    /**
     * \brief Main constructor.
     *
     * \param M_         Number of species
     * \param areaLen    Domain sizes
     * \param cellCount_ Number of cells in each dimension
     * \param isPeriodic If true, domain wraps
     * \param birthRates b[s]
     * \param deathRates d[s]
     * \param ddMatrix   Flattened MxM dd[s1][s2]
     * \param birthX     birth_x[s]
     * \param birthY     birth_y[s]
     * \param deathX_    death_x[s1][s2]
     * \param deathY_    death_y[s1][s2]
     * \param cutoffs    Flattened MxM cutoff distances
     * \param seed       RNG seed
     * \param rtimeLimit Real-time limit in seconds
     */
    Grid(int M_, const std::array<double, DIM> &areaLen, const std::array<int, DIM> &cellCount_,
         bool isPeriodic, const std::vector<double> &birthRates,
         const std::vector<double> &deathRates, const std::vector<double> &ddMatrix,
         const std::vector<std::vector<double>> &birthX,
         const std::vector<std::vector<double>> &birthY,
         const std::vector<std::vector<std::vector<double>>> &deathX_,
         const std::vector<std::vector<std::vector<double>>> &deathY_,
         const std::vector<double> &cutoffs, int seed, double rtimeLimit);

    // --- Basic utilities for indexing cells ---
    int flattenIdx(const std::array<int, DIM> &idx) const;
    std::array<int, DIM> unflattenIdx(int cellIndex) const;
    int wrapIndex(int i, int dim) const;
    bool inDomain(const std::array<int, DIM> &idx) const;
    Cell<DIM> &cellAt(const std::array<int, DIM> &raw);

    // --- Kernel evaluation helpers ---
    double evalBirthKernel(int s, double x) const;
    double evalDeathKernel(int s1, int s2, double dist) const;

    /**
     * \brief Create a random unit vector in DIM dimensions.
     *        (In 1D, returns either +1 or -1).
     */
    std::array<double, DIM> randomUnitVector(std::mt19937 &rng);

    // ------------------------------------------------------------------
    // Refactored interface: direct spawn/kill
    // ------------------------------------------------------------------

    /**
     * \brief Place a new particle of species s at position inPos (wrapping or discarding
     *        if outside domain and periodic==true or false). Update local and global rates.
     *
     * \param s     Species index
     * \param inPos The desired real-space position
     */
    void spawn_at(int s, const std::array<double, DIM> &inPos);

    /**
     * \brief Remove exactly one particle of species s in cell cIdx with coordinate posKill.
     *        If not found, do nothing. Updates local and global rates.
     *
     * \param s       Species index
     * \param cIdx    The cell index array
     * \param posKill The coordinate to match
     */
    void kill_at(int s, const std::array<int, DIM> &cIdx, int victimIdx);

    /**
     * \brief Removes the interactions contributed by a single particle
     *        (sVictim, victimIdx) in cell cIdx.
     *
     * For each neighbor cell (within cull[sVictim][s2]), we subtract i->j
     * and j->i interactions from occupant j in species s2.
     */
    void removeInteractionsOfParticle(const std::array<int, DIM> &cIdx, int sVictim, int victimIdx);

    /**
     * \brief Loop over a list of positions for each species and call spawn_at.
     *        Useful to initialize a population or add partial subpopulations.
     *
     * \param initCoords initCoords[s] is a vector of positions for species s.
     */
    void placePopulation(const std::vector<std::vector<std::array<double, DIM>>> &initCoords);

    // ------------------------------------------------------------------
    // Random birth/death events
    // ------------------------------------------------------------------

    /**
     * \brief Perform a random spawn event:
     *   1) pick cell by cellBirthRate
     *   2) pick species by cellBirthRateBySpecies
     *   3) pick a random parent occupant
     *   4) sample a radius from the species' birth kernel, pick random direction
     *   5) call spawn_at(...)
     */
    void spawn_random();

    /**
     * \brief Perform a random kill event:
     *   1) pick cell by cellDeathRate
     *   2) pick species by cellDeathRateBySpecies
     *   3) pick a victim occupant by that species' per-particle deathRates
     *   4) call kill_at(...)
     */
    void kill_random();

    // ------------------------------------------------------------------
    // Core simulation loop
    // ------------------------------------------------------------------

    /**
     * \brief Perform one birth or death event, chosen by ratio of total_birth_rate
     *        to total_death_rate, then sample the waiting time exponentially.
     *
     * Does nothing if total_birth_rate + total_death_rate < 1e-12.
     */
    void make_event();

    /**
     * \brief Run a fixed number of events (birth or death).
     *
     * Terminates early if the real-time limit is reached.
     *
     * \param events Number of events to perform.
     */
    void run_events(int events);

    /**
     * \brief Run the simulation until \p time units of simulated time have elapsed.
     *
     * Terminates if real-time limit is reached or if total rates vanish.
     *
     * \param time How much additional simulation time to run.
     */
    void run_for(double time);

    /**
     * \brief Returns aggregated coordinates for all particles for each species.
     *
     * For each species s (0 <= s < M), returns a vector of particle coordinates.
     * The return type is a vector (per species) of std::array<double, DIM>.
     */
    std::vector<std::vector<std::array<double, DIM>>> get_all_particle_coords() const;
};

// Explicit template instantiations
extern template class Grid<1>;
extern template class Grid<2>;
extern template class Grid<3>;

extern template struct Cell<1>;
extern template struct Cell<2>;
extern template struct Cell<3>;
