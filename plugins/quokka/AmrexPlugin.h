#ifndef AMREXPLUGIN_H
#define AMREXPLUGIN_H

#include "AMReX.H"
#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_iMultiFab.H"
#include "../../TurbGen.h"

class TurbGen :: public TurbGen
{
public: void get_turb_vector_unigrid(const amrex::Box& box, amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &cellSizes, float * return_grid[]) {
        // ******************************************************
        // Compute physical turbulent vector field on a uniform grid, provided
        // start coordinate pos_beg[ndim] and end coordinate pos_end[ndim]
        // of the grid and number of points in grid n[ndim].
        // Return into turbulent vector field into float * return_grid[ndim].
        // Note that index in return_grid[X][index] is looped with x (index i)
        // as the inner loop and with z (index k) as the outer loop.
        // ******************************************************

        // const double pos_beg[], const double pos_end[], const int n[]
        const amrex::Real pos_beg[] = {
                                    box.smallEnd()[0] * cellSizes[0], 
                                    box.smallEnd()[1] * cellSizes[1],
                                    box.smallEnd()[2] * cellSizes[2]
                                 };

        const amrex::Real pos_end[] = { 
                                    box.bigEnd()[0] * cellSizes[0],
                                    box.bigEnd()[1] * cellSizes[1],
                                    box.bigEnd()[2] * cellSizes[2]
                                 };

         const int n[] = {
                            box.length3d()[0],
                            box.length3d()[1],
                            box.length3d()[2]
                         };
        




        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"entering.\n");
        if (verbose > 1) TurbGen_printf("pos_beg = %f %f %f, pos_end = %f %f %f, n = %i %i %i\n",
                pos_beg[X], pos_beg[Y], pos_beg[Z], pos_end[X], pos_end[Y], pos_end[Z], n[X], n[Y], n[Z]);

        // compute output grid cell width (dx, dy, dz)
        double del[3] = {1.0, 1.0, 1.0};
        for (int d = 0; d < (int)ndim; d++) if (n[d] > 1) del[d] = (pos_end[d] - pos_beg[d]) / (n[d]-1);
        // pre-compute amplitude including normalisation factors
        std::vector<double> ampl(nmodes);
        for (int m = 0; m < nmodes; m++) ampl[m] = 2.0 * sol_weight_norm * this->ampl[m];
        // pre-compute grid position geometry, and trigonometry, to speed-up loops over modes below
        std::vector< std::vector<double> > sinxi(n[X], std::vector<double>(nmodes));
        std::vector< std::vector<double> > cosxi(n[X], std::vector<double>(nmodes));
        std::vector< std::vector<double> > sinyj(n[Y], std::vector<double>(nmodes));
        std::vector< std::vector<double> > cosyj(n[Y], std::vector<double>(nmodes));
        std::vector< std::vector<double> > sinzk(n[Z], std::vector<double>(nmodes));
        std::vector< std::vector<double> > coszk(n[Z], std::vector<double>(nmodes));
        for (int m = 0; m < nmodes; m++) {
            for (int i = 0; i < n[X]; i++) {
                sinxi[i][m] = sin(mode[X][m]*(pos_beg[X]+i*del[X]));
                cosxi[i][m] = cos(mode[X][m]*(pos_beg[X]+i*del[X]));
            }
            for (int j = 0; j < n[Y]; j++) {
                if ((int)ndim > 1) {
                    sinyj[j][m] = sin(mode[Y][m]*(pos_beg[Y]+j*del[Y]));
                    cosyj[j][m] = cos(mode[Y][m]*(pos_beg[Y]+j*del[Y]));
                } else {
                    sinyj[j][m] = 0.0;
                    cosyj[j][m] = 1.0;
                }
            }
            for (int k = 0; k < n[Z]; k++) {
                if ((int)ndim > 2) {
                    sinzk[k][m] = sin(mode[Z][m]*(pos_beg[Z]+k*del[Z]));
                    coszk[k][m] = cos(mode[Z][m]*(pos_beg[Z]+k*del[Z]));
                } else {
                    sinzk[k][m] = 0.0;
                    coszk[k][m] = 1.0;
                }
            }
        }
        // scratch variables
        double v[3];
        double real, imag;

        amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			// clear
            v[X] = 0.0; v[Y] = 0.0; v[Z] = 0.0;
            // loop over modes
            for (int m = 0; m < nmodes; m++) {
                // these are the real and imaginary parts, respectively, of
                //  e^{ i \vec{k} \cdot \vec{x} } = cos(kx*x + ky*y + kz*z) + i sin(kx*x + ky*y + kz*z)
                real =  ( cosxi[i][m]*cosyj[j][m] - sinxi[i][m]*sinyj[j][m] ) * coszk[k][m] -
                        ( sinxi[i][m]*cosyj[j][m] + cosxi[i][m]*sinyj[j][m] ) * sinzk[k][m];
                imag =  ( cosyj[j][m]*sinzk[k][m] + sinyj[j][m]*coszk[k][m] ) * cosxi[i][m] +
                        ( cosyj[j][m]*coszk[k][m] - sinyj[j][m]*sinzk[k][m] ) * sinxi[i][m];
                // accumulate total v as sum over modes
                v[X] += ampl[m] * (aka[X][m]*real - akb[X][m]*imag);
                if (ncmp > 1) v[Y] += ampl[m] * (aka[Y][m]*real - akb[Y][m]*imag);
                if (ncmp > 2) v[Z] += ampl[m] * (aka[Z][m]*real - akb[Z][m]*imag);
            }
            // copy into return grid
            long index = k*n[X]*n[Y] + j*n[X] + i;
            return_grid[X][index] = v[X] * ampl_factor[X];
            if (ncmp > 1) return_grid[Y][index] = v[Y] * ampl_factor[Y];
            if (ncmp > 2) return_grid[Z][index] = v[Z] * ampl_factor[Z];
		});

        if (verbose > 1) TurbGen_printf(FuncSig(__func__)+"exiting.\n");
    } // get_turb_vector_unigrid
};

#endif //AMREXPLUGIN_H