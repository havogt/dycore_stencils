#pragma once
#include "../utils.hpp"
#include "../repository.hpp"

class fast_waves_uv_reference {
public:
    void computePPGradCor(int i, int j, int k, IJKSize& strides, Real* ppgradcor,
        Real* wgtfac, Real* ppuv) {
        ppgradcor[index(i,j,k, strides)] = wgtfac[index(i,j,k, strides)] * ppuv[index(i,j,k, strides)] + ((Real)1.0 - wgtfac[index(i,j,k, strides)]) * ppuv[index(i,j,k-1, strides)];

    }

    fast_waves_uv_reference(repository& repo) : m_repo(repo) {}
    void generate_reference() {

    const Real dt_small = 10;
    const Real edadlat = 1;
    Real* refUField = m_repo.field_h("u_ref");
    Real* refVField = m_repo.field_h("v_ref");

    Real* u_in = m_repo.field_h("u_out");
    Real* v_in = m_repo.field_h("v_out");
    Real* u_pos = m_repo.field_h("u_pos");
    Real* v_pos = m_repo.field_h("v_pos");

    Real* utens_stage = m_repo.field_h("utens_stage");
    Real* vtens_stage = m_repo.field_h("vtens_stage");

    Real* rho = m_repo.field_h("rho");
    Real* ppuv = m_repo.field_h("ppuv");
    Real* fx = m_repo.field_h("fx");

    Real* rho0 = m_repo.field_h("rho0");
    Real* cwp = m_repo.field_h("cwp");
    Real* p0 = m_repo.field_h("p0");
    Real* wbbctens_stage = m_repo.field_h("wbbctens_stage");
    Real* wgtfac = m_repo.field_h("wgtfac");
    Real* hhl = m_repo.field_h("hhl");
    Real* xlhsx = m_repo.field_h("xlhsx");
    Real* xlhsy = m_repo.field_h("xlhsy");
    Real* xdzdx = m_repo.field_h("xdzdx");
    Real* xdzdy = m_repo.field_h("xdzdy");

    m_repo.make_field("xrhsx_ref");
    m_repo.make_field("xrhsy_ref");
    m_repo.make_field("xrhsy_ref");
    m_repo.make_field("xrhsz_ref");

    Real* xrhsy = m_repo.field_h("xrhsy_ref");
    Real* xrhsx = m_repo.field_h("xrhsx_ref");
    Real* xrhsz = m_repo.field_h("xrhsz_ref");


    m_repo.make_field("ppgradcor");
    m_repo.make_field("ppgradu");
    m_repo.make_field("ppgradv");

    Real* ppgradcor = m_repo.field_h("ppgradcor");
    Real* ppgradu = m_repo.field_h("ppgradu");
    Real* ppgradv = m_repo.field_h("ppgradv");


    IJKSize domain = m_repo.domain();
    IJKSize halo = m_repo.halo();
    IJKSize strides;

    compute_strides(domain, halo, strides);

    const int cFlatLimit=10;
    //PPGradCorStage
    int k=cFlatLimit;
    for (int i = 0; i < domain.m_i+1; ++i) {
        for (int j = 0; j < domain.m_j+1; ++j) {
            computePPGradCor(i,j,k,strides, ppgradcor, wgtfac, ppuv);
        }
    }
    for(k=cFlatLimit+1; k < domain.m_k; ++k) {
        for (int i = 0; i < domain.m_i+1; ++i) {
            for (int j = 0; j < domain.m_j+1; ++j) {
                computePPGradCor(i,j,k,strides, ppgradcor, wgtfac, ppuv);
                ppgradcor[index(i,j,k-1,strides)] = (ppgradcor[index(i,j,k,strides)] - ppgradcor[index(i,j,k-1, strides)]);
            }
        }
    }

    // XRHSXStage
    //FullDomain
    k=domain.m_k-1;
    for (int i = -1; i < domain.m_i; ++i) {
        for (int j = 0; j < domain.m_j+1; ++j) {
            xrhsx[index(i,j,k, strides)] = -fx[index(i,j,k, strides)] / ((Real)0.5*(rho[index(i,j,k, strides)] +rho[index(i+1,j,k, strides)])) * (ppuv[index(i+1,j,k, strides)] - ppuv[index(i,j,k, strides)]) +
                    utens_stage[index(i,j,k, strides)];
            xrhsy[index(i,j,k, strides)] = -edadlat / ((Real)0.5*(rho[index(i,j+1,k, strides)] + rho[index(i,j,k, strides)])) * (ppuv[index(i,j+1,k, strides)]-ppuv[index(i,j,k, strides)])
                    +vtens_stage[index(i,j,k, strides)];
        }
    }
    for (int i = 0; i < domain.m_i+1; ++i) {
        for (int j = -1; j < domain.m_j; ++j) {
            xrhsy[index(i,j,k, strides)] = -edadlat / ((Real)0.5*(rho[index(i,j+1,k, strides)] + rho[index(i,j,k,strides)])) * (ppuv[index(i,j+1,k, strides)]-ppuv[index(i,j,k,strides)])
                    +vtens_stage[index(i,j,k, strides)];
        }
    }
    for(int i = 0; i < domain.m_i+1; ++i) {
        for (int j = 0; j < domain.m_j+1; ++j) {
            xrhsz[index(i,j,k,strides)] = rho0[index(i,j,k, strides)] / rho[index(i,j,k, strides)] * 9.8 *
                    ((Real)1.0 - cwp[index(i,j,k, strides)] * (p0[index(i,j,k, strides)] + ppuv[index(i,j,k, strides)])) +
                    wbbctens_stage[index(i,j,k+1, strides)];
        }
    }

    //PPGradStage
    for(k=0; k < domain.m_k-1; ++k) {
        for(int i = 0; i < domain.m_i; ++i) {
            for (int j = 0; j < domain.m_j; ++j) {
                if(k < cFlatLimit) {
                    ppgradu[index(i,j,k, strides)] = (ppuv[index(i+1,j,k, strides)]-ppuv[index(i,j,k, strides)]);
                    ppgradv[index(i,j,k, strides)] = (ppuv[index(i,j+1,k, strides)]-ppuv[index(i,j,k, strides)]);
                }
                else {
                    ppgradu[index(i,j,k, strides)] = (ppuv[index(i+1,j,k, strides)]-ppuv[index(i,j,k, strides)]) + (ppgradcor[index(i+1,j,k, strides)] + ppgradcor[index(i,j,k, strides)])*
                            (Real)0.5 * ( (hhl[index(i,j,k+1, strides)] + hhl[index(i,j,k, strides)]) - (hhl[index(i+1,j,k+1, strides)]+hhl[index(i+1,j,k, strides)])) /
                            ( (hhl[index(i,j,k+1, strides)] - hhl[index(i,j,k, strides)]) + (hhl[index(i+1,j,k+1, strides)] - hhl[index(i+1,j,k, strides)]));
                    ppgradv[index(i,j,k, strides)] = (ppuv[index(i,j+1,k, strides)]-ppuv[index(i,j,k, strides)]) + (ppgradcor[index(i,j+1,k, strides)] + ppgradcor[index(i,j,k, strides)])*
                            (Real)0.5 * ( (hhl[index(i,j,k+1, strides)] + hhl[index(i,j,k, strides)]) - (hhl[index(i,j+1,k+1, strides)]+hhl[index(i,j+1,k, strides)])) /
                            ( (hhl[index(i,j,k+1, strides)] - hhl[index(i,j,k, strides)]) + (hhl[index(i,j+1,k+1, strides)] - hhl[index(i,j+1,k, strides)]));
                }
            }
        }
    }

    //UVStage
    //FullDomain
    for(k=0; k < domain.m_k-1; ++k) {
        for(int i = 0; i < domain.m_i; ++i) {
            for (int j = 0; j < domain.m_j; ++j) {
                Real rhou = fx[index(i,j,k, strides)] / ((Real)0.5*(rho[index(i+1,j,k, strides)] + rho[index(i,j,k, strides)]));
                Real rhov = edadlat / ((Real)0.5*(rho[index(i,j+1,k, strides)] + rho[index(i,j,k, strides)]));

                refUField[index(i,j,k, strides)] = u_pos[index(i,j,k, strides)] + (utens_stage[index(i,j,k, strides)] - ppgradu[index(i,j,k, strides)]*rhou) * dt_small;
                refVField[index(i,j,k, strides)] = v_pos[index(i,j,k, strides)] + (vtens_stage[index(i,j,k, strides)] - ppgradv[index(i,j,k, strides)]*rhov) * dt_small;
            }
        }
    }
    k=domain.m_k-1;
    for(int i = 0; i < domain.m_i; ++i) {
        for (int j = 0; j < domain.m_j; ++j) {
            Real bottU = xlhsx[index(i,j,k, strides)] * xdzdx[index(i,j,k, strides)] * (
                    (Real)0.5*(xrhsz[index(i+1,j,k, strides)]+xrhsz[index(i,j,k, strides)]) -
                    xdzdx[index(i,j,k, strides)] * xrhsx[index(i,j,k, strides)] -
                    (Real)0.5*( (Real)0.5*(xdzdy[index(i+1,j-1,k, strides)]+xdzdy[index(i+1,j,k, strides)]) + (Real)0.5*(xdzdy[index(i,j-1,k, strides)]+xdzdy[index(i,j,k, strides)])) *
                    (Real)0.5*( (Real)0.5*(xrhsy[index(i+1,j-1,k, strides)]+xrhsy[index(i+1,j,k, strides)]) + (Real)0.5*(xrhsy[index(i,j-1,k, strides)]+xrhsy[index(i,j,k, strides)]))
                ) + xrhsx[index(i,j,k, strides)];
            refUField[index(i,j,k, strides)] = u_pos[index(i,j,k, strides)] + bottU * dt_small;
            Real bottV = xlhsy[index(i,j,k, strides)] * xdzdy[index(i,j,k, strides)] * (
                    (Real)0.5*(xrhsz[index(i,j+1,k, strides)]+xrhsz[index(i,j,k, strides)]) -
                    xdzdy[index(i,j,k, strides)] * xrhsy[index(i,j,k, strides)] -
                    (Real)0.5*( (Real)0.5*(xdzdx[index(i-1,j+1,k, strides)]+xdzdx[index(i,j+1,k, strides)]) + (Real)0.5*(xdzdx[index(i-1,j,k, strides)]+xdzdx[index(i,j,k, strides)])) *
                    (Real)0.5*( (Real)0.5*(xrhsx[index(i-1,j+1,k, strides)]+xrhsx[index(i,j+1,k, strides)]) + (Real)0.5*(xrhsx[index(i-1,j,k, strides)]+xrhsx[index(i,j,k, strides)]))
            ) + xrhsy[index(i,j,k, strides)];
            refVField[index(i,j,k, strides)] = v_pos[index(i,j,k, strides)]+bottV*dt_small;
        }
    }

    }
private:

    repository& m_repo;

};

