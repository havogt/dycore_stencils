#pragma once
#include "../utils.hpp"

struct horizontal_diffusion_reference {

    horizontal_diffusion_reference(repository &repo) : m_repo(repo) {}

//    void generate_reference() {
//        double dtr_stage = dtr_stage_(0, 0, 0);

//        ij_storage_type::storage_info_type storage_info_ij(idim_, jdim_, (uint_t)1);
//        ij_storage_type datacol(storage_info_ij, -1., "datacol");
//        storage_type::storage_info_type storage_info_(idim_, jdim_, kdim_);
//        storage_type ccol(storage_info_, -1., "ccol"), dcol(storage_info_, -1., "dcol");

//        init_field_to_value(ccol, 0.0);
//        init_field_to_value(dcol, 0.0);

//        init_field_to_value(datacol, 0.0);

//        // Generate U
//        forward_sweep(1, 0, ccol, dcol);
//        backward_sweep(ccol, dcol, datacol);
//    }

//    void forward_sweep(int ishift, int jshift, storage_type &ccol, storage_type &dcol) {
//        double dtr_stage = dtr_stage_(0, 0, 0);
//        // k minimum
//        int k = 0;
//        for (int i = halo_size_; i < idim_ - halo_size_; ++i) {
//            for (int j = halo_size_; j < jdim_ - halo_size_; ++j) {
//                double gcv = (double)0.25 * (wcon_(i + ishift, j + jshift, k + 1) + wcon_(i, j, k + 1));
//                double cs = gcv * BET_M;

//                ccol(i, j, k) = gcv * BET_P;
//                double bcol = dtr_stage_(0, 0, 0) - ccol(i, j, k);

//                // update the d column
//                double correctionTerm = -cs * (u_stage_(i, j, k + 1) - u_stage_(i, j, k));
//                dcol(i, j, k) =
//                    dtr_stage * u_pos_(i, j, k) + utens_(i, j, k) + utens_stage_ref_(i, j, k) + correctionTerm;

//                double divided = (double)1.0 / bcol;
//                ccol(i, j, k) = ccol(i, j, k) * divided;
//                dcol(i, j, k) = dcol(i, j, k) * divided;

//                // if(i==3 && j == 3)
//                // std::cout << "AT ref at  " << k << "  " << bcol << "  " << ccol(i,j,k) << " " << dcol(i,j,k) << "
//                // " << gcv <<
//                // "  " << wcon_(i,j,k+1) << "  " << wcon_(i+ishift, j+jshift, k+1) << std::endl;
//            }
//        }

//        // kbody
//        for (k = 1; k < kdim_ - 1; ++k) {
//            for (int i = halo_size_; i < idim_ - halo_size_; ++i) {
//                for (int j = halo_size_; j < jdim_ - halo_size_; ++j) {
//                    double gav = (double)-0.25 * (wcon_(i + ishift, j + jshift, k) + wcon_(i, j, k));
//                    double gcv = (double)0.25 * (wcon_(i + ishift, j + jshift, k + 1) + wcon_(i, j, k + 1));

//                    double as = gav * BET_M;
//                    double cs = gcv * BET_M;

//                    double acol = gav * BET_P;
//                    ccol(i, j, k) = gcv * BET_P;
//                    double bcol = dtr_stage - acol - ccol(i, j, k);

//                    double correctionTerm = -as * (u_stage_(i, j, k - 1) - u_stage_(i, j, k)) -
//                                            cs * (u_stage_(i, j, k + 1) - u_stage_(i, j, k));
//                    dcol(i, j, k) =
//                        dtr_stage * u_pos_(i, j, k) + utens_(i, j, k) + utens_stage_ref_(i, j, k) + correctionTerm;

//                    double divided = (double)1.0 / (bcol - (ccol(i, j, k - 1) * acol));
//                    ccol(i, j, k) = ccol(i, j, k) * divided;
//                    dcol(i, j, k) = (dcol(i, j, k) - (dcol(i, j, k - 1) * acol)) * divided;
//                    // if(i==3 && j == 3)
//                    // std::cout << "FORDW REF at  " << k << "  " << acol << "  " << bcol << "  " << ccol(i,j,k) <<
//                    // " " << dcol(i,j,k) << std::endl;
//                }
//            }
//        }

//        // k maximum
//        k = kdim_ - 1;
//        for (int i = halo_size_; i < idim_ - halo_size_; ++i) {
//            for (int j = halo_size_; j < jdim_ - halo_size_; ++j) {
//                double gav = -(double)0.25 * (wcon_(i + ishift, j + jshift, k) + wcon_(i, j, k));
//                double as = gav * BET_M;

//                double acol = gav * BET_P;
//                double bcol = dtr_stage - acol;

//                // update the d column
//                double correctionTerm = -as * (u_stage_(i, j, k - 1) - u_stage_(i, j, k));
//                dcol(i, j, k) =
//                    dtr_stage * u_pos_(i, j, k) + utens_(i, j, k) + utens_stage_ref_(i, j, k) + correctionTerm;

//                double divided = (double)1.0 / (bcol - (ccol(i, j, k - 1) * acol));
//                dcol(i, j, k) = (dcol(i, j, k) - (dcol(i, j, k - 1) * acol)) * divided;
//            }
//        }
//    }

    void generate_reference() {

//        m_repo.make_field("u_diff_ref");
//        m_repo.make_field("lap_ref");
//        m_repo.make_field("flx_ref");
//        m_repo.make_field("fly_ref");

//        Real *u_in = m_repo.field_h("u_in");
//        Real *u_diff_ref = m_repo.field_h("u_diff_ref");
//        Real *lap = m_repo.field_h("lap");
//        Real *flx = m_repo.field_h("flx");
//        Real *fly = m_repo.field_h("fly");
//        Real *coeff = m_repo.field_h("coeff");

//        IJKSize domain = m_repo.domain();
//        IJKSize halo = m_repo.halo();
//        IJKSize strides;

//        compute_strides(domain, strides);
//        for (unsigned int k = 0; k < domain.m_k; ++k) {
//            for (unsigned int i = halo.m_i - 1; i < domain.m_i - halo.m_i + 1; ++i) {
//                for (unsigned int j = halo.m_j - 1; j < domain.m_j - halo.m_j + 1; ++j) {
//                    lap[index(i, j, k, strides)] =
//                        (Real)4 * u_in[index(i, j, k, strides)] -
//                        (u_in[index(i + 1, j, k, strides)] + u_in[index(i, j + 1, k, strides)] +
//                            u_in[index(i - 1, j, k, strides)] + u_in[index(i, j - 1, k, strides)]);
//                }
//            }
//            for (unsigned int i = halo.m_i - 1; i < domain.m_i - halo.m_i; ++i) {
//                for (unsigned int j = halo.m_j; j < domain.m_j - halo.m_j; ++j) {
//                    flx[index(i, j, k, strides)] = lap[index(i + 1, j, k, strides)] - lap[index(i, j, k, strides)];
//                    if (flx[index(i, j, k, strides)] *
//                            (u_in[index(i + 1, j, k, strides)] - u_in[index(i, j, k, strides)]) >
//                        0)
//                        flx[index(i, j, k, strides)] = 0.;
//                }
//            }
//            for (unsigned int i = halo.m_i; i < domain.m_i - halo.m_i; ++i) {
//                for (unsigned int j = halo.m_j - 1; j < domain.m_j - halo.m_j; ++j) {
//                    fly[index(i, j, k, strides)] = lap[index(i, j + 1, k, strides)] - lap[index(i, j, k, strides)];
//                    if (fly[index(i, j, k, strides)] *
//                            (u_in[index(i, j + 1, k, strides)] - u_in[index(i, j, k, strides)]) >
//                        0)
//                        fly[index(i, j, k, strides)] = 0.;
//                }
//            }
//            for (unsigned int i = halo.m_i; i < domain.m_i - halo.m_i; ++i) {
//                for (unsigned int j = halo.m_j; j < domain.m_j - halo.m_j; ++j) {
//                    u_diff_ref[index(i, j, k, strides)] =
//                        u_in[index(i, j, k, strides)] -
//                        coeff[index(i, j, k, strides)] *
//                            (flx[index(i, j, k, strides)] - flx[index(i - 1, j, k, strides)] +
//                                fly[index(i, j, k, strides)] - fly[index(i, j - 1, k, strides)]);
//                }
//            }
//        }
    }

  private:
    repository m_repo;
};
