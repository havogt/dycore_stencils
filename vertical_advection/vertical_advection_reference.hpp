#pragma once
#include "../utils.hpp"

struct vertical_advection_reference {

    vertical_advection_reference(repository &repo) : m_repo(repo), m_domain(repo.domain()), m_halo(repo.halo()) {
        compute_strides(m_domain, m_strides);
    }

    void backward_sweep(Real* ccol, Real* dcol, Real* datacol) {
        // k maximum
        int k = m_domain.m_k - 1;
        for (int i = m_halo.m_i; i < m_domain.m_i - m_halo.m_i; ++i) {
            for (int j = m_halo.m_j; j < m_domain.m_j - m_halo.m_j; ++j) {
                datacol[index(i, j, k, m_strides)] = dcol[index(i, j, k, m_strides)];
                ccol[index(i, j, k, m_strides)] = datacol[index(i, j, k, m_strides)];
                utens_stage_ref_[index(i, j, k, m_strides)] = DTR_STAGE * (datacol[index(i, j, k, m_strides)] - u_pos_[index(i, j, k, m_strides)]);
            }
        }
        // kbody
        for (k = m_domain.m_k - 2; k >= 0; --k) {
            for (int i = m_halo.m_i; i < m_domain.m_i - m_halo.m_i; ++i) {
                for (int j = m_halo.m_j; j < m_domain.m_j - m_halo.m_j; ++j) {
                    datacol[index(i, j, k, m_strides)] = dcol[index(i, j, k, m_strides)] - (ccol[index(i, j, k, m_strides)] * datacol[index(i, j, k + 1, m_strides)]);
                    ccol[index(i, j, k, m_strides)] = datacol[index(i, j, k, m_strides)];
                    utens_stage_ref_[index(i, j, k, m_strides)] = DTR_STAGE * (datacol[index(i, j, k, m_strides)] - u_pos_[index(i, j, k, m_strides)]);
                }
            }
        }
    }
    void forward_sweep(int ishift, int jshift, Real *ccol, Real *dcol) {

        // k minimum
        int k = 0;
        for (int i = m_halo.m_i; i < m_domain.m_i - m_halo.m_i; ++i) {
            for (int j = m_halo.m_j; j < m_domain.m_j - m_halo.m_j; ++j) {
                Real gcv = (Real)0.25 * (wcon_[index(i + ishift, j + jshift, k + 1, m_strides)] +
                                            wcon_[index(i, j, k + 1, m_strides)]);
                Real cs = gcv * BET_M;

                ccol[index(i, j, k, m_strides)] = gcv * BET_P;
                Real bcol = DTR_STAGE - ccol[index(i, j, k, m_strides)];

                // update the d column
                Real correctionTerm =
                    -cs * (u_stage_[index(i, j, k + 1, m_strides)] - u_stage_[index(i, j, k, m_strides)]);
                dcol[index(i, j, k, m_strides)] = DTR_STAGE * u_pos_[index(i, j, k, m_strides)] +
                                                  utens_[index(i, j, k, m_strides)] +
                                                  utens_stage_ref_[index(i, j, k, m_strides)] + correctionTerm;

                Real divided = (Real)1.0 / bcol;
                ccol[index(i, j, k, m_strides)] = ccol[index(i, j, k, m_strides)] * divided;
                dcol[index(i, j, k, m_strides)] = dcol[index(i, j, k, m_strides)] * divided;
            }
        }

        // kbody
        for (k = 1; k < m_domain.m_k - 1; ++k) {
            for (int i = m_halo.m_i; i < m_domain.m_i - m_halo.m_i; ++i) {
                for (int j = m_halo.m_j; j < m_domain.m_j - m_halo.m_j; ++j) {
                    Real gav = (Real)-0.25 *
                               (wcon_[index(i + ishift, j + jshift, k, m_strides)] + wcon_[index(i, j, k, m_strides)]);
                    Real gcv = (Real)0.25 * (wcon_[index(i + ishift, j + jshift, k + 1, m_strides)] +
                                                wcon_[index(i, j, k + 1, m_strides)]);

                    Real as = gav * BET_M;
                    Real cs = gcv * BET_M;

                    Real acol = gav * BET_P;
                    ccol[index(i, j, k, m_strides)] = gcv * BET_P;
                    Real bcol = DTR_STAGE - acol - ccol[index(i, j, k, m_strides)];

                    Real correctionTerm =
                        -as * (u_stage_[index(i, j, k - 1, m_strides)] - u_stage_[index(i, j, k, m_strides)]) -
                        cs * (u_stage_[index(i, j, k + 1, m_strides)] - u_stage_[index(i, j, k, m_strides)]);
                    dcol[index(i, j, k, m_strides)] = DTR_STAGE * u_pos_[index(i, j, k, m_strides)] +
                                                      utens_[index(i, j, k, m_strides)] +
                                                      utens_stage_ref_[index(i, j, k, m_strides)] + correctionTerm;

                    Real divided = (Real)1.0 / (bcol - (ccol[index(i, j, k - 1, m_strides)] * acol));
                    ccol[index(i, j, k, m_strides)] = ccol[index(i, j, k, m_strides)] * divided;
                    dcol[index(i, j, k, m_strides)] =
                        (dcol[index(i, j, k, m_strides)] - (dcol[index(i, j, k - 1, m_strides)] * acol)) * divided;
                    // if(i==3 && j == 3)
                    // std::cout << "FORDW REF at  " << k << "  " << acol << "  " << bcol << "  " << ccol(i,j,k) <<
                    // " " << dcol(i,j,k) << std::endl;
                }
            }
        }

        // k maximum
        k = m_domain.m_k - 1;
        for (int i = m_halo.m_i; i < m_domain.m_i - m_halo.m_i; ++i) {
            for (int j = m_halo.m_j; j < m_domain.m_j - m_halo.m_j; ++j) {
                Real gav = -(Real)0.25 *
                           (wcon_[index(i + ishift, j + jshift, k, m_strides)] + wcon_[index(i, j, k, m_strides)]);
                Real as = gav * BET_M;

                Real acol = gav * BET_P;
                Real bcol = DTR_STAGE - acol;

                // update the d column
                Real correctionTerm =
                    -as * (u_stage_[index(i, j, k - 1, m_strides)] - u_stage_[index(i, j, k, m_strides)]);
                dcol[index(i, j, k, m_strides)] = DTR_STAGE * u_pos_[index(i, j, k, m_strides)] +
                                                  utens_[index(i, j, k, m_strides)] +
                                                  utens_stage_ref_[index(i, j, k, m_strides)] + correctionTerm;

                Real divided = (Real)1.0 / (bcol - (ccol[index(i, j, k - 1, m_strides)] * acol));
                dcol[index(i, j, k, m_strides)] =
                    (dcol[index(i, j, k, m_strides)] - (dcol[index(i, j, k - 1, m_strides)] * acol)) * divided;
            }
        }
    }

    void generate_reference() {
        m_repo.make_field("acol");
        m_repo.make_field("bcol");
        m_repo.make_field("ccol");
        m_repo.make_field("dcol");
        m_repo.make_field("datacol");

        m_repo.init_field("acol", -1.0);
        m_repo.init_field("bcol", -1.0);
        m_repo.init_field("ccol", -1.0);
        m_repo.init_field("dcol", -1.0);
        m_repo.init_field("datacol", -1.0);

        m_repo.update_device("acol");
        m_repo.update_device("bcol");
        m_repo.update_device("ccol");
        m_repo.update_device("dcol");
        m_repo.update_device("datacol");

        Real *ccol = m_repo.field_h("ccol");
        Real *dcol = m_repo.field_h("dcol");
        Real *datacol = m_repo.field_h("datacol");

        u_stage_ = m_repo.field_h("u_stage");
        wcon_ = m_repo.field_h("wcon");
        u_pos_ = m_repo.field_h("u_pos");
        utens_ = m_repo.field_h("utens");
        utens_stage_ref_ = m_repo.field_h("utens_stage_ref");

        // Generate U
        forward_sweep(1, 0, ccol, dcol);
        backward_sweep(ccol, dcol, datacol);
    }

  private:
    repository m_repo;
    IJKSize m_domain;
    IJKSize m_halo;
    IJKSize m_strides;
    Real *utens_stage_, *u_stage_, *wcon_, *u_pos_, *utens_, *utens_stage_ref_;
};
