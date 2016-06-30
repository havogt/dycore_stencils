#pragma once

struct horizontal_diffusion_reference {

    horizontal_diffusion_reference(repository& repo) : m_repo(repo){}

    void generate_reference()
    {

        m_repo.make_field("u_diff_ref");    
        m_repo.make_field("lap_ref");
        m_repo.make_field("flx_ref");
        m_repo.make_field("fly_ref");

        Real* u_in = m_repo.field_h("u_in");
        Real* u_diff_ref = m_repo.field_h("u_diff_ref"); 
        Real* lap = m_repo.field_h("lap");
        Real* flx = m_repo.field_h("flx");
        Real* fly = m_repo.field_h("fly");

        IJKSize domain = repo.domain();
        IJKSize halo = repo.halo();
        for (unsigned int k = 0; k < domain.m_k; ++k) {
            for (unsigned int i = halo.m_i - 1; i < domain.m_i - halo.m_i + 1; ++i) {
                for (unsigned int j = halo.m_j - 1; j < domain.m_j - halo.m_j + 1; ++j) {
                    
                        lap(i, j, (uint_t)0) =
                            (gridtools::float_type)4 * in_(i, j, k) -
                            (in_(i + 1, j, k) + in_(i, j + 1, k) + in_(i - 1, j, k) + in_(i, j - 1, k));
                    }
                }
                for (uint_t i = halo_size_ - 1; i < idim_ - halo_size_; ++i) {
                    for (uint_t j = halo_size_; j < jdim_ - halo_size_; ++j) {
                        flx(i, j, (uint_t)0) = lap(i + 1, j, (uint_t)0) - lap(i, j, (uint_t)0);
                        if (flx(i, j, (uint_t)0) * (in_(i + 1, j, k) - in_(i, j, k)) > 0)
                            flx(i, j, (uint_t)0) = 0.;
                    }
                }
                for (uint_t i = halo_size_; i < idim_ - halo_size_; ++i) {
                    for (uint_t j = halo_size_ - 1; j < jdim_ - halo_size_; ++j) {
                        fly(i, j, (uint_t)0) = lap(i, j + 1, (uint_t)0) - lap(i, j, (uint_t)0);
                        if (fly(i, j, (uint_t)0) * (in_(i, j + 1, k) - in_(i, j, k)) > 0)
                            fly(i, j, (uint_t)0) = 0.;
                    }
                }
                for (uint_t i = halo_size_; i < idim_ - halo_size_; ++i) {
                    for (uint_t j = halo_size_; j < jdim_ - halo_size_; ++j) {
                        out_ref_(i, j, k) = in_(i, j, k) -
                                            coeff_(i, j, k) * (flx(i, j, (uint_t)0) - flx(i - 1, j, (uint_t)0) +
                                                                  fly(i, j, (uint_t)0) - fly(i, j - 1, (uint_t)0));
                    }
                }
            }
    }

private:
    repository m_repo;

}
