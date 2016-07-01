#include "gtest/gtest.h"
#include "../Options.hpp"
#include "vertical_advection.h"
#include "../repository.hpp"
#include "../verifier.hpp"
#include "vertical_advection_reference.hpp"

int main(int argc, char** argv)
{

    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    if (argc < 4) {
        printf( "Usage: interface1_<whatever> dimx dimy dimz tsteps \n where args are integer sizes of the data fields and tstep is the number of timesteps to run in a benchmark run\n" );
        return 1;
    }

    for(int i=0; i!=3; ++i) {
        Options::getInstance().m_size[i] = atoi(argv[i+1]);
    }

    if (argc > 4) {
        Options::getInstance().m_size[3] = atoi(argv[4]);
    }
    if (argc == 6) {
        if ((std::string(argv[5]) == "-d"))
            Options::getInstance().m_verify = false;
    }
    return RUN_ALL_TESTS();
}

TEST(HorizontalDiffusion, Test)
{
    unsigned int x = Options::getInstance().m_size[0];
    unsigned int y = Options::getInstance().m_size[1];
    unsigned int z = Options::getInstance().m_size[2];
    unsigned int t = Options::getInstance().m_size[3];
    bool verify = Options::getInstance().m_verify;

    if(t==0) t=1;

    IJKSize domain(x, y, z);
    IJKSize halo(2, 2, 0);
    repository repo(domain, halo);

    repo.make_field("utens_stage");
    repo.make_field("utens_stage_ref");
    repo.make_field("u_stage");
    repo.make_field("wcon");
    repo.make_field("u_pos");
    repo.make_field("utens");

    // utens_stage is an input/output field, so the reference needs to be set to same data as utens_stage
    repo.fill_field("utens_stage", 3.2, 2.5, 0.95, 1.18, 18.4, 20.3);
    repo.fill_field("utens_stage_ref", 3.2, 2.5, 0.95, 1.18, 18.4, 20.3);
    repo.fill_field("u_stage", 2.2, 1.5, 0.95, 1.18, 18.4, 20.3);
    repo.fill_field("wcon", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("u_pos", 3.4, 0.7, 1.07, 1.51, 1.4, 2.3);
    repo.fill_field("utens", 7.4, 4.3, 1.17, 0.91, 1.4, 2.3);

    repo.update_device("utens_stage");
    repo.update_device("utens_stage_ref");
    repo.update_device("u_stage");
    repo.update_device("wcon");
    repo.update_device("u_pos");
    repo.update_device("utens");

    repo.make_field("ccol");
    repo.make_field("dcol");
    repo.make_field("datacol");

    repo.init_field("ccol", -1.0);
    repo.init_field("dcol", -1.0);
    repo.init_field("datacol", -1.0);

    repo.update_device("ccol");
    repo.update_device("dcol");
    repo.update_device("datacol");

    launch_kernel(repo);

    vertical_advection_reference ref(repo);
    ref.generate_reference();

    repo.update_host("utens_stage");
    verifier verif(domain, halo, 1e-11);
    ASSERT_TRUE(verif.verify(repo.field_h("utens_stage_ref"), repo.field_h("utens_stage")));
}

