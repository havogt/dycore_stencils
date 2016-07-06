#include "gtest/gtest.h"
#include "../Options.hpp"
//#include "horizontal_diffusion.h"
#include "../repository.hpp"
#include "../verifier.hpp"
#include "fast_waves_uv_reference.hpp"
#include "../timer_cuda.hpp"

int main(int argc, char **argv) {

    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    if (argc < 4) {
        printf("Usage: interface1_<whatever> dimx dimy dimz tsteps \n where args are integer sizes of the data fields "
               "and tstep is the number of timesteps to run in a benchmark run\n");
        return 1;
    }

    for (int i = 0; i != 3; ++i) {
        Options::getInstance().m_size[i] = atoi(argv[i + 1]);
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

TEST(HorizontalDiffusion, Test) {
    unsigned int x = Options::getInstance().m_size[0];
    unsigned int y = Options::getInstance().m_size[1];
    unsigned int z = Options::getInstance().m_size[2];
    unsigned int t = Options::getInstance().m_size[3];
    bool verify = Options::getInstance().m_verify;

    if (t == 0)
        t = 1;

    IJKSize domain(x, y, z);
    IJKSize halo(2, 2, 0);
    repository repo(domain, halo);

    repo.make_field("u_out");
    repo.make_field("v_out");
    repo.make_field("u_pos");
    repo.make_field("v_pos");
    repo.make_field("utens_stage");
    repo.make_field("vtens_stage");
    repo.make_field("ppuv");
    repo.make_field("rho");
    repo.make_field("rho0");
    repo.make_field("p0");
    repo.make_field("hhl");
    repo.make_field("wgtfac");
    repo.make_field("fx");
    repo.make_field("cwp");
    repo.make_field("xdzdx");
    repo.make_field("xdzdy");
    repo.make_field("xlhsx");
    repo.make_field("xlhsy");
    repo.make_field("wbbctens_stage");

    repo.make_field("u_ref");
    repo.make_field("v_ref");

    repo.fill_field("u_ref", 3.0, 2.5, 1.25, 0.78, 18.4, 20.3);
    repo.fill_field("v_ref", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);

    repo.fill_field("u_out", 3.0, 2.5, 1.25, 0.78, 18.4, 20.3);
    repo.fill_field("v_out", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("u_pos", 3.4, 5.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("v_pos", 2.4, 1.3, 0.77, 1.11, 1.4, 2.3);
    repo.fill_field("utens_stage", 4.3, 0.3, 0.97, 1.11, 1.4, 2.3);
    repo.fill_field("vtens_stage", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("ppuv", 1.4, 5.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("rho", 1.4, 4.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("rho0", 3.4, 1.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("p0", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("hhl", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("wgtfac", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("fx", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("cwp", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("xdzdx", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("xdzdy", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("xlhsx", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("xlhsy", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);
    repo.fill_field("wbbctens_stage", 1.4, 0.3, 0.87, 1.11, 1.4, 2.3);

    repo.update_device("u_out");
    repo.update_device("v_out");
    repo.update_device("u_pos");
    repo.update_device("v_pos");
    repo.update_device("utens_stage");
    repo.update_device("vtens_stage");
    repo.update_device("ppuv");
    repo.update_device("rho");
    repo.update_device("rho0");
    repo.update_device("p0");
    repo.update_device("hhl");
    repo.update_device("wgtfac");
    repo.update_device("fx");
    repo.update_device("cwp");
    repo.update_device("xdzdx");
    repo.update_device("xdzdy");
    repo.update_device("xlhsx");
    repo.update_device("xlhsy");
    repo.update_device("wbbctens_stage");

//    launch_kernel(repo, NULL);

    fast_waves_uv_reference ref(repo);
    ref.generate_reference();

    repo.update_host("u_out");
    repo.update_host("v_out");
    verifier verif(domain, halo, 1e-11);
//    ASSERT_TRUE(verif.verify(repo.field_h("u_diff_ref"), repo.field_h("u_out")));

    timer_cuda time("fast_waves_uv");
    for(unsigned int i=0; i < cNumBenchmarkRepetitions; ++i)
    {
//        launch_kernel(repo, &time);
    }

    std::cout << "Time for HORIZONTAL DIFFUSION: " << time.total_time() << std::endl;

}
