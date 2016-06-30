#include "horizontal_diffusion.h"
#include "repository.hpp"

#define BLOCK_X_SIZE 32
#define BLOCK_Y_SIZE 8

#define HALO_BLOCK_X_MINUS 1
#define HALO_BLOCK_X_PLUS 1

#define HALO_BLOCK_Y_MINUS 1
#define HALO_BLOCK_Y_PLUS 1


#define PADDED_BOUNDARY 1

__device__
inline unsigned int index(const unsigned int ipos, const unsigned int jpos, const unsigned int kpos, const IJKSize strides)
{
    return ipos*strides.m_i + jpos*strides.m_j * kpos*strides.m_k;
}

__device__
inline unsigned int cache_index(const unsigned int ipos, const unsigned int jpos) 
{
    return (ipos+PADDED_BOUNDARY) + (jpos+HALO_BLOCK_Y_MINUS)*(BLOCK_X_SIZE+HALO_BLOCK_X_MINUS+HALO_BLOCK_X_PLUS);
}

__global__
void cukernel(Real* in, Real* out, Real* coeff, const IJKSize domain, const IJKSize halo, const IJKSize strides)
{

    unsigned int ipos, jpos;
    int iblock, jblock;
    const unsigned int jboundary_limit = BLOCK_Y_SIZE + HALO_BLOCK_Y_MINUS+HALO_BLOCK_Y_PLUS;
    const unsigned int iminus_limit = jboundary_limit + HALO_BLOCK_X_MINUS;
    const unsigned int iplus_limit = iminus_limit + HALO_BLOCK_X_PLUS;

    if(threadIdx.y < jboundary_limit ) {
        ipos = blockIdx.x * BLOCK_X_SIZE + threadIdx.x + halo.m_i;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.y - HALO_BLOCK_Y_MINUS + halo.m_j;
        iblock = threadIdx.x;
        jblock = threadIdx.y - HALO_BLOCK_Y_MINUS;
    }
    else if( threadIdx.y < iminus_limit)
    {
        ipos = blockIdx.x * BLOCK_X_SIZE - PADDED_BOUNDARY + threadIdx.x % PADDED_BOUNDARY;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.x / PADDED_BOUNDARY - HALO_BLOCK_Y_MINUS;
        iblock = -PADDED_BOUNDARY + (int)threadIdx.x % PADDED_BOUNDARY;
        jblock = threadIdx.x / PADDED_BOUNDARY - HALO_BLOCK_Y_MINUS;
    }
    else if( threadIdx.y < iplus_limit)
    {
        ipos = blockIdx.x * BLOCK_X_SIZE + threadIdx.x % PADDED_BOUNDARY +
              BLOCK_X_SIZE;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.x / PADDED_BOUNDARY - HALO_BLOCK_Y_MINUS;
        iblock = threadIdx.x % PADDED_BOUNDARY + BLOCK_X_SIZE;
        jblock = threadIdx.x / PADDED_BOUNDARY - BLOCK_Y_SIZE;
    }

    __shared__ Real lap[(BLOCK_X_SIZE+HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS)*(BLOCK_Y_SIZE+2)]; 
    __shared__ Real flx[(BLOCK_X_SIZE+1)*(BLOCK_Y_SIZE)];
    __shared__ Real fly[(BLOCK_X_SIZE)*(BLOCK_Y_SIZE+1)];
   
    
    

    out[index(ipos, jpos, 0, strides)] = 1;
//(pow(in[ipos*iStride + jpos*jStride] *in[ipos*iStride + jpos*jStride], 3.5) +
//            pow(in[ipos*iStride + jpos*jStride + kStride], 2.3)
//        );
    for(int kpos=0; kpos < domain.m_k; ++kpos)
    {
        lap[ cache_index(iblock, jblock) ] = (Real)4 * in[index(ipos, jpos, kpos, strides)] - ( in[index(ipos+1, jpos, kpos, strides)] + 
            in[index(ipos-1, jpos, kpos, strides)]+in[index(ipos, jpos+1, kpos, strides)]+in[index(ipos, jpos-1, kpos, strides)]);

/*        out[ipos*iStride + jpos*jStride + k*kStride] = (pow(in[ipos*iStride + jpos*jStride + k*kStride] *in[ipos*iStride + jpos*jStride + k*kStride], 3.5) +
            pow(in[ipos*iStride + jpos*jStride + (k+1)*kStride], 2.3) - 
            pow(in[ipos*iStride + jpos*jStride + (k-1)*kStride], 1.3)
        )
        + out[(ipos+1)*iStride + jpos*jStride + k*kStride] + out[(ipos-1)*iStride + jpos*jStride + k*kStride] + 
        out[ipos*iStride + (jpos+1)*jStride + k*kStride] + out[ipos*iStride + (jpos-1)*jStride + k*kStride]
        ;
*/
      
    }
//    out[ipos*iStride + jpos*jStride + (kSize-1)*kStride] = (pow(in[ipos*iStride + jpos*jStride + (kSize-1)*kStride] *in[ipos*iStride + jpos*jStride + (kSize-1)*kStride], 3.5) -
//            pow(in[ipos*iStride + jpos*jStride + (kSize-2)*kStride], 1.3)
//        );

}

void compute_strides(IJKSize const & domain, IJKSize & strides)
{
    strides.m_i = 1;
    strides.m_j = strides.m_i * domain.m_i;
    strides.m_k = strides.m_j * domain.m_j;
}

void launch_kernel(repository& repo)
{
    IJKSize domain = repo.domain();
    IJKSize halo = repo.halo();

    dim3 threads, blocks;
    threads.x = 32;
    threads.y = 8;
    threads.z = 1;

    blocks.x = domain.m_i / 32;
    blocks.y = domain.m_j / 8;
    blocks.z = 1;
    if(domain.m_i % 32 != 0 || domain.m_j % 8 != 0)
        std::cout << "ERROR: Domain sizes should be multiple of 32x8" << std::endl;

    IJKSize strides;
    compute_strides(domain, strides);

    Real* in = repo.field_d("u_in");
    Real* out = repo.field_d("u_out");
    Real* coeff = repo.field_d("coeff");
   
    cukernel<<<blocks, threads,0>>>(in, out, coeff, domain, halo, strides);
}

