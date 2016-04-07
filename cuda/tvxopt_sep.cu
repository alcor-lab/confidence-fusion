// Device code

/*
 * CUDA kernel for computing the primal iteration of the fusion algorithm
 * Updates the values of the variables X, Xhat, and C
 *
 * Author: Valsamis Ntouskos
 * e-mail: ntouskos@diag.uniroma1.it
 * ALCOR Lab, DIAG, Sapienza University of Rome
 */

#include <stdint.h>

#include "decl.h"

template <bool TTGV, bool TADAPTIVE, bool TROF, typename TYPE>
__global__ void TVXopt(TYPE* X, TYPE* Xhat, TYPE* C,
                const TYPE* Err,
                const TYPE* P, const TYPE* Q, const TYPE* G, const TYPE* valK,
                const TYPE* p_opt, const TYPE* p_wght, 
                const TYPE* p_gamma_1, const TYPE p_gamma_2,
                unsigned int M, unsigned int N, unsigned int K)
{      
    unsigned int imsize = M*N;
    
    // compute index
    unsigned int xl = threadIdx.x;
    unsigned int yl = threadIdx.y;
    unsigned int il = (yl+1)*(MEM_TILE_X+1) + xl+1;
    
    unsigned int indx = blockDim.x*blockIdx.x + xl;
    unsigned int indy = blockDim.y*blockIdx.y + yl;
      
    if (indx>=N || indy>=M) {
        return;
    }
    
    unsigned int tind = indy*N + indx;

    TYPE Diff[2] = {0,0}, tmp, sumCD, xprev, vprev;
    TYPE tmperr, parterr, gamma2p, discr;
    
    // load image block to shared memory
    __shared__ TYPE Qxs[(MEM_TILE_X+1)*MEM_TILE_Y];
    __shared__ TYPE Qys[(MEM_TILE_X+1)*MEM_TILE_Y];
    __shared__ TYPE Gs[(MEM_TILE_X+1)*MEM_TILE_Y]; 
   
    __shared__ TYPE Qy2s[(MEM_TILE_X+1)*MEM_TILE_Y]; 
    __shared__ TYPE Qyxs[(MEM_TILE_X+1)*MEM_TILE_Y]; 
    __shared__ TYPE Qx2s[(MEM_TILE_X+1)*MEM_TILE_Y]; 
        
    Gs[il] = G[tind];
    Qys[il] = Q[tind];
    Qxs[il] = Q[tind+imsize];
    if (TTGV) {
        Qy2s[il] = Q[tind+2*imsize];
        Qyxs[il] = Q[tind+3*imsize];
        Qx2s[il] = Q[tind+4*imsize];
    }
    if (xl==0) {
        if (indx==0) {
            Gs[il-1] = 0;
            Qys[il-1] = 0;
            Qxs[il-1] = 0;
            if (TTGV) {
                Qy2s[il-1] = 0;
                Qyxs[il-1] = 0;
                Qx2s[il-1] = 0;
            }
        } else {            
            Gs[il-1] = G[tind-1];
            Qys[il-1] = Q[tind-1];
            Qxs[il-1] = Q[tind-1+imsize];
            if (TTGV) {
                Qy2s[il-1] = Q[tind-1+2*imsize];
                Qyxs[il-1] = Q[tind-1+3*imsize];
                Qx2s[il-1] = Q[tind-1+4*imsize];
            }
        }
    }
    if (yl==0) {
        if (indy==0) {
            Gs[il-MEM_TILE_X-1] = 0;
            Qys[il-MEM_TILE_X-1] = 0;
            Qxs[il-MEM_TILE_X-1] = 0;
            if (TTGV) {
                Qy2s[il-MEM_TILE_X-1] = 0;
                Qyxs[il-MEM_TILE_X-1] = 0;
                Qx2s[il-MEM_TILE_X-1] = 0;
            }
        } else {            
            Gs[il-MEM_TILE_X-1] = G[tind-N];
            Qys[il-MEM_TILE_X-1] = Q[tind-N];
            Qxs[il-MEM_TILE_X-1] = Q[tind-N+imsize];
            if (TTGV) {
                Qy2s[il-MEM_TILE_X-1] = Q[tind-N+2*imsize];
                Qyxs[il-MEM_TILE_X-1] = Q[tind-N+3*imsize];
                Qx2s[il-MEM_TILE_X-1] = Q[tind-N+4*imsize];
            }
        }
    }
    
    __syncthreads();
    
    if (TADAPTIVE) {
        for(uint16_t kk = 0; kk < K; kk++) { 
            unsigned int gm_ind = kk*imsize+tind;
            tmperr = Err[gm_ind];
            if (TROF) {
                parterr = tmperr*tmperr;
            } else {
                parterr = tmperr*P[gm_ind];
            }
            gamma2p = C[gm_ind] - p_opt[3]*(p_gamma_1[gm_ind] + parterr);
            discr = pow(gamma2p,2) + 4*p_opt[3]*(valK[tind]+p_gamma_2-1); 
            C[gm_ind] = 0.5*(gamma2p+sqrt(discr));
        }
    }
    
    // compute gradient of Xhat
    if (indy<(M-1)) {
        Diff[0] = Gs[il-MEM_TILE_X-1]*Qys[il-MEM_TILE_X-1]-Gs[il]*Qys[il];
    } else {
        Diff[0] = Gs[il-MEM_TILE_X-1]*Qys[il-MEM_TILE_X-1];
    }
    if (indx<(N-1)) {
        Diff[1] = Gs[il-1]*Qxs[il-1]-Gs[il]*Qxs[il];
    } else {
        Diff[1] = Gs[il-1]*Qxs[il-1];
    }

    // update X, Xhat
    xprev = X[tind];
    
    sumCD = 0;
    for(uint16_t kk=0; kk<K; kk++) {
        unsigned int gm_ind = tind+imsize*kk;
        unsigned int c_ind;
        if (TADAPTIVE) {
            c_ind = K*imsize+tind;
        } else {
            c_ind = gm_ind;
        }
        
        if (TROF) {
            sumCD += C[c_ind]*Err[gm_ind];
        } else {
            sumCD += C[c_ind]*P[gm_ind];
        }
    }
    tmp = -p_wght[0]*sumCD-(Diff[0]+Diff[1]);
    
    X[tind] += p_opt[0]*tmp;
    Xhat[tind] = X[tind] + p_opt[4]*(X[tind]-xprev);
    // end update X, Xhat
    
   	if (TTGV) {
        // update V
        #pragma unroll
        for(uint8_t kk=0; kk < 2; kk++) {
            Diff[kk] = 0;
        }
        
        if (indy<(M-1)) {
            Diff[0] = Qy2s[il-MEM_TILE_X-1]-Qy2s[il];
            Diff[1] = Qyxs[il-MEM_TILE_X-1]-Qyxs[il];
        } else {
            Diff[0] = Qy2s[il-MEM_TILE_X-1];
            Diff[1] = Qyxs[il-MEM_TILE_X-1];
        }
        if (indx<(N-1)) {
            Diff[0] += Qyxs[il-1]-Qyxs[il];
            Diff[1] += Qx2s[il-1]-Qx2s[il];
        } else {
            Diff[0] += Qyxs[il-1];
            Diff[1] += Qx2s[il-1];
        }
        
        #pragma unroll
        for(uint8_t ii=0; ii < 2; ii++) {
            vprev = X[tind+(ii+1)*imsize];
            tmp = p_wght[1]*Q[tind+ii*imsize]-p_wght[2]*Diff[ii];
            X[tind+(ii+1)*imsize] += p_opt[0]*tmp;
            Xhat[tind+(ii+1)*imsize] = X[tind+(ii+1)*imsize] + p_opt[5]*(X[tind+(ii+1)*imsize]-vprev);
        }
        // end update V
    }

}

template 
__global__ void TVXopt<false,false,false,float>(float* X, float* Xhat, float* C,
                const float* Err, 
                const float* P, const float* Q, const float* G, const float* valK,
                const float* p_opt, const float* p_wght,
                const float* p_gamma_1, const float p_gamma_2, 
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVXopt<false,true,false,float>(float* X, float* Xhat, float* C,
                const float* Err, 
                const float* P, const float* Q, const float* G, const float* valK,
                const float* p_opt, const float* p_wght,
                const float* p_gamma_1, const float p_gamma_2, 
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVXopt<true,false,false,float>(float* X, float* Xhat, float* C,
                const float* Err, 
                const float* P, const float* Q, const float* G, const float* valK,
                const float* p_opt, const float* p_wght,
                const float* p_gamma_1, const float p_gamma_2, 
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVXopt<true,true,false,float>(float* X, float* Xhat, float* C,
                const float* Err, 
                const float* P, const float* Q, const float* G, const float* valK,
                const float* p_opt, const float* p_wght,
                const float* p_gamma_1, const float p_gamma_2, 
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVXopt<false,false,true,float>(float* X, float* Xhat, float* C,
                const float* Err, 
                const float* P, const float* Q, const float* G, const float* valK,
                const float* p_opt, const float* p_wght,
                const float* p_gamma_1, const float p_gamma_2, 
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVXopt<false,true,true,float>(float* X, float* Xhat, float* C,
                const float* Err, 
                const float* P, const float* Q, const float* G, const float* valK,
                const float* p_opt, const float* p_wght,
                const float* p_gamma_1, const float p_gamma_2, 
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVXopt<true,false,true,float>(float* X, float* Xhat, float* C,
                const float* Err, 
                const float* P, const float* Q, const float* G, const float* valK,
                const float* p_opt, const float* p_wght,
                const float* p_gamma_1, const float p_gamma_2, 
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVXopt<true,true,true,float>(float* X, float* Xhat, float* C,
                const float* Err, 
                const float* P, const float* Q, const float* G, const float* valK,
                const float* p_opt, const float* p_wght,
                const float* p_gamma_1, const float p_gamma_2, 
                unsigned int M, unsigned int N, unsigned int K);

// template 
// __global__ void TVXopt<false,false,false,double>(double* X, double* Xhat, double* C,
//                 const double* Err, 
//                 const double* P, const double* Q, const double* G, const double* valK,
//                 const double* p_opt, const double* p_wght,
//                 const double* p_gamma_1, const double p_gamma_2,
//                 unsigned int M, unsigned int N, unsigned int K);
// 
// template 
// __global__ void TVXopt<false,true,false,double>(double* X, double* Xhat, double* C,
//                 const double* Err, 
//                 const double* P, const double* Q, const double* G, const double* valK,
//                 const double* p_opt, const double* p_wght,
//                 const double* p_gamma_1, const double p_gamma_2,
//                 unsigned int M, unsigned int N, unsigned int K);
// 
// template 
// __global__ void TVXopt<true,false,false,double>(double* X, double* Xhat, double* C,
//                 const double* Err, 
//                 const double* P, const double* Q, const double* G, const double* valK,
//                 const double* p_opt, const double* p_wght,
//                 const double* p_gamma_1, const double p_gamma_2,
//                 unsigned int M, unsigned int N, unsigned int K);
// 
// template 
// __global__ void TVXopt<true,true,false,double>(double* X, double* Xhat, double* C,
//                 const double* Err, 
//                 const double* P, const double* Q, const double* G, const double* valK,
//                 const double* p_opt, const double* p_wght,
//                 const double* p_gamma_1, const double p_gamma_2,
//                 unsigned int M, unsigned int N, unsigned int K);
// 
// template 
// __global__ void TVXopt<false,false,true,double>(double* X, double* Xhat, double* C,
//                 const double* Err, 
//                 const double* P, const double* Q, const double* G, const double* valK,
//                 const double* p_opt, const double* p_wght,
//                 const double* p_gamma_1, const double p_gamma_2,
//                 unsigned int M, unsigned int N, unsigned int K);
// 
// template 
// __global__ void TVXopt<false,true,true,double>(double* X, double* Xhat, double* C,
//                 const double* Err, 
//                 const double* P, const double* Q, const double* G, const double* valK,
//                 const double* p_opt, const double* p_wght,
//                 const double* p_gamma_1, const double p_gamma_2, 
//                 unsigned int M, unsigned int N, unsigned int K);
// 
// template 
// __global__ void TVXopt<true,false,true,double>(double* X, double* Xhat, double* C,
//                 const double* Err, 
//                 const double* P, const double* Q, const double* G, const double* valK,
//                 const double* p_opt, const double* p_wght,
//                 const double* p_gamma_1, const double p_gamma_2, 
//                 unsigned int M, unsigned int N, unsigned int K);
// 
// template 
// __global__ void TVXopt<true,true,true,double>(double* X, double* Xhat, double* C,
//                 const double* Err, 
//                 const double* P, const double* Q, const double* G, const double* valK,
//                 const double* p_opt, const double* p_wght,
//                 const double* p_gamma_1, const double p_gamma_2, 
//                 unsigned int M, unsigned int N, unsigned int K);
