// Device code

/*
 * CUDA kernel for computing the dual iteration of the fusion algorithm
 * Updates the values of the variables Q, P, and Err
 *
 * Author: Valsamis Ntouskos
 * e-mail: ntouskos@diag.uniroma1.it
 * ALCOR Lab, DIAG, Sapienza University of Rome
 */

#include <stdint.h>

#include "decl.h"

template <bool TTGV, bool TADAPTIVE, bool TROF, typename TYPE>
__global__ void TVQ2opt(TYPE* Q, TYPE* P, TYPE* Err, const TYPE* Xhat, 
                const TYPE* C, const TYPE* G, const TYPE* D,
                const TYPE* p_opt, const TYPE* p_wght,
                const TYPE p_norm, const TYPE p_huber,
                unsigned int M, unsigned int N, unsigned int K)
{    
    unsigned int imsize = M*N;
    
    // compute index
    unsigned int xl = threadIdx.x;
    unsigned int yl = threadIdx.y;
    unsigned int il = yl*(MEM_TILE_X+1) + xl;
    
    unsigned int indx = blockDim.x*blockIdx.x + xl;
    unsigned int indy = blockDim.y*blockIdx.y + yl;
    
    if (indx>=N || indy>=M) {
        return;
    }
            
    unsigned int tind = indy*N + indx;
    
    TYPE Diff[3] = {0,0,0}, norm_temp;
    TYPE tmpr, tgv_res = 0;
    
    // load image block to shared memory
    __shared__ TYPE Xhs[(MEM_TILE_X+1)*MEM_TILE_Y];
    __shared__ TYPE Vhxs[(MEM_TILE_X+1)*MEM_TILE_Y];
    __shared__ TYPE Vhys[(MEM_TILE_X+1)*MEM_TILE_Y];
    
    Xhs[il]  = Xhat[tind];
    if (TTGV) {
        Vhys[il] = Xhat[tind+imsize];
        Vhxs[il] = Xhat[tind+2*imsize];
    }
    if (xl==(blockDim.x-1)) {
        Xhs[il+1]  = Xhat[tind+1];
        if (TTGV) {
            Vhys[il+1] = Xhat[tind+1+imsize];
            Vhxs[il+1] = Xhat[tind+1+2*imsize];
        }
    }
    if (yl==(blockDim.y-1)) {
        Xhs[il+MEM_TILE_X+1]  = Xhat[tind+N];
        if (TTGV) {
            Vhys[il+MEM_TILE_X+1] = Xhat[tind+N+imsize];
            Vhxs[il+MEM_TILE_X+1] = Xhat[tind+N+2*imsize];
        }
    }
    
    __syncthreads();
      
    bool nanflag = isnan(Xhat[tind]);
    
    for(uint16_t kk = 0; kk < K; kk++) { 
        unsigned int gm_ind = kk*imsize+tind;
        if (nanflag||isnan(D[gm_ind])) {
            Err[gm_ind] = 0;
        } else {
            Err[gm_ind] = Xhat[tind]-D[gm_ind];
        }
    }

    // update P
    if (!TROF) {
        for(uint16_t kk = 0; kk < K; kk++) { 
            unsigned int gm_ind = kk*imsize+tind;
            unsigned int c_ind;
            c_ind = gm_ind;

            tmpr = (P[gm_ind] + p_opt[2]*p_wght[0]*C[c_ind]*Err[gm_ind])/
                        (1+p_huber*p_opt[2]*p_wght[0]*C[c_ind]);

            if (tmpr>1 || tmpr<-1) {
                P[gm_ind] = copysignf(1.0f,tmpr);
            } else {
                P[gm_ind] = tmpr;
            }
        }
    }
    // end update P
    
    // compute gradient of Xhat
    if (indy<(M-1)) {
        Diff[0] = Xhs[il+MEM_TILE_X+1]-Xhs[il];
    }
    if (indx<(N-1)) {            
        Diff[1] = Xhs[il+1]-Xhs[il];
    }
    
    #pragma unroll
    for(uint8_t ii=0; ii < 2; ii++) {
        if (isnan(Diff[ii])) {
            Diff[ii] = 0;
        }
    }
    // end grad(Xhat)

   
    
    // update Q
    #pragma unroll
    for(uint8_t ii=0; ii < 2; ii++) {
         if (TTGV) {
            tgv_res = Xhat[tind+(ii+1)*imsize];
        }
        Q[tind+ii*imsize] += p_opt[1]*p_wght[1]*G[tind]*(Diff[ii]-tgv_res);
    }
            
    if (isinf(p_norm)) {
        norm_temp = max(abs(Q[tind]),abs(Q[tind+imsize]));
    } else {
        norm_temp = pow((pow(abs(Q[tind]),p_norm)+
                        pow(abs(Q[tind+imsize]),p_norm)),1/p_norm);
    }
    if (norm_temp>1) {
        Q[tind] /= norm_temp;
        Q[tind+imsize] /=norm_temp;
    }
    // end update Q
    
     if (TTGV) {
        // update R
        #pragma unroll
        for(uint8_t kk=0; kk < 2; kk++) {
            Diff[kk] = 0;
        }
        
        if (indy<(M-1)) {
            Diff[0] = Vhys[il+MEM_TILE_X+1]-Vhys[il];
            Diff[1] = Vhxs[il+MEM_TILE_X+1]-Vhxs[il];
        }
        if (indx<(N-1)) {
            Diff[1] += Vhys[il+1]-Vhys[il];
            Diff[2] = Vhxs[il+1]-Vhxs[il];
        }
        
        #pragma unroll
        for(uint8_t ii=0; ii < 3; ii++) {
            Q[tind+(ii+2)*imsize] += p_opt[1]*p_wght[2]*Diff[ii];  
        }
        
        if (isinf(p_norm)) {
            norm_temp = max(max(Q[tind+2*imsize],Q[tind+3*imsize]),Q[tind+4*imsize]);
        } else {
            norm_temp = pow((pow(abs(Q[tind+2*imsize]),p_norm)+
                              pow(abs(Q[tind+3*imsize]),p_norm)+
                              pow(abs(Q[tind+4*imsize]),p_norm)),1/p_norm);
        }
        if (norm_temp>1) {
            #pragma unroll
            for(int ii=0; ii < 3; ii++) {
                Q[tind+(ii+2)*imsize] /= norm_temp;
            }
        }
        // end update R   
    }
    
}

template 
__global__ void TVQ2opt<false,false,false,float>(float* Q, float* P, float* Err, 
                const float* Xhat, const float* C, const float* G, const float* D,
                const float* p_opt, const float* p_wght,
                const float p_norm, const float p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<false,true,false,float>(float* Q, float* P, float* Err, 
                const float* Xhat, const float* C, const float* G, const float* D,
                const float* p_opt, const float* p_wght,
                const float p_norm, const float p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<true,false,false,float>(float* Q, float* P, float* Err, 
                const float* Xhat, const float* C, const float* G, const float* D,
                const float* p_opt, const float* p_wght,
                const float p_norm, const float p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<true,true,false,float>(float* Q, float* P, float* Err, 
                const float* Xhat, const float* C, const float* G, const float* D,
                const float* p_opt, const float* p_wght,
                const float p_norm, const float p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<false,false,true,float>(float* Q, float* P, float* Err, 
                const float* Xhat, const float* C, const float* G, const float* D,
                const float* p_opt, const float* p_wght,
                const float p_norm, const float p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<false,true,true,float>(float* Q, float* P, float* Err, 
                const float* Xhat, const float* C, const float* G, const float* D,
                const float* p_opt, const float* p_wght,
                const float p_norm, const float p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<true,false,true,float>(float* Q, float* P, float* Err, 
                const float* Xhat, const float* C, const float* G, const float* D,
                const float* p_opt, const float* p_wght,
                const float p_norm, const float p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<true,true,true,float>(float* Q, float* P, float* Err, 
                const float* Xhat, const float* C, const float* G, const float* D,
                const float* p_opt, const float* p_wght,
                const float p_norm, const float p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<false,false,false,double>(double* Q, double* P, double* Err, 
                const double* Xhat, const double* C, const double* G, const double* D,
                const double* p_opt, const double* p_wght,
                const double p_norm, const double p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<false,true,false,double>(double* Q, double* P, double* Err, 
                const double* Xhat, const double* C, const double* G, const double* D,
                const double* p_opt, const double* p_wght,
                const double p_norm, const double p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<true,false,false,double>(double* Q, double* P, double* Err, 
                const double* Xhat, const double* C, const double* G, const double* D,
                const double* p_opt, const double* p_wght,
                const double p_norm, const double p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<true,true,false,double>(double* Q, double* P, double* Err, 
                const double* Xhat, const double* C, const double* G, const double* D,
                const double* p_opt, const double* p_wght,
                const double p_norm, const double p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<false,false,true,double>(double* Q, double* P, double* Err, 
                const double* Xhat, const double* C, const double* G, const double* D,
                const double* p_opt, const double* p_wght,
                const double p_norm, const double p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<false,true,true,double>(double* Q, double* P, double* Err, 
                const double* Xhat, const double* C, const double* G, const double* D,
                const double* p_opt, const double* p_wght,
                const double p_norm, const double p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<true,false,true,double>(double* Q, double* P, double* Err, 
                const double* Xhat, const double* C, const double* G, const double* D,
                const double* p_opt, const double* p_wght,
                const double p_norm, const double p_huber,
                unsigned int M, unsigned int N, unsigned int K);

template 
__global__ void TVQ2opt<true,true,true,double>(double* Q, double* P, double* Err, 
                const double* Xhat, const double* C, const double* G, const double* D,
                const double* p_opt, const double* p_wght,
                const double p_norm, const double p_huber,
                unsigned int M, unsigned int N, unsigned int K);




