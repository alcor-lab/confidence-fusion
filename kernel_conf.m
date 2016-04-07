if params.pdhg.sep
    tvq2entpnt_pat = '_Z7TVQ2optILb%dELb%dELb%dEfEvPT2_S1_S1_PKS0_S3_S3_S3_S3_S3_S0_S0_jjj';
    tvxentpnt_pat = '_Z6TVXoptILb%dELb%dELb%dEfEvPT2_S1_S1_PKS0_S3_S3_S3_S3_S3_S3_S3_S0_jjj';

    tvq2signat_pat = 'TYPE* Q, TYPE* P, TYPE* Err, const TYPE* Xhat,const TYPE* C, const TYPE* G, const TYPE* D,const TYPE* p_opt, const TYPE* p_wght,const TYPE p_norm, const TYPE p_huber,unsigned int M, unsigned int N, unsigned int K';
    tvxsignat_pat = 'TYPE* X, TYPE* Xhat, TYPE* C,const TYPE* Err,const TYPE* P, const TYPE* Q, const TYPE* G, const TYPE* valK,const TYPE* p_opt, const TYPE* p_wght,const TYPE* p_gamma_1, const TYPE p_gamma_2,unsigned int M, unsigned int N, unsigned int K';

    tvq2ptxpath = 'cuda/tvq2opt_sep.ptx';
    tvxptxpath = 'cuda/tvxopt_sep.ptx';
else
    tvq2entpnt_pat = '_Z7TVQ2optILb%dELb%dELb%dEfEvPT2_S1_S1_PKS0_S3_S3_S3_S3_S3_S0_S0_jjj';
    tvxentpnt_pat = '_Z6TVXoptILb%dELb%dELb%dEfEvPT2_S1_S1_PKS0_S3_S3_S3_S3_S3_S3_S3_S0_jjj';
    
    tvq2signat_pat = 'TYPE* Q, TYPE* P, TYPE* Err, const TYPE* Xhat,const TYPE* C, const TYPE* G, const TYPE* D,const TYPE* p_opt, const TYPE* p_wght,const TYPE p_norm, const TYPE p_huber,unsigned int M, unsigned int N, unsigned int K';                
    tvxsignat_pat = 'TYPE* X, TYPE* Xhat, TYPE* C,const TYPE* Err,const TYPE* P, const TYPE* Q, const TYPE* G, const TYPE* valK,const TYPE* p_opt, const TYPE* p_wght,const TYPE* p_gamma_1, const TYPE p_gamma_2,unsigned int M, unsigned int N, unsigned int K';
    
    tvq2ptxpath = 'cuda/tvq2opt.ptx';
    tvxptxpath = 'cuda/tvxopt.ptx';
end

