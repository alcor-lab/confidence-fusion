% %%
% Confidence driven TGV fusion function
%
% Signature [dm,meta] = cfusion(dmaps,cm,init_data,I_r,params,options)
%
% Input: 
%   dmaps - depth images to be fused (already registered)
%   cm - a priori confidence maps used for the fusion
%   init_data - initial depth values
%   I_r - image corresponding to reference view (necessary to
%                   compute appearance based confidence matrix G)
%   params - parameters of the fusion algorithm (see get_fusion_options
%               function)
%   params - fusion algorithm options (see get_fusion_options
%               function)
% Ouput:
%   dm - fused depth image
%   meta - meta data (time,iterations,variable errors)

% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome

function [dm,meta] = cfusion(dmaps,cm,init_data,I_r,params,options)

if nargin < 6
    [options,paramsd] = get_fusion_options();
end
if nargin < 5
    params = paramsd;
end
if nargin < 4
    I_r = [];
end

num_dm = length(dmaps);
if num_dm < 1
    throw(MException('FUSION:INSUFDATA','Error: Insufficient number of images.'))
end
[nr,nc] = size(dmaps{1});

% %% 
% Auxilary variables

% Regularization parameters
lambda = params.lambda; 

tau_in = params.tau;
tau = tau_in;
zeta = params.zeta;
ord = params.ord;

tau_l= params.adapt.tau;
alpha= params.adapt.alpha;
beta = params.adapt.beta;

eta = params.eta;

zeta_v = params.tgv.zeta;
alpha_1 = params.tgv.alpha_1;
alpha_0 = params.tgv.alpha_0;


% %% Prepare the data (vectorization)
ref_ind = ceil(num_dm/2);
Dref = zeros(nr*nc,num_dm);
Cref = ones(nr*nc,num_dm);
for kk = 1:num_dm
    if ~options.uniform
        Cref(:,kk) = cm{kk}(:)';
    end
    Dref(:,kk) = dmaps{kk}(:)';
end
Cref=[Cref,ones(nr*nc,1)];

% replace nans with zeros
Cref(isnan(Cref)) = 0;

% %% Prepare video capture
if options.capture
    wr_vid(1) = VideoWriter(sprintf('./merging_TV_%s.avi',datestr(now,30)));
    wr_vid(1).('FrameRate') = 15;
    wr_vid(1).('Quality') = 100;
    open(wr_vid(1));
    
    wr_vid(2) = VideoWriter(sprintf('./conf_TV_%s.avi',datestr(now,30)));
    wr_vid(2).('FrameRate') = 15;
    wr_vid(2).('Quality') = 100;
    open(wr_vid(2));
end

% %% initialize variables
p = zeros(nr*nc,num_dm);
a = init_data(:);
val_pts = (~isnan(a)&a~=0);
% val_ptst = (~isnan(init_data)&(init_data~=0));
val_a = a(val_pts);
a(~val_pts) = max(val_a);
q = zeros(2*nr*nc,1);


valK = double(sum(~isnan(Dref),2));
if isfield(params,'aniso') && ~isempty(I_r) 
    if size(I_r,3)>1
        I_r = rgb2gray(I_r);
    end
    [rows,cols] = ind2sub(size(I_r), 1:nr*nc);
    u = [cols; rows];

    % Compute weights
    I1=imfilter(I_r,params.aniso.gfilt); 
    ws = g_u(I1,u,params.aniso.Gw,params.aniso.Gexp);
else
    ws = ones(1,nr*nc);
end

% The matrix G stores the weights
% G = make_G(ws');
G = repmat(ws',2,1);

% %%
% Show the initial estimation
if options.debug_lvl > 1
    % define shared variables
%     conf_fig=[]; post_reg_fig=[]; hImage=[];
    conf_fig = figure; set(conf_fig, 'Name', 'Confidence');
    if any(any(ws~=1))
        weights_fig = figure; set(weights_fig, 'Name', 'WEIGHTS');
        imagesc(reshape(ws, [nr,nc])); colorbar; axis image; axis off;
    end
    post_reg_fig = figure; set(post_reg_fig, 'Name', 'Depth Map AFTER regularization');
    set(post_reg_fig,'Position',[1000,630,640,480]);

    ad = a;
    ad(~val_pts) = nan;
    imaft = disp_to_color(options.vis.max./reshape(ad,[nr,nc]),255);
    hImage = imshow(imaft);
    drawnow;
end
if options.capture   
    F_v = disp_to_color(options.vis.max./reshape(a,[nr,nc]),255);
    F.cdata = F_v;
    F.colormap = [];
    writeVideo(wr_vid(1),F);

    if options.adapt
        Cref_d = Cref(:,end);
    else
        Cref_d = Cref(:,ref_ind);
    end
    Cref_d(~val_pts) = nan;
    b_d = log(3)*1./Cref_d;
    im_or = reshape(b_d,[nr,nc]);
    im_eq = im_or;
    im_eq(im_or>1.0)=1.0;

    map_sz = 256;
    im_ind = rgb2ind(repmat(im_eq,[1,1,3]),gray(map_sz),'nodither');
    im_ind(~val_pts)=map_sz;

    F.cdata = im_ind;
    F.colormap = jet(map_sz);
    writeVideo(wr_vid(2),F);
end


% %% 
% Regularization loop
n = 0;
if options.debug_lvl > 0 
    fprintf('Starting regularization process...\n');
end
ticreg = tic;

ah = a;
ap = -1*ones(size(a));
difa = abs(a-ap);
n_data = sum(sum(~isnan(difa)));

if options.tgv
    sigma = min([1/(4*tau*alpha_0.^2),1/(2*(num_dm+1)*tau*8*alpha_0.^2),1/(2*tau*64*alpha_1.^2)]);
%     sigma = 1/((num_dm+1)*tau*8);
else
    sigma = 1/((num_dm+1)*tau*8);
end
if options.adapt
    sigma_p = 1/((num_dm+1)*tau*max(max(Cref(:,end))).^2);
else
    sigma_p = 1/((num_dm+1)*tau*max(max(Cref(:,1:end-1))).^2);
end

dtype = options.cuda.dtype;
if strcmpi(dtype,'double')
    mtype = dtype;
elseif strcmpi(dtype,'float')
    mtype = 'single';
end

run('kernel_conf');
if ~strcmpi(options.alg,'PDHG')
    tvq2entpnt = strrep(sprintf(tvq2entpnt_pat,options.tgv,false,options.rof),'f',dtype(1));
    tvxentpnt = strrep(sprintf(tvxentpnt_pat,options.tgv,false,options.rof),'f',dtype(1));
else
    tvq2entpnt = strrep(sprintf(tvq2entpnt_pat,options.tgv,options.adapt,options.rof),'f',dtype(1));
    tvxentpnt = strrep(sprintf(tvxentpnt_pat,options.tgv,options.adapt,options.rof),'f',dtype(1));
end
tvq2signat = strrep(tvq2signat_pat,'TYPE',dtype);
tvxsignat = strrep(tvxsignat_pat,'TYPE',dtype);

bxd = params.cuda.bxd;
byd = params.cuda.byd;
funcfullpath = mfilename('fullpath');
funcpath = fileparts(funcfullpath);

tvq2kern = parallel.gpu.CUDAKernel(fullfile(funcpath,tvq2ptxpath), tvq2signat, tvq2entpnt); 
tvq2kern.ThreadBlockSize=[bxd-1,byd-1,1]; 
tvq2kern.GridSize=[ceil(nc/(bxd-1)),ceil(nr/(byd-1)),1];

tvxkern = parallel.gpu.CUDAKernel(fullfile(funcpath,tvxptxpath), tvxsignat, tvxentpnt); 
tvxkern.ThreadBlockSize=[bxd-1,byd-1,1]; 
tvxkern.GridSize=[ceil(nc/(bxd-1)),ceil(nr/(byd-1)),1];

Dnaux = reshape(Dref,[nr,nc,num_dm]);
Dnaux = permute(Dnaux,[1,3,2]);
Dnaux = reshape(Dnaux,[nr*num_dm,nc]);
d_g = gpuArray(eval([mtype,'(Dnaux'')']));

Cnaux = reshape(Cref,[nr,nc,num_dm+1]);
Cnaux = permute(Cnaux,[1,3,2]);
Cnaux = reshape(Cnaux,[nr*(num_dm+1),nc]);
c_g = gpuArray(eval([mtype,'(Cnaux'')']));

vKaux = reshape(valK,[nr,nc,1]);
vKaux = permute(vKaux,[1,3,2]);
vKaux = reshape(vKaux,[nr,nc]);
vk_g = gpuArray(eval([mtype,'(vKaux'')']));

err_g = gpuArray.zeros(nc,nr*num_dm,mtype);

xh_g = gpuArray(eval([mtype,'(reshape(ah,[nr,nc])'')']));
x_g = gpuArray(eval([mtype,'(reshape(a,[nr,nc])'')']));

q_g = gpuArray.zeros(nc,nr*2,mtype);
p_g = gpuArray.zeros(nc,nr*num_dm,mtype);

Gaux = reshape(G(1:nr*nc),[nr,nc,1]);
Gaux = permute(Gaux,[1,3,2]);
Gaux = reshape(Gaux,[nr*1,nc]);
g_g = gpuArray(eval([mtype,'(Gaux'')']));

p_opt_g = gpuArray(eval([mtype,'([tau,sigma,sigma_p,tau_l,zeta,zeta_v])']));
p_norm_g = gpuArray(eval([mtype,'(ord)']));
p_wght_g = gpuArray(eval([mtype,'([lambda,alpha_0,alpha_1])']));
p_hub_g = gpuArray(eval([mtype,'(eta)']));

p_gam_g = 2.*beta.*c_g;

deb_g = gpuArray(eval([mtype,'(1)']));

p_gam_2 =  gpuArray(eval([mtype,'([alpha])']));
if options.tgv
    r_g = gpuArray.zeros(nc,nr*3,mtype);
    v_g = gpuArray.zeros(nc,nr*2,mtype);
    vh_g = gpuArray.zeros(nc,nr*2,mtype);

    q_g = [q_g,r_g];
    x_g = [x_g,v_g];
    xh_g = [xh_g,vh_g];
end

% val_dg = ~isnan(d_g);
% val_xg = ~isnan(x_g);
out_converge = false;
errL = inf;
ff_g = 1;
c_adapt_ind = num_dm*nr+(1:nr);
% err_b = inf;
% val_dg = ~isnan(d_g);
while (~out_converge && n<params.stop.iterlim)
    in_converge = false;
    errV = inf;
    errX = inf;
    errQ = inf;
    errP = inf;
    if ~strcmpi(options.alg,'PDHG')
        c_g1p = c_g(:,1:nr);
    end
    while (~in_converge && n<params.stop.iterlim)
        x_gp = x_g;
        q_gp = q_g;
        p_gp = p_g;
        xh_gp = xh_g;

        if strcmpi(options.alg,'PDHG')
            c_g1p = c_g(:,c_adapt_ind);
        end

        [q_g,p_g,err_g] =...
                feval(tvq2kern,q_g,p_g,err_g,xh_g,c_g,g_g,d_g,...
                p_opt_g,p_wght_g,p_norm_g,p_hub_g,nr,nc,num_dm);

        [x_g,xh_g,c_g] =...
                feval(tvxkern,x_g,xh_g,c_g,err_g,p_g,q_g,g_g,...
                vk_g,p_opt_g,p_wght_g,p_gam_g,p_gam_2,nr,nc,num_dm);

        if options.adapt
            if strcmpi(options.alg,'PDHG')
                errL = sum(sum(abs(c_g1p-c_g(:,c_adapt_ind)).^2))./sum(sum(abs(c_g1p).^2));
            end
            if options.rof
               tau_l = 1./((num_dm+1)*max(sum(sum(err_g.^2))));               
            end
            if strcmpi(options.alg,'AMA')
               tau = tau_in*params.ama.mu/(tau_in+params.ama.mu);
            end
                sigma_p = 1/((num_dm+1)*tau*max(max(c_g(:,c_adapt_ind))).^2);
%             tau_l = 1/((num_dm+1)*tau*max(max(c_g)).^2);
            p_opt_g = gpuArray(eval([mtype,'([tau,sigma,sigma_p,tau_l,zeta,zeta_v])']));
        end

        errX = sum(sum(abs(x_g-x_gp).^2))./sum(sum(abs(x_gp).^2));
        errQ = sum(sum(abs(q_g-q_gp).^2))./sum(sum(abs(q_gp).^2));
        errV = errX + errQ;
        if ~options.rof
            errP = sum(sum(abs(p_g-p_gp).^2))./sum(sum(abs(p_gp).^2));
            errV = errV + errP;
        end

        x_gb = x_g;
        n = n+1;
        
        in_converge = errX<=params.stop.crit;
        if strcmp(options.alg,'PDHG') && options.adapt
            in_converge = in_converge & errL<=params.stop.crit;
        end

        % %% Show current regularized depth map
        if options.debug_lvl > 1 && mod(n,100)==0
            ad = double(gather(x_g(1:nc,1:nr)'));
            ad(~val_pts)=nan;
            Cref = double(gather(c_g'));
            Cref = reshape(Cref,[nr,num_dm+1,nc]);
            Cref = permute(Cref,[1,3,2]);
            Cref = reshape(Cref,[nr*nc,num_dm+1]);
            set(0,'CurrentFigure',conf_fig); 
            
            if options.adapt
                Cref_d = Cref(:,end);
            else
                Cref_d = Cref(:,ref_ind);
            end
            Cref_d(~val_pts)=nan;

            imagesc(reshape(abs(Cref_d),[nr,nc]));
            colorbar; axis image; axis off;

            set(0,'CurrentFigure',post_reg_fig)

            imaft = disp_to_color(options.vis.max./reshape(ad,[nr,nc]),255);
            set(hImage,'CData',imaft);
            drawnow;
        end
        if options.capture      
            F_v = disp_to_color(options.vis.max./reshape(a,[nr,nc]),255);
            F.cdata = F_v;
            F.colormap = [];
            writeVideo(wr_vid(1),F);

            if options.adapt
                Cref_d = Cref(:,end);
            else
                Cref_d = Cref(:,ref_ind);
            end
            Cref_d(~val_pts)=nan;
            b_d = log(3)*1./Cref_d;
            im_or = reshape(b_d,[nr,nc]);
            im_eq = im_or;
            im_eq(im_or>1.0)=1.0;

            map_sz = 256;
            im_ind = rgb2ind(repmat(im_eq,[1,1,3]),gray(map_sz),'nodither');
            im_ind(~val_pts)=nan;

            F.cdata = im_ind;
            F.colormap = jet(map_sz);
            writeVideo(wr_vid(2),F);
        end
        if options.debug_lvl >= 1 && mod(n,100)==0
            if exist('count','var')
                fprintf(1, repmat('\b',1,count)); %delete line before
            end
            count = fprintf(1,'Iteration: %d, errorX: %.4g, errorQ: %.4g, errorP: %.4g errorL: %.4g\n', n,errX,errQ,errP,errL);
        end
    end
    if strcmpi(options.alg,'PDHG') || ~options.adapt
        out_converge = true;
    else
        x_g = x_gb;
        c_g1p = c_g(:,c_adapt_ind);
        err_resh = reshape(err_g,[nc,nr,num_dm]);
        if strcmpi(options.alg,'ACS')
            c_g1= alpha./(sum(abs(err_resh),3)+beta);
        elseif strcmpi(options.alg,'AMA')
            sq_b = c_g1p-params.ama.nu*(sum(abs(err_resh),3)+beta);
            Disc = sq_b.^2-params.ama.nu*(1-alpha-reshape(valK,nr,nc)');
            c_g1= 0.5*(sq_b+sqrt(Disc));          
        end
        c_g(:,c_adapt_ind) = c_g1;
        errL = sum(sum(abs(c_g1p-c_g1).^2))./sum(sum(abs(c_g1p).^2));

        out_converge = errL<=params.stop.crit;
    end
end
if options.debug_lvl == 1
    fprintf(1,'\n');
end
t = toc(ticreg);
if options.debug_lvl >= 0 
    fprintf('Regularization completed after %d iterations\nRegularization time: %f\n',n,t);
    fprintf('[Final] Iteration: %d, errorX: %.4g, errorQ: %.4g, errorP: %.4g errorL: %.4g\n', n,errX,errQ,errP,errL);
end
meta.errorX=errX;
meta.errorQ=errQ;
meta.errorP=errP;
meta.errorL=errL;
meta.iterations=n;

if options.capture
    close(wr_vid(1))
    close(wr_vid(2))
end

a = double(gather(x_gb(1:nc,1:nr)'));
        
dm = reshape(a,[nr,nc]);
dm(~val_pts) = nan;
if options.debug_lvl > 1 && ~isempty(I_r)
    ad = a;
    ad(~val_pts) = nan;
    overlayed_result_figure = figure; set(overlayed_result_figure, 'Name', 'Overlayed result depth map');
    imagesc(heatmap_overlay(I_r, reshape(ad,[nr,nc])));
end

end

