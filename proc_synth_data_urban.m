
% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome


clc
close all
% clear 

% specify the name of the parameter that varies {bsteps,bsizes,dscales}
varname = 'dscales';
basedir = 'G:\Data\Stereo\processed'; % directory where the data is stored
w_path = 'G:\Fusion\Results\test'; % output directory

objfn = {'agiasophia_data','atlantica_data','chrysler_data','kelvingrove_data'};
styles= {'rof','tgvfus','l1-heur','heur-adapt','heur-adapt-G','rof-adapt','l1-adapt'};
algs = {'PDHG','AMA','ACS'};

% Specify parameters
debuglvl = 1;
ffact = 1.2;
imgsize = [640 480]; 
intrinsics = [imgsize(2)*ffact imgsize(2)*ffact imgsize(1)/2 imgsize(2)/2]; % intrinsic camera parameters (fu,fv,cu,cv)
ref = 6; 
bsizes = 3:2:11;
bsteps = 1:3;
dscales = 1:1:10;  
distrfun = @randlap; % change to randn for gaussian noise

bsize = bsizes(end);
bstep = bsteps(1);
dscale = dscales(6);

xvar = eval(varname);

results = cell([length(xvar),length(objfn),length(styles),length(algs)]);
for mm=1:length(xvar)
    if ~isempty(results{mm,end,end,1})
        continue;
    end
for jj=1:length(objfn)
    if ~isempty(results{mm,jj,end,1})
        continue;
    end
    datafile = objfn{jj};
    
    % Load data from file
    load(fullfile(basedir,datafile));
    d = model.data(1).step(1); 
    cB =imgsize(2)*ffact*d;
    
    %
    % Prepare data
    K = [intrinsics(1),0,intrinsics(3);0,intrinsics(2),intrinsics(4);0,0,1];
    if strcmpi(varname,'bsteps')
        winsize= floor(bsize/(2*bsteps(end)));
    elseif strcmpi(varname,'bsizes')
        winsize= floor(xvar(mm)/(2*bstep));
    else
        winsize= floor(bsize/(2*bstep));
    end
    ims = prod(imgsize);
    if strcmpi(varname,'bsteps')
        bundle = ref-winsize*xvar(mm):xvar(mm):ref+winsize*xvar(mm);
    else
        bundle = ref-winsize*bstep:bstep:ref+winsize*bstep;
    end
    nbsize = length(bundle);
    dm = cell(1,nbsize);
    H  = cell(1,nbsize);
    for bb=1:nbsize
        dataind = bundle(bb);
        if strcmpi(varname,'dscales')
            noise1d = distrfun(1,ims)*xvar(mm);
        else
            noise1d = distrfun(1,ims)*dscale;
        end
        noise= reshape(noise1d,fliplr(imgsize));

        dmt = double(model.data(dataind).depth);  
        val_pts = ~isnan(dmt);
        if dataind == ref
            ref_rel = bb;
            dm_gt = dmt;
            I_r = model.data(dataind).image;
        end
        dmt(~val_pts)=0;
        noise(~val_pts) = 0;
        dm{bb} = dmt+noise;
        t = [model.data(dataind).step(1)*dataind,0,300]';
        ry=0;
        R = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)];

        H{bb} = [R',-R'*t;zeros(1,3),1];
    end
    data_fuse = dm_fusion_prepare(dm,K,H,H{ref_rel},'geometric',0,[],debuglvl);

    %
    % Fuse depthmaps

    for kk=1:length(styles)
        [options,params] = get_fusion_options(styles{kk});
        params.stop.iterlim = 10000;
        params.stop.crit=1e-10;
        params.ama.mu = 1/sqrt(ims);
        options.debug_lvl = debuglvl;
        for ll=1:length(algs)
            if ~isempty(strfind(styles{kk},'adapt')) 
%                 if strcmpi(algs{ll},'PDHG')
%                     continue;
%                 end
            else
                if ~strcmpi(algs{ll},'PDHG')
                    continue;
                end
            end
            if ~isempty(results{mm,jj,kk,ll})
                continue;
            end
            options.alg=algs{ll};
            if debuglvl>1
                close all;
            end
            [dmm,meta] = cfusion(data_fuse.depth,data_fuse.confidence,data_fuse.median,I_r,params,options);
             %
            % Compute errors
            lim = 3;

            errmask = model.data(ref).mask;
            fprintf('Variable: %g, object: %s, style: %s, algorithm: %s\n',xvar(mm),objfn{jj},styles{kk},algs{ll});
            try
                err=compute_errors(dmm.*errmask,dm_gt,K,errmask,cB);
                fprintf('[DEPTH] RMSE: %.3g, ZMAE: %.3g, NMAE: %.3g, Avg.: %.3g\n',err.depth.RMSE,err.depth.ZMAE,err.depth.NMAE,err.depth.average);
                fprintf('[DISP_%d] Avg: %.2f [pix], Out: %.2f [%%], Comp: %.2f [%%]\n',lim,err.disp.avg(1),err.disp.out(lim,1),err.disp.complete*100);
            catch
                err=[];
                fprintf('Interpolation failed!\n');
            end
            
            res_data.err=err;
            res_data.options=options;
            res_data.parameters=params;
            res_data.result = dmm;
            res_data.mean = data_fuse.mean;
            res_data.median = data_fuse.median;
            res_data.dm_ref= data_fuse.dm_ref;
            res_data.meta = meta;
            results{mm,jj,kk,ll}=res_data;
            try
                if mod(kk,2)
                    save(fullfile(w_path,'tmp_urb_lap1_add.mat'),'results','objfn','styles','algs','-v7.3');
                else
                    save(fullfile(w_path,'tmp_urb_lap2_add.mat'),'results','objfn','styles','algs','-v7.3');
                end
            catch
                warning('Saving of the workspace failed ')
            end 
        end
    end
end
end