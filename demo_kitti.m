
% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome


clc
close all
clear

stylesall = {'rof','tgvfus','l1-heur','heur-adapt','heur-adapt-G','rof-adapt','l1-adapt','l1'};
algsall = {'ACS','AMA','PDHG'};

% %%%%%%%%%%%%%%%%%%
% Main options and parameters 
styles = stylesall(4);
algs = algsall(1);

basedir = 'G:\Data\Stereo';
w_path = 'G:\Fusion\Results';

type = 'testing';
% type = 'training';

locvers = 'viso';
% locvers = 'gm';

% bundle images to use for fusion
ind_b = 9:13;

lim = 3;
debuglvl = 2;
% %%%%%%%%%%%%%%%%%

load(sprintf('data/bundle_loc_%s_%s.mat',locvers,type));

cal_path = fullfile(basedir,'KITTI_dataset',sprintf('%s',type),'calib');
d_path = fullfile(basedir,'KITTI_dataset',sprintf('%s',type));
w_path = fullfile(w_path);

if strcmp(type,'training')    
    eval_set = 0:193;
elseif strcmp(type,'testing')
    eval_set = 0:194;
else
    return;
end
numbundles = length(eval_set);

% %%
% Depth map parameters
dmin = 0.7;
dmax = 100.0;

% %%
% % ELAS method parameters, uncomment if you use ELAS (see below)
% param.disp_min    = 0;           % minimum disparity (positive integer)
% param.disp_max    = 255;         % maximum disparity (positive integer)
% param.subsampling = 0; % process only each 2nd pixel (1=active)
% param.add_corners = 1; 
% param.match_texture = 0; 
% gfilt = fspecial('gaussian',[7,7],0.7);
% %%

t_d = zeros(numbundles,1);
Imp = zeros(numbundles,1);
for bb = eval_set
    fprintf(1,'Processing bundle %d\n',bb);
    tictot = tic;
    fcal = fopen(fullfile(cal_path,sprintf('%.6d.txt',bb)),'r');
    incal = fscanf(fcal,'P0: %f %f %f %f %f %f %f %f %f %f %f %f\nP1: %f %f %f %f %f %f %f %f %f %f %f %f');
    fclose(fcal);

    Pr = reshape(incal(13:end),[4,3])';
    cB = -Pr(1,4);

    K=Pr(1:3,1:3);    

    if strcmpi(type,'training')
        if bb==31 || bb==82
            first_frame = 5;
        else
            first_frame = 0;
        end
        if bb==114
            last_frame  = 18;
        else
            last_frame  = 20;
        end
   elseif strcmpi(type,'testing')
        if bb==127 || bb==182
            first_frame = 5;
        else
            first_frame = 0;
        end
        last_frame  = 20;
    end
    
    nm_d = last_frame-first_frame+1;
    dm = cell(nm_d,1);
    ref_rel = 10-first_frame+1;
    
    fprintf('Computing depth maps...');
    for ll = first_frame:last_frame
        lfile = fullfile(d_path,'image_0',sprintf('000%03d_%02d.png',bb,ll));
        rfile = fullfile(d_path,'image_1',sprintf('000%03d_%02d.png',bb,ll));
        dispfile = fullfile(d_path,'disp_0',sprintf('000%03d_%02d.png',bb,ll));
        
        I1or = imread(lfile);
        I2or = imread(rfile);
        if ll==10
            I_r = I1or;
        end
        
        D1 = im2double(imread(dispfile)')*255;
%       % Replace the previous line with these if you want to test with ELAS
%         I1=imfilter(I1or,gfilt); 
%         I2=imfilter(I2or,gfilt); 
%         if length(size(I1))==3
%             I1 = rgb2gray(I1);
%         end
%         if length(size(I2))==3
%             I2 = rgb2gray(I2);
%         end
%         [D1,D2] = elasMex(I1',I2',param); 

        D1t=D1;
        D1t(D1t<(cB./dmax))=nan;
        D1m = cB./D1t;
        D1m(isnan(D1m))=0;
        
        dm{ll-first_frame+1} = double(D1m');
        if ll==10
            Dspr = D1t';
        end

        if debuglvl > 2
            imagesc(D1'); 
            colorbar; axis image; axis off;
            pause(0.01)
        end
    end
    fprintf(1,'\tdone!\n');
    [nr,nc] = size(I_r);
    ims = nr*nc;
    
    H = bundle_data(bb+1).localization; 
    
    data_fuse = dm_fusion_prepare(dm(ind_b),K,H(ind_b),H{ref_rel},'geometric',0,[],debuglvl);

    if strcmpi(type,'training')
        dm_gt_noc =  im2double(imread(fullfile(d_path,'disp_noc',sprintf('%06d_10.png',bb))))*255;
        dm_gt_occ =  im2double(imread(fullfile(d_path,'disp_occ',sprintf('%06d_10.png',bb))))*255;
    end

    % Fuse depthmaps
    results =cell([1,1,length(styles),length(algs)]);
    for kk=1:length(styles)
        [options,params] = get_fusion_options(styles{kk});
        options.debug_lvl = debuglvl;
        for ll=1:length(algs)
            options.alg=algs{ll};
            if debuglvl>1
                close all;
            end
            [dmm,meta] = cfusion(data_fuse.depth,data_fuse.confidence,data_fuse.median,I_r,params,options);
                        
            dme = cB./dmm;
            dme(isnan(dmm))=0;
            dme(dme>255|dme<0)=0;
            
            if strcmpi(type,'training')
                fprintf('Option: %s, algorithm: %s\n',styles{kk},algs{ll});
                [err.disp.avg,err.disp.out,err.disp.complete]=evaluate_error_kitti(dm_gt_noc,dme);    
                [errocc.disp.avg,errocc.disp.out,errocc.disp.complete]=evaluate_error_kitti(dm_gt_occ,dme);
                fprintf('[DISP_%d] Avg: %.2f [pix], Out: %.2f [%%], Comp: %.2f [%%]\n',lim,err.disp.avg(1),err.disp.out(lim,1),err.disp.complete*100);

                dmeor = cB./dm{ref_rel};
                dmeor(isnan(dm{ref_rel}))=0;
                dmeor(dmeor>255|dmeor<0)=0;
            
                [error.disp.avg,error.disp.out,error.disp.complete]=evaluate_error_kitti(dm_gt_noc,dmeor); 
                Imp(bb+1)= error.disp.out(lim,1)-err.disp.out(lim,1);
                fprintf('[Improvement] Out: %.4f\n',error.disp.out(lim,1)-err.disp.out(lim,1));
                
                res_data.err=err;
                res_data.errocc=errocc;
            else
                imwrite(uint16(dme*257),fullfile(w_path,type,sprintf('%.6d_10.png',bb)),'png')
            end
            res_data.options=options;
            res_data.parameters=params;
            res_data.result = dmm;
            res_data.mean = data_fuse.mean;
            res_data.median = data_fuse.median;
            res_data.dm_ref= data_fuse.dm_ref;
            res_data.meta = meta;
            results{1,1,kk,ll} = res_data;     
           
        end
    end
    if strcmpi(type,'training')
        save(fullfile(w_path,type,sprintf('res_kitti_%s_%03d.mat',locvers,bb)),'results','styles','algs','dm_gt_noc','dm_gt_occ','-v7.3');
    else
        save(fullfile(w_path,type,sprintf('res_kitti_testing_%s_%03d.mat',locvers,bb)),'results','styles','algs','-v7.3');
    end
    t_d(bb+1) = toc(tictot);
       
    fprintf(1,'Total time: %f\n',t_d(bb+1));
%     beep
    close all
end

