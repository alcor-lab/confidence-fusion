% %%
% Function for preparing depth data for confidence driven fusion
%
% Signature [data_out] = dm_fusion_prepare(dmaps,K,Hset,Href,hconf,nan_th,debug_lvl)
%
% Input: 
%   dmaps - cell array containing the input depth images
%   K - 3x3 caibration matrix
%   Hset - cell array containing the pose of the corresponding depth image
%           as a 4x4 transformation matrix w.r.t. the world frame
%   Href - the pose of the reference frame of the fused depth image
%   hconf (optional) - one of the following: a handle of the function for
%           computing confidence values; a cell array containing confidence
%           maps; the string 'geometric' for normal based confidence (see paper)
%   nan_th (optional) -  minimum depth value to be replaced by NaNs
%   debug_lvl (optional) - level of debug information
% Ouput:
%   data_out - structure containing all the necessary data for performing
%       confidence driven fusion (see cfusion function)

% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome

function [data_out] = dm_fusion_prepare(dmaps,K,Hset,Href,hconf,nan_th,auto,debug_lvl)
if nargin < 7
    debug_lvl = 0;
end
if nargin < 6
    nan_th = 0;
end
if nargin < 5
    hconf = 0;
end

num_dm = length(dmaps);
ref = floor(num_dm/2)+1;
if num_dm < 1
    throw(MException('MERGING:INSUFDATA','Error: Insufficient number of depth maps.'))
end
if isa(hconf,'function_handle')
    for kk = 1:num_dm
        cm = hconf(dmaps{kk});
    end
elseif iscell(hconf)
    cm = hconf;
else
    cm = cell(1,num_dm);
end

[nr,nc] =size(dmaps{ref});
ind_b = 1:num_dm;

if isstruct(auto)
    th_ang = zeros(1,num_dm);
    sp_disp = zeros(1,num_dm);
    for kk=ind_b
        Lm = logm(Href\Hset{kk});
        th_ang(kk) = norm([-Lm(2,3);Lm(1,3);-Lm(1,2)]);
        sp_disp(kk) = norm(Lm(1:3,4));
    end
    mask = th_ang<auto.ang & sp_disp<auto.dist;
    ind_b = ind_b(mask);
    ref = find(ind_b==ref);
    data_out.bundle = ind_b;
    data_out.ref = ref;
end
b_len = length(ind_b);

if debug_lvl>0
    fprintf('Reprojecting depth maps to reference frame...');
end
dm_rep = cell(b_len,1);
cm_rep = cell(b_len,1);
nm_rep = cell(b_len,1);
dmvect = inf(nr*nc,num_dm);
dmval = true(nr*nc,num_dm);
for kk=1:b_len
    dmaps{kk}(isnan(dmaps{kk})) = 0;
    [dm_rep{kk}, cm_rep{kk}, nm_rep{kk}] = reproject_dm_rel(dmaps{ind_b(kk)},K,Href,Hset{ind_b(kk)},cm{ind_b(kk)},debug_lvl);
    dmvect(:,kk) =  dm_rep{kk}(:);
    dmval(:,kk) = (dmvect(:,kk)>nan_th);
    dmvect(~dmval(:,kk),kk) = 0;
    dm_rep{kk}(~dmval(:,kk)) = nan;
end
if debug_lvl>0
    fprintf('\tdone!\n');
end
data_out.dm_ref = dm_rep{ref};
data_out.depth = dm_rep;

% %%
% Assign mean and median
dm_mean = sum(dmvect,2)./sum(dmval,2);
data_out.mean = reshape(dm_mean,[nr,nc]);

dmvect(~dmval)=nan;
dm_med = median(dmvect,2,'omitnan');
data_out.median = reshape(dm_med,[nr,nc]);


% %% Assign confidence 
if isa(hconf,'function_handle')
    cm = cm_rep;
elseif strcmpi(hconf,'geometric')
    % confidence based on normals
    [uy,ux] = ind2sub([nr,nc],1:nr*nc);
    u=[ux;uy;ones(1,length(ux))];
    d_v = K\u;
    d_v = d_v ./ repmat(sqrt(sum(d_v.^2)),3,1);
    d_vmap = reshape(d_v',[nr,nc,3]);
    for kk=1:b_len
        cm{kk} = max(0,min(1,abs(sum(d_vmap.*nm_rep{kk},3))));
%         cm{kk}(~dmval(:,kk))=nan;
    end
else
    for kk=1:b_len
        cm{kk} = ones(nr,nc);
%         cm{kk}(~dmval(:,kk))=nan;
    end
end
data_out.confidence = cm;
end