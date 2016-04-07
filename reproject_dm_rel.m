% %%
% Reproject depth images and confidence maps 
%
% Signature: [dm_rep,cm_rep] = reproject_dm_rel(dm,K,Href,Hm,cm,debug_lvl)
%
% Input:
%   dm - cell array containing the input depth images
%   K - 3x3 caibration matrix
%   Href - the pose of the reference frame of the fused depth image
%   Hm - cell array containing the pose of the corresponding depth image
%           as a 4x4 transformation matrix w.r.t. the world frame
%   cm (optional) - confidence maps
%   debug_lvl (optional) - level of debug information
% Ouput:
%   dm_rep - cell array containing reprojected depth images
%   cm_rep - cell array containing reprojected confidence maps

% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome

function [dm_rep,cm_rep,nm_rep] = reproject_dm_rel(dm,K,Href,Hm,cm,debug_lvl)

if nargin < 6
    debug_lvl = 0;
end
if nargin < 5
    cm = [];
end

Hrel = Href\Hm;
if abs(trace(Hrel(1:3,1:3)) - 3) < 1e-6 && norm(Hrel(1:3,4)) < 1e-4
    dm_rep = dm;
    if ~isempty(cm)
        cm_rep = cm;
    else
        cm_rep = [];
    end
    if nargout == 3
        [~,nm_rep] = dmap2normap(dm,K);
    end
    return
else
    H_mr = Href\Hm;

    R_r = H_mr(1:3,1:3);
    t_r = H_mr(1:3,4);
end

% %% 
% Reproject depthmap on a different viewpoint
[nr, nc] = size(dm);
[y,x] = ind2sub([nr, nc],1:nr*nc);
u = [x;y;ones(1,nr*nc)];
dm_v = dm(:)';
pcl_v = repmat(dm_v,3,1).*(K\u);

Xr = pcl_v;

Xc = R_r*Xr+repmat(t_r,1,size(Xr,2));
[Xc,indsort] = sortrows(Xc',-3);
Xc = Xc';

xc = K*Xc;

valid_ind = xc(3,:) > 0;
xc_v = xc(:,valid_ind);
xc_v = round(xc_v./repmat(xc_v(3,:),3,1));
[xc_un,un_ind] = unique(xc_v','rows','last');
valid_ind = true(size(xc_un,1),1);
valid_ind = valid_ind & xc_un(:,1)>0 & xc_un(:,1)<=nc & xc_un(:,2)>0 & xc_un(:,2)<=nr;
xc_val = xc_un(valid_ind,:);


kk = sub2ind([nr,nc],xc_val(:,2),xc_val(:,1));

dm_rep = zeros(nr,nc);
dm_rep(kk) = Xc(3,un_ind(valid_ind));
    
dm_m = dm_rep;

% fill in small gaps
se = strel('disk',5);
i_b = dm_m~=0;
i_dil = imdilate(i_b,se);
i_er = imerode(i_dil,se);
mask = i_er~=i_b;

F = scatteredInterpolant(xc_val(:,1),xc_val(:,2),Xc(3,un_ind(valid_ind))','natural');
[yi,xi] = find(mask);
qz = F(xi,yi);
dm_rep(sub2ind([nr,nc],yi,xi)) = qz;

if debug_lvl > 2
    figure(); imagesc(dm_rep); axis image; axis off;
end

if ~isempty(cm)
    cm_rep = zeros(nr,nc);
    cm_rep(kk) = cm(indsort(un_ind(valid_ind)));
    Fcm = scatteredInterpolant(xc_val(:,1),xc_val(:,2),cm(indsort(un_ind(valid_ind))),'natural');
    [yic,xic] = find(mask);
    qzc = Fcm(xic,yic);
    cm_rep(sub2ind([nr,nc],yic,xic)) = qzc;
    if debug_lvl > 2
        figure(); imagesc(cm_rep); axis image; axis off;
    end
else
    cm_rep = [];
end

if nargout == 3
    [~,nm] = dmap2normap(dm_rep,K);
    nm_rep = nm;
    if debug_lvl > 2
        figure(); imshow(uint8((nm_rep./2+.5).*255))
    end
end


