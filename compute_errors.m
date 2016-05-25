% %%
% Compute evaluation metrics
%
% Signature:  err = compute_errors(Z,Z_gt,K,mask_im,cB)
%
% Input:
%   Z -  computed depth map
%   Z_gt - ground truth depth map
%   K - camera calibration matrix
%   mask_im - valid image regions (logical matrix)
%   cB - product of focal length with camera basesline  
% Output:
%   err - structure containing the results of depth and disparity evaluation 

% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome

function err = compute_errors(Z,Z_gt,K,mask_im,cB)

if nargin<4
    mask_im = true;
end

Z(isnan(Z))=0;
Z_gt(isnan(Z_gt))=0;

mask = logical(Z~=0 & Z_gt~=0 & mask_im);

% se = strel('disk',1);
% mask = imerode(mask,se);

[nr,nc,~] = size(Z_gt);
ims=nr*nc;

zdiffsr = (Z_gt(mask)-mean(Z_gt(mask)))-(Z(mask)-mean(Z(mask)));
err.depth.RMSE = sqrt(sum(zdiffsr.^2)/sum(mask(:)));

zdiffs = Z_gt(1:ims)-Z(1:ims);
medz = median(Z_gt(mask)-Z(mask));
err.depth.ZMAE = mean( abs(zdiffs(mask)-medz));

[~,Nm] = dmap2normap(Z,K);
[~,Nm_gt] = dmap2normap(Z_gt,K);
Nm(isnan(Nm)) = 0;
Nm_gt(isnan(Nm_gt)) = 0;


Nm_v = reshape(Nm,[ims,3]);
Nm_gt_v = reshape(Nm_gt,[ims,3]);

% compare normals
ndiffs = sum(Nm_v.*Nm_gt_v,2);
angdiffs = acos(min(1,ndiffs));
err.depth.NMAE = mean(angdiffs(mask));

disp_gt = cB./Z_gt;
disp_gt(isnan(disp_gt))=0;
disp_gt(disp_gt<0|disp_gt>255)=0;

disp = cB./Z;
disp(isnan(disp))=0;
disp(disp<0|disp>255)=0;

[avg,out,compl]=evaluate_error_kitti(disp_gt,disp);

err.disp.avg = avg;
err.disp.out = out;
err.disp.complete = compl;

errarr = struct2array(err.depth);
err.depth.average = geomean(errarr);