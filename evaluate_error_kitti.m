% %%
% Compute KITTI benchmark's evaluation metrics
%
% Signature:  [avg,out,compl]=evaluate_error_kitti(disp_gt,disp_or)
%
% Input:
%   disp_gt - ground truth disparity map
%   disp_or - computed disparity map
% Output:
%   avg - average disparity error in pixels
%   out - percentage of pixels above the threshold
%   compl - density/completeness

% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome

function [avg,out,compl]=evaluate_error_kitti(disp_gt,disp_or)

disp_i = disp_int(disp_or);
val_or = disp_or>0;

val_gt = disp_gt>0;
val_comm = val_gt&val_or;

pts_gt = sum(sum(val_gt));
pts_comm = sum(sum(val_comm));

disp_diff = abs(disp_gt-disp_i);

avg=[sum(sum(disp_diff(val_gt)))/pts_gt,sum(sum(disp_diff(val_comm)))/pts_comm];
out = zeros(5,2);
for kk=1:5
    out(kk,:)=[sum(sum((disp_diff.*val_gt)>kk))./pts_gt,sum(sum((disp_diff.*val_comm)>kk))/pts_comm]*100;
end
compl = pts_comm./pts_gt;
end