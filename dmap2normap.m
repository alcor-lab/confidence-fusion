%%
% Compute the normal map from a depth map
% Signature: Nnu = dmap2normap(dm,K)
% 
% Input:
%   dm - depth image
%   K - 3x3 calibration matrix
% Output:
%   Nnu - normal map

% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome

function [Nnu,Nn] = dmap2normap(dm,K)

[nr,nc] = size(dm);
[y,x] = ind2sub([nr,nc],1:nr*nc);
u = [x;y;ones(1,nr*nc)];

XX = repmat(double(dm(:)'),3,1).*(K\u);
clear u x y dm K

Vm = zeros(nr,nc,3);
for ii=1:3
    Vm(:,:,ii) = reshape(XX(ii,:),[nr,nc]);
end
clear XX

Yc = zeros([nr,nc,3]);
Xc = zeros([nr,nc,3]);
Xc(2:end,2:end,:) = Vm(2:end,1:end-1,:)-Vm(2:end,2:end,:);
Yc(2:end,2:end,:) = Vm(1:end-1,1:end-1,:)-Vm(2:end,1:end-1,:);
Nm = cross(Xc,Yc,3);
clear Xc Yc Vm nr nc

Nn = bsxfun(@rdivide,Nm,sqrt(sum(Nm.^2,3)));
clear Nm

Nnu = uint8((Nn./2+.5).*255);