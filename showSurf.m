% Script for visualizing the surfaces corresponding to the fused depth data.
% (Loading of the baseline data is required)

% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome

% styles= {'rof','tgvfus','l1-heur','heur-adapt','heur-adapt-G','rof-adapt','l1-adapt'};
% algs = {'PDHG','AMA','ACS'};

value = 1; 
model = 1;
style = 1;
alg = 1;

const = 300;
l_az=-130;
l_el=65;
v_az=-70; 
v_el=30; 

K = [576   0   320;
     0   576   240;
     0     0     1];

data = results{value,model,style,alg};
mask = isnan(baselines{model}.dm_gt);
% dmm = baselines{model}.dm_gt;
% dmm = data.dm_ref;
% dmm = data.median;
dmm = data.result;
dmm(mask)=nan;
 
[nr,nc] = size(dmm);
[uy,ux] = ind2sub([nr,nc],1:nr*nc);
u=[ux;uy;ones(1,nr*nc)];

U=K\u;

h = surf(reshape(U(1,:),[nr,nc]).*dmm,reshape(U(2,:),[nr,nc]).*dmm,const-dmm,repmat(0.8,[nr,nc,3]),'edgealpha',0); 
axis equal; axis off
shading faceted
lighting flat
lightangle(l_az,l_el)
view(v_az,v_el)
h.AmbientStrength = 0.3;
h.DiffuseStrength = 0.8;
h.SpecularStrength = 0.9;
h.SpecularExponent = 25;
h.BackFaceLighting = 'unlit';
