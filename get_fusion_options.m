%GET_FUSION_OPTION Provides presets of options and parameters for
% confidence driven fusion

% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome

function [ options, params ] = get_fusion_options( type )

if nargin<1
    type = '';
end

options.vis.max = 128;

options.alg = 'PDHG';
options.uniform = true;

options.cuda.dtype = 'float';

options.debug_lvl = 1;
options.capture = false;

% % Dummy options
options.rof = false;
options.adapt = false;
options.tgv = true;

% % Default paramsization parameters
params.lambda = 1;
params.eta = 0.0; 
params.tau = 0.01;
params.zeta = 1;
params.ord = 2;

params.adapt.tau = params.tau;
params.adapt.alpha= 1.5;
params.adapt.beta = 0.1;

params.tgv.zeta = 1;
params.tgv.alpha_1 = 2;
params.tgv.alpha_0 = 1;

params.stop.crit = 1e-10;
params.stop.iterlim = 20000;

params.aniso.gfilt = fspecial('gaussian',[1,1],1.7);
params.aniso.Gw = 0;
params.aniso.Gexp = 1;

params.cuda.bxd = 32;
params.cuda.byd = 32;

params.ama.mu = 1;
params.ama.nu = 1;

params.pdhg.sep = false;
switch type
    case 'rof'
        options.rof = true;
        options.tgv = false;   
     case 'l1'
        options.tgv = true;
    case 'tgvfus'
        params.eta=0.05;
    case 'l1-heur'
        options.uniform = false;
    case 'heur-adapt'
        options.uniform = false;
        options.adapt = true;
    case 'heur-adapt-G'
        options.uniform = false;
        options.adapt = true;
        params.aniso.Gw = 2;
        params.aniso.Gexp = 1.5;
    case 'rof-adapt'
        options.rof = true;
        options.adapt = true;
    case 'l1-adapt'
        options.adapt = true;
    otherwise
        fprintf('Loading default fusion options\n');
end

