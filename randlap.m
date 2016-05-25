% %%
% Sample Laplace distribution
%
% Signature: lapsamp = randlap( arg1, arg2 )
%
% Input:
%   arg1: number of rows of output matrix 
%   arg2: number of columns of output matrix (rows=columns if arg2 is not
%         provided)
% Output:
%   lapsamp: matrix containing Laplace distribution samples 

% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome

function [ lapsamp ] = randlap( arg1, arg2 )

if nargin < 2
    unsamp = rand(arg1);
else
    unsamp = rand(arg1,arg2);
end
lapsamp = -sign(unsamp-0.5).*log(1-2.*abs(unsamp-0.5));

end

