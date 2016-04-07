
% img = image on which to overlay heatmap
% heatmap = the heatmap
% (optional) colorfunc .. this can be 'jet' , or 'hot' , or 'flag'

function omap = heatmap_overlay( img , heatmap, par, colorfun )

if ( nargin < 3 )
    par = 0.98;
end
if ( nargin < 4 )
    colorfun = 'parula';
end

if ( ischar(img))
    img = imread(img); 
end
if ( isa(img,'uint8') == 1 ) 
    img = double(img)/255; 
end

szh = size(heatmap);
szi = size(img);

if ( (szh(1)~=szi(1)) || (szh(2)~=szi(2)) )
  heatmap = imresize( heatmap , [ szi(1) szi(2) ] , 'bicubic' );
end
  
if ( size(img,3) == 1 )
  img = repmat(img,[1 1 3]);
end
  

colorfunc = eval(sprintf('%s(50)',colorfun));

heatmap = double(heatmap) / max(heatmap(:));
omap = par*(1-repmat(heatmap.^par,[1 1 3])).*double(img)/max(double(img(:))) + repmat(heatmap.^par,[1 1 3]).* shiftdim(reshape( interp2(1:3,1:50,colorfunc,1:3,1+49*reshape( heatmap , [ numel(heatmap) 1 ] ))',[ 3 size(heatmap) ]),1);
omap = real(omap);