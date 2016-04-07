% %%
% This function interpolates disparity values similarly to KITTI
% dataset devkit's interpolateBackground() function

% %%
% Author: Valsamis Ntouskos
% e-mail: ntouskos@diag.uniroma1.it
% ALCOR Lab, DIAG, Sapienza University of Rome

function d_int = disp_int(d_im)

    d_int = d_im;
    
    mask = (d_im<=0);
    d_int(mask) = inf;
    dmask = diff(mask')';
    [Iin,Jin] = find(dmask==1);
    [Iout,Jout] = find(dmask==-1);
    
    ind_f = union(Iin,Iout);
    for kk = ind_f'
        set_in = Jin(Iin==kk);
        set_out = Jout(Iout==kk)+1;
        if ~isempty(set_in)
            for ll = 1:length(set_in)-1
                d_int(kk,set_in(ll)+1:set_in(ll+1)-1) = min(d_int(kk,set_in(ll)+1:set_in(ll+1)-1),d_int(kk,set_in(ll))); 
            end
            d_int(kk,set_in(end)+1:end) = min(d_int(kk,set_in(end)+1:end),d_int(kk,set_in(end))); 
        end
        if ~isempty(set_out)
            for ll = 2:length(set_out)
                d_int(kk,set_out(ll-1)+1:set_out(ll)-1) = min(d_int(kk,set_out(ll-1)+1:set_out(ll)-1),d_int(kk,set_out(ll))); 
            end
            d_int(kk,1:set_out(1)-1) = min(d_int(kk,1:set_out(1)-1),d_int(kk,set_out(1)));
        end
    end
    
    maskv = isinf(d_int);
    dmaskv = diff(maskv);
    [Iinv,Jinv] = find(dmaskv==1);
    [Ioutv,Joutv] = find(dmaskv==-1);
    ind_fv = union(Jinv,Joutv);
    for kk = ind_fv'
        set_in = Iinv(Jinv==kk);
        set_out = Ioutv(Joutv==kk)+1;
        if ~isempty(set_in)
            for ll = 1:length(set_in)-1
                d_int(set_in(ll)+1:set_in(ll+1)-1,kk) = min(d_int(set_in(ll)+1:set_in(ll+1)-1,kk),d_int(set_in(ll),kk)); 
            end
            d_int(set_in(end)+1:end,kk) = min(d_int(set_in(end)+1:end,kk),d_int(set_in(end),kk)); 
        end
        if ~isempty(set_out)
            for ll = 2:length(set_out)
                d_int(set_out(ll-1)+1:set_out(ll)-1,kk) = min(d_int(set_out(ll-1)+1:set_out(ll)-1,kk),d_int(set_out(ll),kk)); 
            end
            d_int(1:set_out(1)-1,kk) = min(d_int(1:set_out(1)-1,kk),d_int(set_out(1),kk));
        end
    end
    
    d_int(~mask) = d_im(~mask);
    
end