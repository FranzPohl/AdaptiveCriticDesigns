% Create and Open File for Recording Video

if livestream == true      
    M(n) = struct('cdata',[],'colormap',[]);        % Vector for saving frames
    v = VideoWriter('DHP_Control.avi');             % Video setter
    open(v); 
end