function show_gradients( path )
    load([path '/dims.txt']);
    clear('size');
        
    grads=load3D([path '/grads.raw'], dims);
    points=load3D([path '/points.raw'], dims);
    preds=load3D([path '/preds.raw'], dims);
    maximin_seg=loadSeg([path '/maximin_seg.raw'], dims);
    seg=loadSeg([path '/seg.raw'], dims);
    
    compTrue = flip_aff(connectedComponents(flip_aff(points)));   
    
    overlay = 0.05 * avg3D(points) - grads;
    BrowseComponents('cici', seg, preds, maximin_seg, overlay);
    
    %BrowseComponents('iii', points, preds, ...
        %0.05 * avg3D(points) - grads);
    BrowseComponents('iii', points, preds, ...    
        0.05 * avg3D(points) - toRed3D(mean(grads, 4)));
end

function a = flip_aff(M)
    a = M(end:-1:1, end:-1:1, end:-1:1, :);
end

function out = loadSeg( file, dims )
    fid = fopen(file, 'r', 'ieee-be');
    out = fread(fid, Inf, 'int32');
    fclose(fid);
    out = permute(reshape(out, fliplr(dims)), [3,2,1]);   
end

function out = load3D( file, dims )
    fid = fopen(file, 'r', 'ieee-be');
    out = fread(fid, Inf, 'float');
    fclose(fid);
    out = permute(reshape(out, [3 fliplr(dims)]), [4,3,2,1]);   
end

function a = avg3D(X)
    a = repmat(mean(X,4), 1, 1, 1, 3);
end

function a = toRed3D(X)
    a = cat(4, X, zeros([size(X), 2]));
end