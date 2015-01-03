function compare_files( file1, file2 )
    root = [file1 '/..'];
    files = dir(root);
    a = {};
    for k=3:length(files)
        if(files(k).isdir)
            a = [a [root '/' files(k).name]];
        end
    end
    dims = get_dimensions(a);
    [affTrue, affEst1] = load_affs(file1, dims);
    [~, affEst2] = load_affs(file2, dims);
    
    %threshold = 0.956800;
    threshold = 0.93;
    compEst1 = flip_aff(connectedComponents(flip_aff(affEst1)>threshold));
    diff = (affEst2 - affEst1);
    %f = 200 * mean(abs(diff(:)));
    f = 0.2;
    
    overlay = f*avg3D(affTrue) + toRed3D(mean(-diff, 4)>0.1);
    %overlay = f*mean(affTrue, 4) + mean(diff.^2, 4);
    BrowseComponents('ioi', affEst1, compEst1, overlay);
end

function a = avg3D(X)
    a = repmat(mean(X,4), 1, 1, 1, 3);
end

function a = toRed3D(X)
    a = cat(4, X, zeros([size(X), 2]));
end

function a = flip_aff(M)
    a = M(end:-1:1, end:-1:1, end:-1:1, :);
end