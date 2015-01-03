function [ affTrue, affEst, dimensions ] = load_affs( file, dims )
%     load([file '/dimensions.txt']);
    fid = fopen([file '/labels.raw'], 'r', 'ieee-be');
    labels = fread(fid, Inf, 'float');
    fclose(fid);
    for k=1:size(dims, 1)
       if(length(labels) == 3 * prod(dims(k, :)))
           dimensions = dims(k, :);
       end
    end
    fid = fopen([file '/predictions.raw'], 'r', 'ieee-be');
    predictions = fread(fid, 3*prod(dimensions), 'float');
    fclose(fid);
%     load([file '/labels.txt']);
%     load([file '/predictions.txt']);

    % FILE IS IN ROW MAJOR ORDER, BUT MATLAB USES COLUMN MAJOR ORDER
    affTrue = permute(reshape(labels, [3 fliplr(dimensions)]), [4,3,2,1]);
    affEst = permute(reshape(predictions, [3 fliplr(dimensions)]), [4,3,2,1]);
end

