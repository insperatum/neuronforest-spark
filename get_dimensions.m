function [ dims ] = get_dimensions( files )
dims = [];
for i=1:length(files)
    file = files{i};
    load([file '/dimensions.txt']);
    dims = [dims; dimensions];
end
end

