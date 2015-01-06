%11 trees, 0 offsets;  11 trees, -2,0,2;   11 iterations, 0 offsets;
for expt = {'s3/2015-01-04 14-19-09/predictions', ...
            's3/2015-01-04 15-39-23/predictions', ...
            's3/2015-01-04 18-34-59/predictions', ...
            's3/2015-01-04 22-04-13/predictions'}
    dir1 = ['/masters_predictions/' expt{1}];
    files1 = dir(dir1);
    for i = 3:length(files1)
        partial = files1(i).name;
        dir2 = [dir1 '/' partial];
        files2 = dir(dir2);
        for j = 3:length(files2)           
            trainortest = files2(j).name;
            root = [dir2 '/' trainortest];
            
            if(exist([root '/errors.mat'], 'file') == 2)
                fprintf([root ' already evaluated. ignoring\n']);
            else
                %%
                description = fileread([root '/0/description.txt']);

                fprintf('\nExperiment: %s\n', root);
                files3 = dir(root);
                a = {};
                for k=3:length(files3)
                    if(files3(k).isdir)
                        a = [a [root '/' files3(k).name]];
                    end
                end
                dims = get_dimensions(a);
                evaluate_predictions(a, dims, description)
                %%
            end
        end
    end
end