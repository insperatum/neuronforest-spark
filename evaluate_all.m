%11 trees, 0 offsets;  11 trees, -2,0,2;   11 iterations, 0 offsets;
for expt = {'2015-03-27 03-42-56/predictions', ...
            '2015-03-27 05-12-31/predictions'}
    dir1 = ['/masters_predictions/' expt{1}];
    files1 = dir(dir1);
    for i = 3:length(files1)
        partial = files1(i).name;
        dir2 = [dir1 '/' partial];
        files2 = dir(dir2);
        for j = 4 %3:length(files2)           
            depth = files2(j).name;
            dir3 = [dir2 '/' depth];
            files3 = dir(dir3);
            for k = 3:length(files3)
                trainortest = files3(k).name;
                root = [dir3 '/' trainortest];

                if(exist([root '/errors.mat'], 'file') == 2)
                    fprintf([root ' already evaluated. ignoring\n']);
                else
                    %%
                    description = fileread([root '/0/description.txt']);

                    fprintf('\nExperiment: %s\n', root);
                    files4 = dir(root);
                    a = {};
                    for l=3:length(files4)
                        if(files4(l).isdir)
                            a = [a [root '/' files4(l).name]];
                        end
                    end
                    dims = get_dimensions(a);
                    evaluate_predictions(a, dims, description)
                    %%
                end
            end
        end
    end
end