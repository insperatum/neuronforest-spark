for expt = {'2015-04-02 15-27-28', ...
            '2015-04-06 12-31-04', ...
            '2015-04-06 14-04-13', ...
            '2015-04-06 17-59-35', ...
            '2015-04-02 00-47-01', ...
            '2015-04-01 23-34-27', ...
            '2015-04-05 14-54-08', ...
            '2015-04-05 16-01-43'
            }
    dir1 = ['/masters_predictions/' expt{1} '/predictions'];
    files1 = dir(dir1);
    for i = 3:length(files1)
        partial = files1(i).name;
        dir2 = [dir1 '/' partial];
        files2 = dir(dir2);
        for j = 3:length(files2)           
            depth = files2(j).name;
            dir3 = [dir2 '/' depth];
            fprintf('\nExperiment: %s\n', dir3);
            
            description = fileread([dir3 '/train/errors_new.txt']);
            tokens = regexp(description, 'Best Threshold for Rand F-score: ([0-9\.]*)', 'tokens');
            threshold = str2double(tokens{1});
            
            root = [dir3 '/test'];
            files4 = dir(root);
            a = {};
            for l=3:length(files4)
                if(files4(l).isdir)
                    a = [a [root '/' files4(l).name]];
                end
            end
            dims = get_dimensions(a);
            evaluate_predictions_with_threshold(a, dims, description, threshold)
        end
    end
end