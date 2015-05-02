for expt = {'2015-04-30 12-37-56'}
    dir1 = ['/masters_predictions/' expt{1} '/predictions'];
    files1 = dir(dir1);
    for i = 3:length(files1)
        partial = files1(i).name;
        dir2 = [dir1 '/' partial];
        files2 = dir(dir2);
        for j = 3:length(files2)           
            depth = files2(j).name;
            dir3 = [dir2 '/' depth];
            if(exist([dir3 '/test/errors_new_threshold.txt'], 'file') == 2)
                fprintf('Ignoring %s because already evaluated\n', dir3);
            else
                try 
                    %(exist([dir3 '/train/errors_new.txt'], 'file') == 2)
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
                    try
                        evaluate_predictions_with_threshold(a, dims, description, threshold)
                    catch
                        fprintf('Failed to evaluate???\n')
                    end
                catch
                    fprintf('Ignoring %s because no threshold\n', dir3);
                end
            end
        end
    end
end