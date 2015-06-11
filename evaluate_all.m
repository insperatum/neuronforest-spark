for expt = {'expt-2015-06-06 15-08-16'}
    dir1 = ['/isbi_predictions/' expt{1} '/predictions'];
    files1 = dir(dir1);
    for i = 3:length(files1)
        partial = files1(i).name;
        dir2 = [dir1 '/' partial];
%         files2 = dir(dir2);
%         for j = 3:length(files2)           
%             depth = files2(j).name;
%             dir3 = [dir2 '/' depth];
            dir3 = dir2;
            files3 = dir(dir3);
%             for k = 3:length(files3)
%                 trainortest = files3(k).name;
                trainortest = 'test';
                root = [dir3 '/' trainortest];
                 if(exist([root '/errors_new.mat'], 'file') == 2)
                    fprintf([root ' already evaluated. ignoring\n']);
                 else
                    fprintf('\nExperiment: %s\n', root);
                    files4 = dir(root);
                    a = {};
                    for l=3:length(files4)
                        if(files4(l).isdir)
                            a = [a [root '/' files4(l).name]];
                            description = fileread([root '/' files4(l).name '/description.txt']);
                        end
                    end
                    dims = get_dimensions(a);
                    %try 
                        evaluate_predictions(a, dims, description)
                    %catch
                    %    fprintf('Failed to evaluate???\n')
                    %end
                 end
%             end
%         end
    end
end