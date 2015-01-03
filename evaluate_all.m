for expt = {'s3/2015-01-02 18-28-33', 's3/2015-01-02 19-01-34', 's3/2015-01-02 19-23-55'}
    dir1 = ['/masters_predictions/' expt{1}];
    files1 = dir(dir1);
    for i = 3:length(files1)
        partial = files1(i).name;
        dir2 = [dir1 '/' partial];
        files2 = dir(dir2);
        for j = 3:length(files2)
            trainortest = files2(j).name;
            root = [dir2 '/' trainortest];
            fdesc = fopen([root '/0/description.txt']);
            description = textscan(fdesc, '%s', 'Delimiter', '\n');
            description = description{1};
            description = description{1};
            fclose(fdesc);

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
        end
    end
end