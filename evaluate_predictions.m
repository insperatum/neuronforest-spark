function evaluate_predictions(files, dims, description)
    initial_thresholds = -2:0.5:3;
    min_step = 0.001;
    
    f = fopen([files{1} '/../errors.txt'], 'w');
    saveAndPrint(f, 'Description:\n%s\n\n', description);
    
    [r_thresholds, r_err, r_tp, r_fp, r_pos, r_neg, ~] = ...
        evaluate_thresholds(files, dims, initial_thresholds, 'rand', min_step);
    
    [p_thresholds, p_err, p_tp, p_fp, p_pos, p_neg, p_sqerr] = ...
        evaluate_thresholds(files, dims, initial_thresholds, 'pixel', min_step);
    
    saveAndPrint(f, 'Mean Pixel Square Error: %f\n', p_sqerr);
    
    [e, idx] = min(r_err);
    best_threshold = r_thresholds(idx);
    saveAndPrint(f, 'Best Threshold for Rand Error: %f\n', best_threshold);
    saveAndPrint(f, 'Best Rand Error: %f\n', e);
    
    [e, idx] = min(p_err);
    best_threshold = p_thresholds(idx);
    saveAndPrint(f, 'Best Threshold for Pixel Error: %f\n', best_threshold);
    saveAndPrint(f, 'Best Pixel Error: %f\n', e);
    
    
    min_threshold_idx = (length(initial_thresholds)+1)/2;
    max_threshold_idx = length(r_thresholds) + 1 - min_threshold_idx;
    
    clf;
    plt = subplot(2,2,1);
    plot(r_thresholds, r_err);
    title('Rand Error');
    xlabel('Threshold');
    ylabel('Rand Error');
    xlim([r_thresholds(min_threshold_idx), r_thresholds(max_threshold_idx)]);
    ylim([0, 1]);
    
    subplot(2,2,2);
    plot(r_fp/r_neg, r_tp/r_pos);
    title('Rand Error ROC');
    xlabel('False Positive');
    ylabel('True Positive');
    xlim([0, 1]);
    ylim([0, 1]);
    
    subplot(2,2,3);
    plot(p_thresholds, p_err);
    title('Pixel Error');
    xlabel('Threshold');
    ylabel('Pixel Error');
    xlim([p_thresholds(min_threshold_idx), p_thresholds(max_threshold_idx)]);
    ylim([0, 1]);

    subplot(2,2,4);
    plot(p_fp/p_neg, p_tp/p_pos);
    title('Pixel Error ROC');
    xlabel('False Positive');
    ylabel('True Positive');
    xlim([0, 1]);
    ylim([0, 1]);
    
    saveas(plt, [files{1} '/../errors.fig']);
    saveas(plt, [files{1} '/../errors.png'], 'png');
    
    fclose(f);
    
    save([files{1} '/../errors.mat'], ...
        'r_thresholds', 'r_err', 'r_tp', 'r_fp', 'r_pos', 'r_neg', ...
        'p_thresholds', 'p_err', 'p_tp', 'p_fp', 'p_pos', 'p_neg', 'p_sqerr');
end

function saveAndPrint(varargin)
    file = varargin{1};
    fprintf(varargin{2:end});
    fprintf(file, varargin{2:end});
end

function [thresholds, err, tp, fp, pos, neg, p_sqerr] = ...
evaluate_thresholds(files, dims, thresholds, randOrPixel, min_step)
    err = zeros(length(files), length(thresholds));
    tp = zeros(length(files), length(thresholds));
    fp = zeros(length(files), length(thresholds));
    pos = zeros(length(files), 1);
    neg = zeros(length(files), 1);
    p_sqerr = zeros(length(files), 1);
    %[err, tp, fp, pos, neg, p_sqerr] = get_stats(files{1}, thresholds, dims, randOrPixel);
    parfor i=1:length(files)
        [err_, tp_, fp_, pos_, neg_, p_sqerr_] = get_stats(files{i}, thresholds, dims, randOrPixel);
        err(i,:) = err_;
        tp(i,:) = tp_;
        fp(i,:) = fp_;
        pos(i) = pos_;
        neg(i) = neg_;
        p_sqerr(i) = p_sqerr_;
    end
    err = sum(err) / length(files); % TODO WEIGHT BY NUMBER OF EXAMPLES
    tp = sum(tp);
    fp = sum(fp);
    pos = sum(pos);
    neg = sum(neg);
    p_sqerr = sum(p_sqerr) / length(files); % TODO WEIGHT BY NUMBER OF EXAMPLES
    

    step = thresholds(2) - thresholds(1);
    [best_err, idx] = min(err);
    best_threshold = thresholds(idx);
    fprintf('Thresholds = %f:%f:%f, Best %s error = %f\n', thresholds(1), step, thresholds(end), randOrPixel, best_err);

    if(step > min_step)
        new_step = 2 * step/(length(thresholds)-1);
        inner_thresholds = best_threshold-step:new_step:best_threshold+step;
        [thresholds_, err_, tp_, fp_, ~, ~, ~] = ...
            evaluate_thresholds(files, dims, inner_thresholds, randOrPixel, min_step);
        
        thresholds = [thresholds(1:idx-1), thresholds_, thresholds(idx+1:end)];
        err = [err(1:idx-1), err_, err(idx+1:end)];
        tp = [tp(1:idx-1), tp_, tp(idx+1:end)];
        fp = [fp(1:idx-1), fp_, fp(idx+1:end)];
    else
        fprintf('\n');
    end
end


function [err, tp, fp, pos, neg, p_sqerr] = get_stats(file, thresholds, dims, randOrPixel)
    
    [ affTrue, affEst, dimensions ] = load_affs( file, dims );
    
%     load([file '/gradients.txt'])
%     [~, idxs_idxs] = sort(sum(gradients(:,2:4), 2));
%     idxs = gradients(idxs_idxs, 1);
%     foo = sum(predictions,2)/3;
%     foo(idxs(1)) = 3;
%     bar = permute(reshape(foo, fliplr(dimensions)), [3,2,1]);
%     %BrowseComponents('ii', affEst, bar);
%     
%     load('/home/luke/Documents/asdf1/seg.txt')
%     df = int32(seg);
%     load('/home/luke/Documents/asdf2/seg.txt')
%     dt = int32(seg);
%     asdf = zeros(size(labels, 1), 1);
%     asdf(df(:, 1) + 1) = 1;
%     asdf(dt(:, 1) + 1) = 2;
%     asdf = permute(reshape(asdf, fliplr(dimensions)), [3,2,1]);
%     
%     load([file '/seg.txt'])
%     seg = int32(seg);
%     segs = zeros(size(seg, 1), 1);
%     segs(seg(:, 1) + 1) = seg(:, 2);
%     segs = permute(reshape(segs, fliplr(dimensions)), [3,2,1]);
%     BrowseComponents('ic', bar, asdf)
    
    if(strcmp(randOrPixel,'rand'))
        compTrue = flip_aff(connectedComponents(flip_aff(affTrue)));   
        p_sqerr = -1;
    else
        p_sqerr = (affEst(:) - affTrue(:))' * (affEst(:) - affTrue(:)) / numel(affEst);
    end
    
    err = [];
    tp = [];
    fp = [];
    
    for threshold=thresholds
        if(strcmp(randOrPixel,'rand'))
            [err_, tp_, fp_, pos, neg] = ...
                get_rand_stats_for_threshold(compTrue, affEst, threshold);
            err = [err err_];
            tp = [tp tp_];
            fp = [fp fp_];
        else
            [err_, tp_, fp_, pos, neg] = ...
                get_pixel_stats_for_threshold(affTrue, affEst, threshold);
            err = [err err_];
            tp = [tp tp_];
            fp = [fp fp_];
        end
    end
end

function a = flip_aff(M)
    a = M(end:-1:1, end:-1:1, end:-1:1, :);
end

function [p_err, p_tp, p_fp, p_pos, p_neg] = ...
        get_pixel_stats_for_threshold(affTrue, affEst, threshold)
    p_err = 1-sum((affEst(:)>threshold)==affTrue(:))/numel(affTrue);
    p_tp = sum((affEst(:)>threshold) & affTrue(:));
    p_fp = sum((affEst(:)>threshold) & ~affTrue(:));

    p_pos = sum(affTrue(:));
    p_neg = sum(~affTrue(:));
end

function [r_err, r_tp, r_fp, r_pos, r_neg] = ...
        get_rand_stats_for_threshold(compTrue, affEst, threshold)

    compEst = flip_aff(connectedComponents(flip_aff(affEst)>threshold));
    [ri, stats] = randIndex(compTrue, compEst);
    r_err = 1-ri;
    r_tp = stats.truePos;
    r_fp = stats.falsePos;

    r_pos = stats.pos;
    r_neg = stats.neg;
end