function evaluate_predictions_with_threshold(files, dims, description, threshold)
    f = fopen([files{1} '/../errors_new_threshold.txt'], 'w');
    saveAndPrint(f, 'Description:\n%s\n\n', description);
    saveAndPrint(f, 'Using Threshold:%0.3f\n\n', threshold);
    
    [r_thresholds, r_fscore, r_tp, r_fp, r_pos, r_neg, ~] = ...
        evaluate_thresholds(files, dims, threshold, 'rand', 0);
    
    [p_thresholds, p_err, p_tp, p_fp, p_pos, p_neg, p_sqerr] = ...
        evaluate_thresholds(files, dims, threshold, 'pixel', 0);
    
    e = r_fscore;
    saveAndPrint(f, 'Rand F-score: %f\n', e);
    
    e = p_err;
    saveAndPrint(f, 'Pixel Error: %f\n', e);
    
    fclose(f);
end

function saveAndPrint(varargin)
    file = varargin{1};
    fprintf(varargin{2:end});
    fprintf(file, varargin{2:end});
end

function [thresholds, err, tp, fp, pos, neg, p_sqerr] = ...
evaluate_thresholds(files, dims, thresholds, randOrPixel, min_step)
    err = nan(length(files), length(thresholds));
    tp = nan(length(files), length(thresholds));
    fp = nan(length(files), length(thresholds));
    pos = nan(length(files), 1);
    neg = nan(length(files), 1);
    p_sqerr = nan(length(files), 1);
    n_examples = nan(length(files), 1);
    %[err, tp, fp, pos, neg, p_sqerr] = get_stats(files{1}, thresholds, dims, randOrPixel);
    parfor i=1:length(files)
        try 
            [err_, tp_, fp_, pos_, neg_, p_sqerr_, n_examples_] = get_stats(files{i}, thresholds, dims, randOrPixel);
            err(i,:) = err_;
            tp(i,:) = tp_;
            fp(i,:) = fp_;
            pos(i) = pos_;
            neg(i)= neg_;
            p_sqerr(i) = p_sqerr_;
            n_examples(i) = n_examples_;
        catch
            err(i,:) = 0;
            tp(i,:) = 0;
            fp(i,:) = 0;
            pos(i) = 0;
            neg(i)= 0;
            p_sqerr(i) = 0;
            n_examples(i) = 0;
        end
%          fprintf(':O');
    end
    
    tp = sum(bsxfun(@times, tp, n_examples));
    fp = sum(bsxfun(@times, fp, n_examples));
    pos = sum(pos .* n_examples);
    neg = sum(neg .* n_examples);
    p_sqerr = sum(p_sqerr .* n_examples) / sum(n_examples);
    if(strcmp(randOrPixel,'rand'))
        prec = tp ./ (tp + fp);
        rec = tp / pos;
        err = 2 * (prec .* rec) ./ (prec + rec);
        [best_err, idx] = max(err);
    else
        err = sum(bsxfun(@times, err, n_examples)) ./ sum(n_examples);
        [best_err, idx] = min(err);
    end

    if(length(thresholds)>1)
        step = thresholds(2) - thresholds(1);
        best_threshold = thresholds(idx);
        fprintf('Thresholds = %f:%f:%f, Best %s = %f\n', thresholds(1), step, thresholds(end), randOrPixel, best_err);

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
end


function [err, tp, fp, pos, neg, p_sqerr, n_examples] = get_stats(file, thresholds, dims, randOrPixel)
    
    [ affTrue, affEst, dimensions ] = load_affs( file, dims );
    n_examples = numel(affEst)/3;
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
    pos = NaN;
    neg = NaN;
    for threshold=thresholds
        if(strcmp(randOrPixel,'rand'))
            [err_, tp_, fp_, pos_, neg_] = ...
                get_rand_stats_for_threshold(compTrue, affEst, threshold);
            err = [err err_];
            tp = [tp tp_];
            fp = [fp fp_];
            if(~isnan(pos_)) pos=pos_; end
            if(~isnan(neg_)) neg=neg_; end
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

    if(~ all(affEst(:)<=threshold) && ~ all(affEst(:)>threshold))
        compEst = flip_aff(connectedComponents(flip_aff(affEst)>threshold));

        watershed = markerWatershed(affEst, -eye(3), compEst);

        [ri, stats] = randIndex(compTrue, watershed);
        r_err = 1-ri;
        r_tp = stats.truePos;
        r_fp = stats.falsePos;

        r_pos = stats.pos;
        r_neg = stats.neg;
%         r_fscore = 2 * (stats.prec * stats.rec) / (stats.prec + stats.rec);
    else 
        r_err = NaN;
        r_tp = NaN;
        r_fp = NaN;

        r_pos = NaN;
        r_neg = NaN;
%         r_fscore = -1;
    end
end