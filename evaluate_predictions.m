function evaluate_predictions(files)
    thresholds = 0:0.02:1;
    [r_err, r_tp, r_fp, r_pos, r_neg, p_err, p_tp, p_fp, p_pos, p_neg] = get_stats(files{1}, thresholds);
    for i=2:length(files)
        [r_err_, r_tp_, r_fp_, r_pos_, r_neg_, p_err_, p_tp_, p_fp_, p_pos_, p_neg_] = get_stats(files{1}, thresholds);
        r_err = r_err + r_err_;
        r_tp = r_tp + r_tp_;
        r_fp = r_fp + r_fp_;
        r_pos = r_pos + r_pos_;
        r_neg = r_neg + r_neg_;
        p_err = p_err + p_err_;
        p_tp = p_tp + p_tp_;
        p_fp = p_fp + p_fp_;
        p_pos = p_pos + p_pos_;
        p_neg = p_neg + p_neg_;
    end
    r_err = r_err / length(files);
    p_err = p_err / length(files);
    
    [~, idx] = min(r_err);
    best_threshold = thresholds(idx);
    fprintf('Best Threshold for Rand Error: %f\n', best_threshold);
    [~, idx] = min(p_err);
    best_threshold = thresholds(idx);
    fprintf('Best Threshold for Pixel Error: %f\n', best_threshold);
    
    subplot(2,2,1);
    plot(thresholds, r_err);
    title('Rand Error');
    xlabel('Threshold');
    ylabel('Rand Error');
    xlim([0, 1]);
    ylim([0, 1]);
    
    subplot(2,2,2);
    plot(r_fp/r_neg, r_tp/r_pos);
    title('Rand Error ROC');
    xlabel('False Positive');
    ylabel('True Positive');
    xlim([0, 1]);
    ylim([0, 1]);
    
    subplot(2,2,3);
    plot(thresholds, p_err);
    title('Pixel Error');
    xlabel('Threshold');
    ylabel('Pixel Error');
    xlim([0, 1]);
    ylim([0, 1]);

    subplot(2,2,4);
    plot(p_fp/p_neg, p_tp/p_pos);
    title('Pixel Error ROC');
    xlabel('False Positive');
    ylabel('True Positive');
    xlim([0, 1]);
    ylim([0, 1]);
end

function [r_err, r_tp, r_fp, r_pos, r_neg, p_err, p_tp, p_fp, p_pos, p_neg] = get_stats(file, thresholds)
    load([file '/dimensions.txt']);
    load([file '/labels.txt']);
    load([file '/predictions.txt']);

    flip_aff = @(M) M(end:-1:1, end:-1:1, end:-1:1, :);
    % FILE IS IN ROW MAJOR ORDER, BUT MATLAB USES COLUMN MAJOR ORDER
    affTrue = permute(reshape(labels, [fliplr(dimensions) 3]), [3,2,1,4]);
    affEst = permute(reshape(predictions, [fliplr(dimensions) 3]), [3,2,1,4]);
    
    r_err = [];
    r_tp = [];
    r_fp = [];
    p_err = [];
    p_tp = [];
    p_fp = [];
    compTrue = flip_aff(connectedComponents(flip_aff(affTrue)));
    for threshold=thresholds
        compEst = flip_aff(connectedComponents(flip_aff(affEst)>threshold));
        [ri, stats] = randIndex(compTrue, compEst);
        r_err = [r_err 1-ri];
        r_tp = [r_tp stats.truePos];
        r_fp = [r_fp stats.falsePos];

        r_pos = stats.pos;
        r_neg = stats.neg;
        
        pixelError = 1-sum((affEst(:)>threshold)==affTrue(:))/numel(affTrue);
        p_err = [p_err pixelError];
        p_tp = [p_tp sum((affEst(:)>threshold) & affTrue(:))];
        p_fp = [p_fp sum((affEst(:)>threshold) & ~affTrue(:))];
        
        p_pos = sum(affTrue(:));
        p_neg = sum(~affTrue(:));
        fprintf('Threshold = %f, Rand Error = %f, Pixel Error = %f\n', threshold, 1-ri, pixelError)
    end
end