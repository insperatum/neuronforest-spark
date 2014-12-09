function evaluate_predictions(file)
    load([file '/dimensions.txt']);
    load([file '/labels.txt']);
    load([file '/predictions.txt']);

    flip_aff = @(M) M(end:-1:1, end:-1:1, end:-1:1, :);
    % FILE IS IN ROW MAJOR ORDER, BUT MATLAB USES COLUMN MAJOR ORDER
    affTrue = permute(reshape(labels, [fliplr(dimensions) 3]), [3,2,1,4]);
    affEst = permute(reshape(predictions, [fliplr(dimensions) 3]), [3,2,1,4]);

    thresholds = 0:0.02:1;
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
        r_tp = [r_tp stats.truePos/stats.pos];
        r_fp = [r_fp stats.falsePos/stats.neg];

        pixelError = 1-sum((affEst(:)>threshold)==affTrue(:))/numel(affTrue);
        p_err = [p_err pixelError];
        p_tp = [p_tp sum((affEst(:)>threshold) & affTrue(:))/sum(affTrue(:))];
        p_fp = [p_fp sum((affEst(:)>threshold) & ~affTrue(:))/sum(~affTrue(:))];
        fprintf('Threshold = %f, Rand Error = %f, Pixel Error = %f\n', threshold, 1-ri, pixelError)
    end

    subplot(2,2,1);
    plot(thresholds, r_err);
    title('Rand Error');
    xlabel('Threshold');
    ylabel('Rand Error');
    xlim([0, 1]);
    ylim([0, 1]);

    subplot(2,2,2);
    plot(r_fp, r_tp);
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
    plot(p_fp, p_tp);
    title('Pixel Error ROC');
    xlabel('False Positive');
    ylabel('True Positive');
    xlim([0, 1]);
    ylim([0, 1]);


    [~, idx] = min(r_err);
    best_threshold = thresholds(idx);
    fprintf('Best Threshold for Rand Error: %f\n', best_threshold);
    %compEst = flip_aff(connectedComponents(flip_aff(affEst)>best_threshold));
    %BrowseComponents('icc', affEst, compEst, compTrue);
end