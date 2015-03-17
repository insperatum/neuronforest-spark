for trainOrTest = {'train', 'test'}
    trainOrTest = trainOrTest{1};
    roots = {...
        '/masters_predictions/s3/2015-01-04 14-19-09/predictions/partial5', ...
        '/masters_predictions/s3/2015-01-04 15-39-23/predictions/partial5'
        };
    clear results;
    for i = 1:length(roots)
        results(i) = load([roots{i} '/' trainOrTest '/errors.mat']);
    end

    cols = 'rgb';
    figure;

    for i = 1:length(results)
        result = results(i);
        plot(result.r_thresholds, result.r_err, cols(i));
        title(['Rand Error (' trainOrTest ')']);
        xlabel('Threshold');
        ylabel('Rand Error');
        xlim([0, 1]);
        ylim([0, 1]);
        hold on;
    end
    legend('No Offsets', '27 Offsets, Spacing=2');

    figure;
    for i = 1:length(results)
        result = results(i);
        plot(result.r_fp/result.r_neg, result.r_tp/result.r_pos, cols(i));
        title(['Rand Error ROC (' trainOrTest ')']);
        xlabel('False Positive');
        ylabel('True Positive');
        xlim([0, 1]);
        ylim([0, 1]);
        hold on;
    end
    legend('No Offsets', '27 Offsets, Spacing=2');
%{
    figure;
    for i = 1:length(results)
        result = results(i);
        plot(result.p_thresholds, result.p_err, cols(i));
        title(['Pixel Error (' trainOrTest ')']);
        xlabel('Threshold');
        ylabel('Pixel Error');
        xlim([0, 1]);
        ylim([0, 1]);
        hold on;
    end
    legend('Vanilla RF', 'Gradient Boosting with MALIS');

    figure;
    for i = 1:length(results)
        result = results(i);
        plot(result.p_fp/result.p_neg, result.p_tp/result.p_pos, cols(i));
        title(['Pixel Error ROC (' trainOrTest ')']);
        xlabel('False Positive');
        ylabel('True Positive');
        xlim([0, 1]);
        ylim([0, 1]);
        hold on;
    end
    legend('Vanilla RF', 'Gradient Boosting with MALIS');
%}
end