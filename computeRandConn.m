function [fscore,prec,rec] = computeRandConn(gt,conn,thresholds)
% [fscore,prec,rec] = computeRand(gt,conn,thresholds)
% gt: groundtruth segmentation
% conn: connectivity graph
% thresholds: vector of thresholds to evaluate

    if isexist('thresholds','var') || isempty(thresholds),
        thresholds = 0:0.05:1.0;
    end
    nTh = length(thresholds);

    prec = zeros(1,nTh);
    rec = zeros(1,nTh);
    for th = 1:nTh,
        mrkr = connectedComponents(conn>thresholds(th));
        seg = markerWatershed(conn,[],mrkr);
        [prec(th),rec(th)] = get_prec_rec_rand_n2(gt,seg);
    end
    fscore = 2*prec.*rec/(prec+rec);

    function [prec, rec] = get_prec_rec_rand_n2_srini( segA, segB )
        % segA is the ground truth segmentation and segB is the prediction

        n = sum(segA(:)>0);

        segA = double(segA)+1;
        segB = double(segB)+1;

        n_labels_A = max(segA(:));
        n_labels_B = max(segB(:));

        % compute overlap matrix
        p_ab = sparse(segA(:),segB(:),1/n,n_labels_A,n_labels_B);

        % a_i
        p_a = sum(p_ab(2:end,:), 2);

        % b_j
        p_b = sum(p_ab(2:end,2:end), 1);

        p_i0 = p_ab(2:end,1);   % pixels marked as BG in segB which are not BG in segA
        p_ab = p_ab(2:end,2:end);

        sumA2 = sum(p_a.*p_a);
        sumB2 = sum(p_b.*p_b) +  sum(p_i0)/n;
        sumAB2 = sum(sum(p_ab.^2)) + sum(p_i0)/n;
        %re = full(sumA + sumB - 2*sumAB);

        % precision
        prec = full(sumAB2 / sumB2);

        % recall
        rec = full(sumAB2 / sumA2);
    end

end