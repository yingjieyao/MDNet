function [ result ] = mdnet_run(images, region, net, display)
% MDNET_RUN
% Main interface for MDNet tracker
%
% INPUT:
%   images  - 1xN cell of the paths to image sequences
%   region  - 1x4 vector of the initial bounding box [left,top,width,height]
%   net     - The path to a trained MDNet
%   display - True for displying the tracking result
%
% OUTPUT:
%   result - Nx4 matrix of the tracking result Nx[left,top,width,height]
%
% Hyeonseob Nam, 2015
%

if(nargin<4), display = true; end

%% Initialization
fprintf('Initialization...\n');

nFrames = length(images);

img = imread(images{1});
if(size(img,3)==1), img = cat(3,img,img,img); end
targetLoc = region;
result = zeros(nFrames, 4); result(1,:) = targetLoc;

[net_conv, net_fc, opts] = mdnet_init(img, net);

%% Train a bbox regressor
if(opts.bbreg)
    % 得到第一帧的样本, 这些样本是在第一帧附近的随机取样的结果, 他们的大小有变化，长宽比也有变化
    % 这是用来训练线性回归分类器的

    % 这里先随机取了k*10个样本，然后剔除其中的负样本，最后从中随机取k个,下同
    pos_examples = gen_samples('uniform_aspect', targetLoc, opts.bbreg_nSamples*10, opts, 0.3, 10);
    r = overlap_ratio(pos_examples,targetLoc);
    % 与第一帧的目标交叠比大于0.6，则为正样本
    pos_examples = pos_examples(r>0.6,:);
    % 在这些正样本里面随机取样，得到了bbreg_nSamples个正样本
    pos_examples = pos_examples(randsample(end,min(opts.bbreg_nSamples,end)),:);

    % 提取第一帧的样本的特征，conv3, 3 * 3 * 512 * sizeof(pos_examples, 1)
    feat_conv = mdnet_features_convX(net_conv, img, pos_examples, opts);

    X = permute(gather(feat_conv),[4,3,1,2]);

    % size X = sizeof(pos_examples, 1) * 4608
    X = X(:,:);
    bbox = pos_examples;
    bbox_gt = repmat(targetLoc,size(pos_examples,1),1);
    % 训练基于第一帧的linear regression
    % TODO:train_bbox_regressor 的代码没看
    bbox_reg = train_bbox_regressor(X, bbox, bbox_gt);
end

%% Extract training examples
fprintf('  extract features...\n');

% draw positive/negative samples
% 这里判断正样本还是负样本的阈值要比上面高, 0.7, 0.5
pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_init*2, opts, 0.1, 5);
r = overlap_ratio(pos_examples,targetLoc);
pos_examples = pos_examples(r>opts.posThr_init,:);
pos_examples = pos_examples(randsample(end,min(opts.nPos_init,end)),:);

% gen_samples whole, sample from the whole image for negative samples
neg_examples = [gen_samples('uniform', targetLoc, opts.nNeg_init, opts, 1, 10);...
    gen_samples('whole', targetLoc, opts.nNeg_init, opts)];
r = overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_init,end)),:);

examples = [pos_examples; neg_examples];
pos_idx = 1:size(pos_examples,1);
neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

% extract conv3 features
feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
pos_data = feat_conv(:,:,:,pos_idx);
neg_data = feat_conv(:,:,:,neg_idx);

% 得到了基于第一帧的目标区域得到的正负样本特征，pos_data, neg_data


%% Learning CNN
fprintf('  training cnn...\n');
%% fine-tuning 最后几层
% TODO: mdnet_finetune_hnm代码没有读
net_fc = mdnet_finetune_hnm(net_fc,pos_data,neg_data,opts,...
    'maxiter',opts.maxiter_init,'learningRate',opts.learningRate_init);

%% Initialize displayots
if display
    figure(2);
    set(gcf,'Position',[200 100 600 400],'MenuBar','none','ToolBar','none');

    hd = imshow(img,'initialmagnification','fit'); hold on;
    rectangle('Position', targetLoc, 'EdgeColor', [1 0 0], 'Linewidth', 3);
    set(gca,'position',[0 0 1 1]);

    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;
end

%% Prepare training data for online update
total_pos_data = cell(1,1,1,nFrames);
total_neg_data = cell(1,1,1,nFrames);

% 在这里，生成的负样本的参数跟上面生成负样本的参数不一样, trans_f, scale_f
neg_examples = gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
r = overlap_ratio(neg_examples,targetLoc);
neg_examples = neg_examples(r<opts.negThr_init,:);
neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);

examples = [pos_examples; neg_examples];
pos_idx = 1:size(pos_examples,1);
neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
total_pos_data{1} = feat_conv(:,:,:,pos_idx);
total_neg_data{1} = feat_conv(:,:,:,neg_idx);

success_frames = 1;
trans_f = opts.trans_f;
scale_f = opts.scale_f;

%% Main loop
for To = 2:nFrames;
    fprintf('Processing frame %d/%d... ', To, nFrames);

    img = imread(images{To});
    if(size(img,3)==1), img = cat(3,img,img,img); end

    spf = tic;
    %% Estimation
    % draw target candidates
    samples = gen_samples('gaussian', targetLoc, opts.nSamples, opts, trans_f, scale_f);
    feat_conv = mdnet_features_convX(net_conv, img, samples, opts);

    % evaluate the candidates
    feat_fc = mdnet_features_fcX(net_fc, feat_conv, opts);
    feat_fc = squeeze(feat_fc)';
    [scores,idx] = sort(feat_fc(:,2),'descend');

    % NOTE: 得到排名前五的samples的score，然后用这些score求个平均值来得到最后的targetLoc
    target_score = mean(scores(1:5));
    targetLoc = round(mean(samples(idx(1:5),:)));

    % final target
    % 这个是box regression之前的位置，仅仅根据网络的输出来确定
    result(To,:) = targetLoc;

    % extend search space in case of failure
    % 如果跟踪失败的话，增大搜索的区域
    if(target_score<0)
        trans_f = min(1.5, 1.1*trans_f);
    else
        trans_f = opts.trans_f;
    end

    % bbox regression
    if(opts.bbreg && target_score>0)
        X_ = permute(gather(feat_conv(:,:,:,idx(1:5))),[4,3,1,2]);
        X_ = X_(:,:);
        % 取分数最前面的五个去做regression
        bbox_ = samples(idx(1:5),:);
        % TODO: predict_bbox_regressor代码没看
        pred_boxes = predict_bbox_regressor(bbox_reg.model, X_, bbox_);
        % 这个结果是根据regression过程算出来的
        result(To,:) = round(mean(pred_boxes,1));
    end

    %% Prepare training data
    if(target_score>0)
        pos_examples = gen_samples('gaussian', targetLoc, opts.nPos_update*2, opts, 0.1, 5);
        r = overlap_ratio(pos_examples,targetLoc);
        pos_examples = pos_examples(r>opts.posThr_update,:);
        pos_examples = pos_examples(randsample(end,min(opts.nPos_update,end)),:);

        neg_examples = gen_samples('uniform', targetLoc, opts.nNeg_update*2, opts, 2, 5);
        r = overlap_ratio(neg_examples,targetLoc);
        neg_examples = neg_examples(r<opts.negThr_update,:);
        neg_examples = neg_examples(randsample(end,min(opts.nNeg_update,end)),:);

        examples = [pos_examples; neg_examples];
        pos_idx = 1:size(pos_examples,1);
        neg_idx = (1:size(neg_examples,1)) + size(pos_examples,1);

        feat_conv = mdnet_features_convX(net_conv, img, examples, opts);
        % 这个是一个cell，第t个位置保存着第t帧图片的样本的特征（正负）
        total_pos_data{To} = feat_conv(:,:,:,pos_idx);
        total_neg_data{To} = feat_conv(:,:,:,neg_idx);

        % success_frames队列后面加入当前帧
        success_frames = [success_frames, To];
        % 移除区间内跟踪成功帧的最前面的那个
        if(numel(success_frames)>opts.nFrames_long)
            total_pos_data{success_frames(end-opts.nFrames_long)} = single([]);
        end
        if(numel(success_frames)>opts.nFrames_short)
            total_neg_data{success_frames(end-opts.nFrames_short)} = single([]);
        end
    else
        % 如果跟踪失败，那么当前帧没有可以用来训练的样本及特征
        total_pos_data{To} = single([]);
        total_neg_data{To} = single([]);
    end

    %% Network update
    if((mod(To,opts.update_interval)==0 || target_score<0) && To~=nFrames)
        if (target_score<0) % short-term update
            pos_data = cell2mat(total_pos_data(success_frames(max(1,end-opts.nFrames_short+1):end)));
        else % long-term update
            pos_data = cell2mat(total_pos_data(success_frames(max(1,end-opts.nFrames_long+1):end)));
        end
        % 不管是长期更新还是短期更新，用的负样本都是短期的负样本
        % 而长期更新或者短期更新都只更新一种
        neg_data = cell2mat(total_neg_data(success_frames(max(1,end-opts.nFrames_short+1):end)));

%         fprintf('\n');
        % update过程
        [net_fc] = mdnet_finetune_hnm(net_fc,pos_data,neg_data,opts,...
            'maxiter',opts.maxiter_update,'learningRate',opts.learningRate_update);
    end

    spf = toc(spf);
    fprintf('%f seconds\n',spf);

    %% Display
    if display
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',img); hold on;

        rectangle('Position', result(To,:), 'EdgeColor', [1 0 0], 'Linewidth', 3);
        set(gca,'position',[0 0 1 1]);

        text(10,10,num2str(To),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
        hold off;
        drawnow;
    end
end
