type = 2;

tol = 2e-2;
frac = 5e-6;

if (type == 1)
    data_path = '../Data/Phantom/clustercalc/';
    load(strcat(data_path, 'train.mat'));
    load(strcat(data_path, 'val.mat'));
    load(strcat(data_path, 'test.mat'));
    signal_name = 'clustercalc_signal';
    dim = 65;
    load(strcat(data_path, 'clustercalc_signal.mat'));
    signal = clustercalc_signal;
    
    offset = 5000;
    val_offset = 625;
    labels_val = [zeros(size(val,1)/2, 1); ones(size(val,1)/2, 1)];
    labels_test = [zeros(size(test,1)/2, 1); ones(size(test,1)/2, 1)];
    val = [val(1:end/2,:); val(end/2+1:end,:)];
    test = [test(1:end/2,:); test(end/2+1:end,:)];
    
    num_images_list = [250, 500, 1000, 2000, 5000];
    
    folder = 'Test_Statistics_Clustercalc';
    
    model_path = '../Models/Phantom/clustercalc_signal';
    
    pls_filters = [20, 20, 20, 20, 20];
    conv_filters = [17, 19, 15, 20, 19];%ones(5, 1) * 12; %[14, 19, 12, 12, 12];
    conv_width = [2, 1, 1.5, 15.25, 1.5]; %ones(5, 1) * 1;%[1, 0.75, 1, 1, 1];
    ae_channels = [9, 15, 17, 20, 17];%ones(5, 1) * 17; %[6, 17, 9, 20, 15];
    fig_title = 'Clustercalc Signal';
elseif (type == 2)
    data_path = '../Data/Phantom/spicmass/';
    load(strcat(data_path, 'train.mat'));
    load(strcat(data_path, 'val.mat'));
    load(strcat(data_path, 'test.mat'));
    signal_name = 'spiculated_mass_signal';
    dim = 109;
    load(strcat(data_path, 'spiculated_mass_signal.mat'));
    signal = spiculated_mass_signal;
    
    offset = 5000;
    val_offset = 625;
    labels_val = [zeros(size(val,1)/2, 1); ones(size(val,1)/2, 1)];
    %labels_test = [zeros(size(test,1)/2, 1); ones(size(test,1)/2, 1)];
    val = [val(1:end/2,:); val(end/2+1:end,:)];
    test = [test(1:end/2,:); test(end/2+1:end,:)];
    %test = [test(1:150,:); test(end/2+1:end/2+150,:)];
    labels_test = [zeros(size(test,1)/2, 1); ones(size(test,1)/2, 1)];
    
    num_images_list = [250, 500, 1000, 2000, 5000];
    
    folder = 'Test_Statistics_Phantom';
    
    model_path = '../Models/Phantom/spiculated_mass_signal';
    
    pls_filters = [8, 13, 12, 19, 19];%20;
    conv_filters = [15, 19, 20, 6, 7];%ones(5, 1) * 19; %[14, 14, 19, 19, 19];%16;
    conv_width = [1.25, 1, 1.25, 2.25, 2.25];%ones(5, 1) * 1;%[1.25, 1.25, 1.25, 1.25, 1];%1.75;
    ae_channels = [14, 10, 17, 14, 18];%ones(5, 1) * 19; %[4, 6, 18, 19, 15];
    fig_title = 'Spiculated Mass Signal';
elseif (type == 3)
    data_path = '../Data/Lumpy/';
    load(strcat(data_path, 'train.mat'));
    load(strcat(data_path, 'val.mat'));
    load(strcat(data_path, 'test.mat'));
    signal_name = 'elliptical_signal_sy1_5';
    dim = 64;
    load(strcat(data_path, 'elliptical_signal_sy1_5.mat'));
    signal = elliptical_signal_sy1_5;
    
    offset = 30000;
    val_offset = 5000;
    labels_val = [zeros(size(val,1)/2, 1); ones(size(val,1)/2, 1)];
    labels_test = [zeros(size(test,1)/2, 1); ones(size(test,1)/2, 1)];
    %train = [train(1:end/2,:); train(end/2+1:end,:) + signal(:)'];
    %val = [val(1:end/2,:); val(end/2+1:end,:) + signal(:)'];
    %test = [test(1:end/2,:); test(end/2+1:end,:) + signal(:)'];
    
    num_images_list = [250, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000];
    
    folder = 'Test_Statistics_Elliptical';
    
    model_path = '../Models/Lumpy/Elliptical/elliptical_signal_sy1_5';
    
    pls_filters = [20, 15, 17, 9, 13, 13, 13, 15, 14, 13]; %20;
    conv_filters = [15, 16, 20, 20, 20, 20, 20, 14, 14, 13]; %ones(10, 1) * 20;
    conv_width = [30, 30, 25.75, 25.75, 25.5, 25.25, 25.25, 3, 3, 3];%ones(10, 1) * 1.75;
    ae_channels = [14, 9, 19, 18, 5, 10, 16, 7, 9, 12];%ones(10, 1) * 10;
    fig_title = 'Elliptical Signal';
elseif (type == 4)
    data_path = '../Data/Lumpy_SKS/';
    load(strcat(data_path, 'train.mat'));
    load(strcat(data_path, 'val.mat'));
    load(strcat(data_path, 'test.mat'));
    signal_name = 'mean_signal_sy1_5_rot4';
    dim = 64;
    load(strcat(data_path, 'mean_signal_sy1_5_rot4.mat'));
    signal = mean_signal_sy1_5_rot4;
    
    offset = 30000;
    val_offset = 5000;
    labels_val = [zeros(size(val,1)/2, 1); ones(size(val,1)/2, 1)];
    labels_test = [zeros(size(test,1)/2, 1); ones(size(test,1)/2, 1)];
    
    num_images_list = [250, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000];
    
    folder = 'Test_Statistics_Elliptical_SKS';
    
    model_path = '../Models/Lumpy_SKS/Elliptical_SKS/mean_signal_sy1_5_rot4';
    
    pls_filters = [20, 15, 8, 8, 12, 13, 13, 13, 13, 13];%13;
    conv_filters = [15, 15, 17, 17, 20, 20, 20, 20, 20, 20];%ones(10, 1) * 20;
    conv_width = [30, 30, 29.75, 30, 25.5, 25.5, 25.5, 25.5, 25.5, 25.5];%ones(10, 1) * 2.75;
    ae_channels = [1, 14, 16, 19, 5, 5, 12, 6, 16, 4]; %ones(10, 1) * 14;
    fig_title = 'Elliptical Signal SKS';
end

stop_setting = 'Val';

rng(1, 'twister') % for reproducability

filters = 1:20;

ae_channels = zeros(length(num_images_list), 1);
ae_val_auc = zeros(length(num_images_list), 1);

for j = 1:length(num_images_list)
    num_images = num_images_list(j);
    h0 = train(1:num_images, :);
    h1 = train(offset+1:offset+num_images, :);
    signal = mean(h1) - mean(h0);
    
    auc = zeros(length(filters), 1);
    
    for i = 1:length(filters)
        base_dir = sprintf('%s/TMI_%iLS_T_%i_%sStop/output/', model_path, i, num_images, stop_setting);
        
        W = fread(fopen([base_dir, 'W1.dat']), [i, dim*dim], 'float32');
        W = W';
        
        C0_v = cov(h0 * W);
        C1_v = cov(h1 * W);
        C_v = 0.5 * (C0_v + C1_v);
        observer_ae = (pinv(C_v) * (W' * signal(:)))' * W';
        
        thresh_ae = val*observer_ae';
        [x_ae, y_ae, t_ae, auc_ae] = perfcurve(labels_val, thresh_ae, 1);
        auc(i) = auc_ae;
        i
    end
    auc
    [ae_val_auc(j), ae_channels(j)] = max(auc);
end

hyperparams = zeros(length(num_images_list), 4);
sizes = 0.5:0.25:30;

for k = 1:length(num_images_list)
    auc = zeros(length(filters), length(sizes));
    
    num_images = num_images_list(k);
    h0 = train(1:num_images, :);
    h1 = train(offset+1:offset+num_images, :);
    signal = reshape(mean(h1) - mean(h0), [dim, dim]);
    
    tr_images = [h0; h1];
    val_images = [val(1:val_offset,:); val(val_offset+1:end, :)];
    
    roi_tr = reshape(tr_images', [dim, dim, 2*num_images_list(k)]);
    saroi_tr = roi_tr(:,:,1:num_images_list(k));
    sproi_tr = roi_tr(:,:,num_images_list(k)+1:end);

    roi_val = reshape(val_images', [dim, dim, 2*val_offset]);
    saroi_val = roi_val(:,:,1:val_offset);
    sproi_val = roi_val(:,:,val_offset+1:end);

    labels_val = [zeros(size(test,1)/2, 1); ones(size(test,1)/2, 1)];

    for i = 1:length(filters)
        for j = 1:length(sizes)
            [snr2, t_sp2, t_sa2, chimg2,tplimg2,meanSP2,meanSA2,meanSig2, k_ch2]=conv_LG_CHO_2d(saroi_tr, sproi_tr, saroi_val, sproi_val,sizes(j),filters(i),1, signal);
            thresh_conv = val*tplimg2(:);
            [x_conv, y_conv, t_conv, auc_conv] = perfcurve(labels_val, thresh_conv, 1);
            auc(i,j) = auc_conv;
        end
        i
    end

    [val1, ind1] = max(auc,[],1);
    [val2, ind2] = max(val1);

    index1 = ind1(ind2);
    index2 = ind2;
    val3 = auc(index1, index2);
    val4 = max(auc,[],'all');

    channels = filters(index1); %16
    width = sizes(index2); %1.75
    
    hyperparams(k, 1) = num_images_list(k);
    hyperparams(k, 2) = channels;
    hyperparams(k, 3) = width;
    hyperparams(k, 4) = val3;
    
end

pls_channels = zeros(length(num_images_list), 1);
pls_val_auc = zeros(length(num_images_list), 1);

for j = 1:length(num_images_list)
    auc = zeros(length(filters), 1);
    
    num_images = num_images_list(j);
    h0 = train(1:num_images, :);
    h1 = train(offset+1:offset+num_images, :);
    signal = mean(h1) - mean(h0);
    
    labels_train = [zeros(size(h0,1), 1); ones(size(h1,1), 1)];
    
    for i = 1:length(filters)
        [XL] = pls([h0; h1], labels_train, filters(i));
        
        C_pls = cov(train*XL);
        observer_pls = (C_pls \ (XL' * signal(:)))' * XL';
        
        thresh_pls = val*observer_pls';
        [x_pls, y_pls, t_pls, auc_pls] = perfcurve(labels_val, thresh_pls, 1);
        auc(i) = auc_pls;
        i
    end
    [pls_val_auc(j), pls_channels(j)] = max(auc);
end

%figure; imagesc(reshape(observer_ae, [dim, dim]))
%[val1, ind1] = max(auc,[],1);
%[val2, ind2] = max(val1);

%index1 = ind1(ind2);
%index2 = ind2;
%val3 = auc(index1, index2);
%val4 = max(auc,[],'all');

%channels = filters(index1); %16
%width = sizes(index2); %1.75