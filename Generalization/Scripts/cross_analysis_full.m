dim = 64;

%param_dict = {[7, 20, 0.5, 40], [7, 20, 0.5, 30], [7, 20, 0.5, 50], [7, 20, 1.0, 40], [7, 20, 2.0, 40]};
%param_dict = {[7, 15, 0.5, 40], [7, 25, 0.5, 40]};
%param_dict = {[7, 0, 0.5, 40], [7, 5, 0.5, 40], [7, 10, 0.5, 40], [7, 15, 0.5, 40], [7, 20, 0.5, 40]};
%param_dict = {[7, 5, 0.5, 40], [7, 5, 1.0, 40], [7, 5, 2.0, 40], [7, 5, 5.0, 40], [7, 5, 10.0, 40]};
%param_dict = {[7, 1e-12, 0.5, 40], [7, 1e-12, 1.0, 40], [7, 1e-12, 2.0, 40], [7, 1e-12, 5.0, 40], [7, 1e-12, 10.0, 40]};
%param_dict = {[7, 1e-12, 1.0, 40], [7, 1e-12, 2.0, 40], [7, 1e-12, 3.0, 40], [7, 1e-12, 4.0, 40], [7, 1e-12, 5.0, 40]};
param_dict = {[7, 1e-12, 1.0, 40], [7, 1e-12, 2.0, 40], [7, 1e-12, 4.0, 40]};

n_imgs = 30000;
n_val_imgs = 5000;
n_test_imgs = 5000;
offset = 30000;
val_offset = n_val_imgs;
stop_setting = 'Val';

pls_filters_list = 1:20;
filters = 1:20;
sizes = 0.5:0.5:30;
num_filters = length(filters);
num_sizes = length(sizes);

alter_signal = false;

results_matrix_nho = zeros(3,3);
results_matrix_ae = zeros(3,3);
results_matrix_conv = zeros(3,3);
results_matrix_pls = zeros(3,3);

%%

figure

for k = 1:length(param_dict)
    fprintf('Pulse\n')
    params = param_dict{k};
    s = params(1);
    sig = params(2);
    h_w = params(3);
    h_h = params(4);
    %T = sprintf('../Data/hw_%.1g_hh_%i_sig_%i', h_w, h_h, sig);
    T = sprintf('../Data_FirstSet/hw_%.1g_hh_%i_sf_%.1g', h_w, h_h, sig);
    load([T, '/train.mat'])
    load([T, '/val.mat'])
    load([T, '/test.mat'])
    load([T, '/elliptical_signal_sy1_5.mat'])
    
    %elliptical_signal_sy1_5 = reshape(mean([train(end/2+1:end,:); val(end/2+1:end,:)]) - mean([train(1:end/2,:); val(1:end/2,:)]), [dim, dim]);
    signal = elliptical_signal_sy1_5;
    
    rng(1, 'twister') % for reproducability
    
    labels_val = [zeros(size(val,1)/2, 1); ones(size(val,1)/2, 1)];
    labels_test = [zeros(size(test,1)/2, 1); ones(size(test,1)/2, 1)];
    
    
    %% NHO
    
    num_images_list = [n_imgs];
    
    for i=1:length(num_images_list)
        num_images = num_images_list(i);
        images_pls = train(1:num_images, :);
        labels_pls = [zeros(size(images_pls,1), 1); ones(size(images_pls,1), 1)];
        images_pls = [train(1:num_images, :); train(offset+1:offset+num_images, :)];
        
        C0_nho = cov([train(1:num_images, :); val(1:val_offset, :)]);
        C1_nho = cov([train(offset+1:offset+num_images, :); val(val_offset+1:end, :)]);
        C_nho = 0.5 * (C0_nho + C1_nho);
        %observer_pls = (C_pls \ (XL' * signal(:)))' * XL';
        observer_nho = (pinv(C_nho) * signal(:))';
        
        thresh_nho = val*observer_nho';
        [x_nho, y_nho, t_nho, auc_nho] = perfcurve(labels_val, thresh_nho, 1);
        auc = auc_nho;
    end

    %% PLS
    pls_filters_list = 1:20;
    
    auc = zeros(1,20);
    
    for j = 1:length(pls_filters_list)
        pls_filters = pls_filters_list(j);
        for i=1:length(num_images_list)
            num_images = num_images_list(i);
            images_pls = train(1:num_images, :);
            labels_pls = [zeros(size(images_pls,1), 1); ones(size(images_pls,1), 1)];
            images_pls = [train(1:num_images, :); train(offset+1:offset+num_images, :)];
            
            [XL] = pls(images_pls, labels_pls, pls_filters);
            
            C0_pls = cov([train(1:num_images, :); val(1:val_offset, :)] * XL);
            C1_pls = cov([train(offset+1:offset+num_images, :); val(val_offset+1:end, :)] * XL);
            C_pls = 0.5 * (C0_pls + C1_pls);
            %observer_pls = (C_pls \ (XL' * signal(:)))' * XL';
            observer_pls = (pinv(C_pls) * (XL' * signal(:)))' * XL';
            
            thresh_pls = val*observer_pls';
            [x_pls, y_pls, t_pls, auc_pls] = perfcurve(labels_val, thresh_pls, 1);
            auc(j) = auc_pls;
        end
    end
    
    [auc_pls, index] = max(auc);
    c = pls_filters_list(index);
    
    for i=1:length(num_images_list)
        num_images = num_images_list(i);
        images_pls = train(1:num_images, :);
        labels_pls = [zeros(size(images_pls,1), 1); ones(size(images_pls,1), 1)];
        images_pls = [train(1:num_images, :); train(offset+1:offset+num_images, :)];

        [XL] = pls(images_pls, labels_pls, c);

        C0_pls = cov([train(1:num_images, :); val(1:val_offset, :)] * XL);
        C1_pls = cov([train(offset+1:offset+num_images, :); val(val_offset+1:end, :)] * XL);
        C_pls = 0.5 * (C0_pls + C1_pls);
        %observer_pls = (C_pls \ (XL' * signal(:)))' * XL';
        observer_pls = (pinv(C_pls) * (XL' * signal(:)))' * XL';
    end
    
    %% Conv LG
    n = 1;
    auc = zeros(num_filters, num_sizes);

    tr_images = [train(1:num_images_list(n),:); train(offset+1:offset+num_images_list(n), :)];
    val_images = [val(1:val_offset,:); val(val_offset+1:end, :)];

    roi_tr = reshape(tr_images', [dim, dim, 2*num_images_list(n)]);
    saroi_tr = roi_tr(:,:,1:num_images_list(n));
    sproi_tr = roi_tr(:,:,num_images_list(n)+1:end);

    roi_val = reshape(val_images', [dim, dim, 2*val_offset]);
    saroi_val = roi_val(:,:,1:val_offset);
    sproi_val = roi_val(:,:,val_offset+1:end);

    labels_val = [zeros(size(test,1)/2, 1); ones(size(test,1)/2, 1)];

    for i = 1:num_filters
        for j = 1:num_sizes
            [snr2, t_sp2, t_sa2, chimg2,tplimg2,meanSP2,meanSA2,meanSig2, k_ch2]=conv_LG_CHO_2d(saroi_tr, sproi_tr, saroi_val, sproi_val,sizes(j),filters(i),1, signal);
            thresh_conv = val*tplimg2(:);
            [x_conv, y_conv, t_conv, auc_conv] = perfcurve(labels_val, thresh_conv, 1);
            auc(i,j) = auc_conv;
        end
    end

    [val1, ind1] = max(auc,[],1);
    [val2, ind2] = max(val1);

    index1 = ind1(ind2);
    index2 = ind2;
    val3 = auc(index1, index2);
    val4 = max(auc,[],'all');

    channels = filters(index1); %16
    width = sizes(index2); %1.75
    
    [snr2, t_sp2, t_sa2, chimg2,tplimg2,meanSP2,meanSA2,meanSig2, k_ch2]=conv_LG_CHO_2d(saroi_tr, sproi_tr, saroi_val, sproi_val,width,channels,1, signal);
    %% AE
    
    scores_ae = zeros(1,20);
    model_path = sprintf('../Models_FirstSet/hw_%.1g_hh_%i_sf_1e-12/elliptical_signal_sy1_5', h_w, h_h);

    for j = 1:length(pls_filters_list)
        num_channels = pls_filters_list(j);
        for i=1:length(num_images_list)
            num_images = num_images_list(i);
            base_dir = sprintf('%s/TMI_%iLS_T_%i_%sStop/output/', model_path, num_channels, num_images, stop_setting);

            W = fread(fopen([base_dir, 'W1.dat']), [num_channels, dim*dim], 'float32');
            W = W';

            C0_v = cov([train(1:num_images, :); val(1:val_offset, :)] * W);
            C1_v = cov([train(offset+1:offset+num_images, :); val(val_offset+1:end, :)] * W);
            C_v = 0.5 * (C0_v + C1_v);
            %observer_ae = (C_v \ (W' * signal(:)))' * W';
            observer_ae = (pinv(C_v) * (W' * signal(:)))' * W';

            thresh_ae = val*observer_ae';
            [x_ae, y_ae, t_ae, auc_ae] = perfcurve(labels_test, thresh_ae, 1);
            scores_ae(j) = auc_ae;
        end
    end
    
    [auc_ae, index] = max(scores_ae);
    c_ae = pls_filters_list(index);
    
    for i=1:length(num_images_list)
        num_images = num_images_list(i);
        base_dir = sprintf('%s/TMI_%iLS_T_%i_%sStop/output/', model_path, c_ae, num_images, stop_setting);

        W = fread(fopen([base_dir, 'W1.dat']), [c_ae, dim*dim], 'float32');
        W = W';

        C0_v = cov([train(1:num_images, :); val(1:val_offset, :)] * W);
        C1_v = cov([train(offset+1:offset+num_images, :); val(val_offset+1:end, :)] * W);
        C_v = 0.5 * (C0_v + C1_v);
        %observer_ae = (C_v \ (W' * signal(:)))' * W';
        observer_ae = (pinv(C_v) * (W' * signal(:)))' * W';
    end
    
    %% Plots and performance
%     subplot(5,3,k)
%     T = sprintf('sig=%.1g, s=%i, H\\_w=%.2g, h=%i', sig, s, h_w, h_h);
%     %imagesc(reshape(train(2,:), [64, 64])); colormap gray; axis image; axis off; title(T); caxis([-60, 150])
%     imagesc(reshape(train(2,:), [64, 64])); colormap gray; axis image; axis off; title(T); caxis([0, 1.0])
%     subplot(5,3,k+3)
%     T = sprintf('NHO, AUC=%.4g', auc_nho);
%     imagesc(reshape(observer_nho, [64, 64])); colormap gray; axis image; axis off; title(T)
%     subplot(5,3,k+6)
%     T = sprintf('Conv LG, N = %i, W=%.3g, AUC=%.4g', channels, width, val3);
%     imagesc(reshape(tplimg2, [64, 64])); colormap gray; axis image; axis off; title(T)
%     subplot(5,3,k+9)
%     T = sprintf('PLS, N=%i, AUC=%.4g', c, auc_pls);
%     imagesc(reshape(observer_pls, [64, 64])); colormap gray; axis image; axis off; title(T)
%     subplot(5,3,k+12)
%     T = sprintf('AETSI, N=%i, AUC=%.4g', c_ae, auc_ae);
%     imagesc(reshape(observer_ae, [64, 64])); colormap gray; axis image; axis off; title(T)
    
    figure;
    imagesc(reshape(train(2,:), [dim, dim])); axis off;colormap gray;axis image;
    set(gcf,'Position',[100 100 400 400])
    figname = ['images/','gen_observer_img_', num2str(k), '.eps'];
    saveas(gcf,figname)
    close all;

    figure;
    imagesc(reshape(observer_pls, [dim, dim])); axis off;colormap gray;axis image;
    set(gcf,'Position',[100 100 400 400])
    figname = ['images/','gen_observer_pls_', num2str(k), '.eps'];
    saveas(gcf,figname)
    close all;

    figure;
    imagesc(reshape(tplimg2(:), [dim, dim])); axis off;colormap gray;axis image;
    set(gcf,'Position',[100 100 400 400])
    figname = ['images/','gen_observer_conv_', num2str(k), '.eps'];
    saveas(gcf,figname)
    close all;
    
    figure;
    imagesc(reshape(observer_nho, [dim, dim])); axis off;colormap gray;axis image;
    set(gcf,'Position',[100 100 400 400])
    figname = ['images/','gen_observer_nho_', num2str(k), '.eps'];
    saveas(gcf,figname)
    close all;
    
    figure;
    imagesc(reshape(observer_ae, [dim, dim])); axis off;colormap gray;axis image;
    set(gcf,'Position',[100 100 400 400])
    figname = ['images/','gen_observer_ae_', num2str(k), '.eps'];
    saveas(gcf,figname)
    close all;
    
    for m = 1:length(param_dict)
        params = param_dict{m};
        s = params(1);
        sig = params(2);
        h_w = params(3);
        h_h = params(4);
        %T = sprintf('../Data/hw_%.1g_hh_%i_sig_%i', h_w, h_h, sig);
        T = sprintf('../Data_FirstSet/hw_%.1g_hh_%i_sf_%.1g', h_w, h_h, sig);
        load([T, '/train.mat'])
        load([T, '/val.mat'])
        load([T, '/test.mat'])
        load([T, '/elliptical_signal_sy1_5.mat'])
        
        if alter_signal
            fprintf('Computing Signal\n')
            signal = reshape(mean([train(end/2+1:end,:); val(end/2+1:end,:)]) - mean([train(1:end/2,:); val(1:end/2,:)]), [dim, dim]);
            observer_nho = (pinv(C_nho) * signal(:))';
            observer_ae = (pinv(C_v) * (W' * signal(:)))' * W';
            [snr2, t_sp2, t_sa2, chimg2,tplimg2,meanSP2,meanSA2,meanSig2, k_ch2]=conv_LG_CHO_2d(saroi_tr, sproi_tr, saroi_val, sproi_val,width,channels,1, signal);
        end
        
        thresh_nho = test*observer_nho';
        [x_nho, y_nho, t_nho, auc_nho] = perfcurve(labels_test, thresh_nho, 1);
        
        thresh_conv = test*tplimg2(:);
        [x_conv, y_conv, t_conv, auc_conv] = perfcurve(labels_test, thresh_conv, 1);
        
        thresh_ae = test*observer_ae';
        [x_ae, y_ae, t_ae, auc_ae] = perfcurve(labels_test, thresh_ae, 1);
        
        thresh_pls = test*observer_pls';
        [x_pls, y_pls, t_pls, auc_pls] = perfcurve(labels_test, thresh_pls, 1);
        
        results_matrix_nho(k,m) = auc_nho;
        results_matrix_conv(k,m) = auc_conv;
        results_matrix_ae(k,m) = auc_ae;
        results_matrix_pls(k,m) = auc_pls;
        
    end
end

%keyboard;

%%
figure;
fprintf('Pulse\n')
T = sprintf('../Data/amalg');
load([T, '/train.mat'])
load([T, '/val.mat'])
load([T, '/test.mat'])
load([T, '/elliptical_signal_sy1_5.mat'])

%elliptical_signal_sy1_5 = reshape(mean([train(end/2+1:end,:); val(end/2+1:end,:)]) - mean([train(1:end/2,:); val(1:end/2,:)]), [dim, dim]);
signal = elliptical_signal_sy1_5;

rng(1, 'twister') % for reproducability

labels_val = [zeros(size(val,1)/2, 1); ones(size(val,1)/2, 1)];
labels_test = [zeros(size(test,1)/2, 1); ones(size(test,1)/2, 1)];
% PLS
pls_filters_list = 1:20;

num_images_list = [n_imgs];
scores_pls = zeros(length(num_images_list), 1);
ts_pls = zeros(length(num_images_list), size(test, 1));

auc = zeros(1,20);

for j = 1:length(pls_filters_list)
    pls_filters = pls_filters_list(j);
    for i=1:length(num_images_list)
        num_images = num_images_list(i);
        images_pls = train(1:num_images, :);
        labels_pls = [zeros(size(images_pls,1), 1); ones(size(images_pls,1), 1)];
        images_pls = [train(1:num_images, :); train(offset+1:offset+num_images, :)];

        [XL] = pls(images_pls, labels_pls, pls_filters);

        C0_nho = cov([train(1:num_images, :); val(1:val_offset, :)] * XL);
        C1_nho = cov([train(offset+1:offset+num_images, :); val(val_offset+1:end, :)] * XL);
        C_nho = 0.5 * (C0_nho + C1_nho);
        %observer_pls = (C_pls \ (XL' * signal(:)))' * XL';
        observer_pls = (pinv(C_nho) * (XL' * signal(:)))' * XL';

        thresh_pls = val*observer_pls';
        [x_pls, y_pls, t_pls, auc_pls] = perfcurve(labels_val, thresh_pls, 1);
        auc(j) = auc_pls;
    end
end

[auc_pls, index] = max(auc);
c = pls_filters_list(index);

for i=1:length(num_images_list)
    num_images = num_images_list(i);
    images_pls = train(1:num_images, :);
    labels_pls = [zeros(size(images_pls,1), 1); ones(size(images_pls,1), 1)];
    images_pls = [train(1:num_images, :); train(offset+1:offset+num_images, :)];

    [XL] = pls(images_pls, labels_pls, c);

    C0_pls = cov([train(1:num_images, :); val(1:val_offset, :)] * XL);
    C1_pls = cov([train(offset+1:offset+num_images, :); val(val_offset+1:end, :)] * XL);
    C_pls = 0.5 * (C0_pls + C1_pls);
    %observer_pls = (C_pls \ (XL' * signal(:)))' * XL';
    observer_pls = (pinv(C_pls) * (XL' * signal(:)))' * XL';
end

% Conv LG
n = 1;
auc = zeros(num_filters, num_sizes);

tr_images = [train(1:num_images_list(n),:); train(offset+1:offset+num_images_list(n), :)];
val_images = [val(1:val_offset,:); val(val_offset+1:end, :)];

roi_tr = reshape(tr_images', [dim, dim, 2*num_images_list(n)]);
saroi_tr = roi_tr(:,:,1:num_images_list(n));
sproi_tr = roi_tr(:,:,num_images_list(n)+1:end);

roi_val = reshape(val_images', [dim, dim, 2*val_offset]);
saroi_val = roi_val(:,:,1:val_offset);
sproi_val = roi_val(:,:,val_offset+1:end);

%labels_val = [zeros(size(test,1)/2, 1); ones(size(test,1)/2, 1)];

for i = 1:num_filters
    for j = 1:num_sizes
        [snr2, t_sp2, t_sa2, chimg2,tplimg2,meanSP2,meanSA2,meanSig2, k_ch2]=conv_LG_CHO_2d(saroi_tr, sproi_tr, saroi_val, sproi_val,sizes(j),filters(i),1, signal);
        thresh_conv = val*tplimg2(:);
        [x_conv, y_conv, t_conv, auc_conv] = perfcurve(labels_val, thresh_conv, 1);
        auc(i,j) = auc_conv;
    end
end

[val1, ind1] = max(auc,[],1);
[val2, ind2] = max(val1);

index1 = ind1(ind2);
index2 = ind2;
val3 = auc(index1, index2);
val4 = max(auc,[],'all');

channels = filters(index1); %16
width = sizes(index2); %1.75

[snr2, t_sp2, t_sa2, chimg2,tplimg2,meanSP2,meanSA2,meanSig2, k_ch2]=conv_LG_CHO_2d(saroi_tr, sproi_tr, saroi_val, sproi_val,width,channels,1, signal);

scores_ae = zeros(1,20);
model_path = sprintf('../Models/amalg/elliptical_signal_sy1_5');

for j = 1:length(pls_filters_list)
    num_channels = pls_filters_list(j);
    for i=1:length(num_images_list)
        num_images = num_images_list(i);
        base_dir = sprintf('%s/TMI_%iLS_T_%i_%sStop/output/', model_path, num_channels, num_images, stop_setting);

        W = fread(fopen([base_dir, 'W1.dat']), [num_channels, dim*dim], 'float32');
        W = W';

        C0_v = cov([train(1:num_images, :); val(1:val_offset, :)] * W);
        C1_v = cov([train(offset+1:offset+num_images, :); val(val_offset+1:end, :)] * W);
        C_v = 0.5 * (C0_v + C1_v);
        %observer_ae = (C_v \ (W' * signal(:)))' * W';
        observer_ae = (pinv(C_v) * (W' * signal(:)))' * W';

        thresh_ae = val*observer_ae';
        [x_ae, y_ae, t_ae, auc_ae] = perfcurve(labels_val, thresh_ae, 1);
        scores_ae(j) = auc_ae;
    end
end

[auc_ae, index] = max(scores_ae);
c_ae = pls_filters_list(index);

for i=1:length(num_images_list)
    num_images = num_images_list(i);
    base_dir = sprintf('%s/TMI_%iLS_T_%i_%sStop/output/', model_path, c_ae, num_images, stop_setting);

    W = fread(fopen([base_dir, 'W1.dat']), [c_ae, dim*dim], 'float32');
    W = W';

    C0_v = cov([train(1:num_images, :); val(1:val_offset, :)] * W);
    C1_v = cov([train(offset+1:offset+num_images, :); val(val_offset+1:end, :)] * W);
    C_v = 0.5 * (C0_v + C1_v);
    %observer_ae = (C_v \ (W' * signal(:)))' * W';
    observer_ae = (pinv(C_v) * (W' * signal(:)))' * W';
end

for i=1:length(num_images_list)
    num_images = num_images_list(i);
    images_pls = train(1:num_images, :);
    labels_pls = [zeros(size(images_pls,1), 1); ones(size(images_pls,1), 1)];
    images_pls = [train(1:num_images, :); train(offset+1:offset+num_images, :)];
    
    C0_nho = cov([train(1:num_images, :); val(1:val_offset, :)]);
    C1_nho = cov([train(offset+1:offset+num_images, :); val(val_offset+1:end, :)]);
    C_nho = 0.5 * (C0_nho + C1_nho);
    %observer_pls = (C_pls \ (XL' * signal(:)))' * XL';
    observer_nho = (pinv(C_nho) * signal(:))';
    
    thresh_nho = val*observer_nho';
    [x_nho, y_nho, t_nho, auc_nho] = perfcurve(labels_val, thresh_nho, 1);
    auc = auc_nho;
end

% Plots and results
% subplot(1,4,1)
% T = sprintf('Amalg');
% imagesc(reshape(train(2,:), [64, 64])); colormap gray; axis image; axis off; title(T); caxis([0, 1.0])
% subplot(1,4,2)
% T = sprintf('PLS, N=%i, AUC=%.4g', c, auc_nho);
% imagesc(reshape(observer_nho, [64, 64])); colormap gray; axis image; axis off; title(T)
% subplot(1,4,3)
% T = sprintf('Conv LG, N = %i, W=%.3g, AUC=%.4g', channels, width, val3);
% imagesc(reshape(observer_ae, [64, 64])); colormap gray; axis image; axis off; title(T)
% subplot(1,4,4)
% T = sprintf('AETSI, N=%i, AUC=%.4g', c_ae, auc_ae);
% imagesc(reshape(observer_ae, [64, 64])); colormap gray; axis image; axis off; title(T)

figure;
imagesc(reshape(observer_pls, [dim, dim])); axis off;colormap gray;axis image;
set(gcf,'Position',[100 100 400 400])
figname = ['images/','amalg_observer_pls_', num2str(k), '.eps'];
saveas(gcf,figname)
close all;

figure;
imagesc(reshape(tplimg2(:), [dim, dim])); axis off;colormap gray;axis image;
set(gcf,'Position',[100 100 400 400])
figname = ['images/','amalg_observer_conv_', num2str(k), '.eps'];
saveas(gcf,figname)
close all;

figure;
imagesc(reshape(observer_nho, [dim, dim])); axis off;colormap gray;axis image;
set(gcf,'Position',[100 100 400 400])
figname = ['images/','amalg_observer_nho_', num2str(k), '.eps'];
saveas(gcf,figname)
close all;

figure;
imagesc(reshape(observer_ae, [dim, dim])); axis off;colormap gray;axis image;
set(gcf,'Position',[100 100 400 400])
figname = ['images/','amalg_observer_ae_', num2str(k), '.eps'];
saveas(gcf,figname)
close all;
%%
n=1;
for m = 1:length(param_dict)
    params = param_dict{m};
    s = params(1);
    sig = params(2);
    h_w = params(3);
    h_h = params(4);
    T = sprintf('../Data_FirstSet/hw_%.1g_hh_%i_sf_1e-12', h_w, h_h);
    load([T, '/train.mat'])
    load([T, '/val.mat'])
    load([T, '/test.mat'])
    load([T, '/elliptical_signal_sy1_5.mat'])
    
    tr_images = [train(1:num_images_list(n),:); train(offset+1:offset+num_images_list(n), :)];
    val_images = [val(1:val_offset,:); val(val_offset+1:end, :)];
    
    roi_tr = reshape(tr_images', [dim, dim, 2*num_images_list(n)]);
    saroi_tr = roi_tr(:,:,1:num_images_list(n));
    sproi_tr = roi_tr(:,:,num_images_list(n)+1:end);

    roi_val = reshape(val_images', [dim, dim, 2*val_offset]);
    saroi_val = roi_val(:,:,1:val_offset);
    sproi_val = roi_val(:,:,val_offset+1:end);

    if alter_signal
        fprintf('Computing Signal\n')
        signal = reshape(mean([train(end/2+1:end,:); val(end/2+1:end,:)]) - mean([train(1:end/2,:); val(1:end/2,:)]), [dim, dim]);
        observer_nho = (pinv(C_nho) * (XL' * signal(:)))' * XL';
        observer_ae = (pinv(C_v) * (W' * signal(:)))' * W';
        %[snr2, t_sp2, t_sa2, chimg2,tplimg2,meanSP2,meanSA2,meanSig2, k_ch2]=conv_LG_CHO_2d(saroi_tr, sproi_tr, saroi_val, sproi_val,width,channels,1, signal);
    end

    thresh_pls = test*observer_pls';
    [x_pls, y_pls, t_pls, auc_pls] = perfcurve(labels_test, thresh_pls, 1);
    
    thresh_nho = test*observer_nho';
    [x_nho, y_nho, t_nho, auc_nho] = perfcurve(labels_test, thresh_nho, 1);

    thresh_conv = test*tplimg2(:);
    [x_conv, y_conv, t_conv, auc_conv] = perfcurve(labels_test, thresh_conv, 1);

    thresh_ae = test*observer_ae';
    [x_ae, y_ae, t_ae, auc_ae] = perfcurve(labels_test, thresh_ae, 1);

    results_amalg_pls(1,m) = auc_pls;
    results_amalg_nho(1,m) = auc_nho;
    results_amalg_conv(1,m) = auc_conv;
    results_amalg_ae(1,m) = auc_ae;
end