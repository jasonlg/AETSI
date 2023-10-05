%%
%clear all; close all; clc;

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

%num_images_list = [250];
scores_pls = zeros(length(num_images_list), 1);
ts_pls = zeros(length(num_images_list), size(test, 1));

for i=1:length(num_images_list)
    num_images = num_images_list(i);
    h0 = train(1:num_images, :);
    h1 = train(offset+1:offset+num_images, :);
    signal = mean(h1) - mean(h0);
    
    labels_train = [zeros(size(h0,1), 1); ones(size(h1,1), 1)];
    
    [XL] = pls([h0; h1], labels_train, pls_filters(i));
    
    C0_pls = cov(h0 * XL);
    C1_pls = cov(h1 * XL);
    C_pls = 0.5 * (C0_pls + C1_pls);

    observer_pls = (pinv(C_pls) * (XL' * signal(:)))' * XL';
    
    thresh_pls = test*observer_pls';
    [x_pls, y_pls, t_pls, auc_pls] = perfcurve(labels_test, thresh_pls, 1);
    scores_pls(i) = auc_pls;
    ts_pls(i,:) = thresh_pls;
end
figure; plot(num_images_list, scores_pls, '-o','linewidth', 2); hold on;

for i=1:length(num_images_list)
    fid = fopen(sprintf('%s/fitroc_pls_%iim.txt', folder, num_images_list(i)),'w');
    fprintf(fid,'LABROC\n');
    fprintf(fid,'Large\n');
    fprintf(fid,'%f\n',ts_pls(i,1:val_offset)); %H0
    fprintf(fid,'*\n');
    fprintf(fid,'%f\n',ts_pls(i,val_offset+1:2*val_offset));  %H1
    fprintf(fid,'*');
    fclose(fid);
end

%%
scores_conv = zeros(length(num_images_list), 1);
ts_conv = zeros(length(num_images_list), size(test, 1));

roi_val = reshape(val', [dim, dim, 2*val_offset]);
saroi_val = roi_val(:,:,1:val_offset);
sproi_val = roi_val(:,:,val_offset+1:end);

for i=1:length(num_images_list)
    num_images = num_images_list(i);
    h0 = train(1:num_images, :);
    h1 = train(offset+1:offset+num_images, :);
    signal = reshape(mean(h1) - mean(h0), [dim, dim]);
    
    tr_images = [h0; h1];
    val_images = [val(1:val_offset,:); val(val_offset+1:end, :)];
    
    roi_tr = reshape(tr_images', [dim, dim, 2*num_images_list(i)]);
    saroi_tr = roi_tr(:,:,1:num_images_list(i));
    sproi_tr = roi_tr(:,:,num_images_list(i)+1:end);
    
    labels_test = [zeros(size(test,1)/2, 1); ones(size(test,1)/2, 1)];
    
    [snr2, t_sp2, t_sa2, chimg2,tplimg2,meanSP2,meanSA2,meanSig2, k_ch2]=conv_LG_CHO_2d(saroi_tr, sproi_tr, saroi_val, sproi_val, conv_width(i), conv_filters(i), 1, signal, tol);
    thresh_conv = test*tplimg2(:);
    [x_conv, y_conv, t_conv, auc_conv] = perfcurve(labels_test, thresh_conv, 1);
    scores_conv(i) = auc_conv;
    ts_conv(i,:) = thresh_conv;
end

plot(num_images_list, scores_conv, '-s','linewidth', 2);

for i=1:length(num_images_list)
    fid = fopen(sprintf('%s/fitroc_conv_%iim.txt', folder, num_images_list(i)),'w');
    fprintf(fid,'LABROC\n');
    fprintf(fid,'Large\n');
    fprintf(fid,'%f\n',ts_conv(i,1:val_offset)); %H0
    fprintf(fid,'*\n');
    fprintf(fid,'%f\n',ts_conv(i,val_offset+1:2*val_offset));  %H1
    fprintf(fid,'*');
    fclose(fid);
end

%%
scores_nho = zeros(length(num_images_list), 1);
ts_nho = zeros(length(num_images_list), size(test, 1));

for i=1:length(num_images_list)
    num_images = num_images_list(i);
    h0 = train(1:num_images, :);
    h1 = train(offset+1:offset+num_images, :);
    signal = reshape(mean(h1) - mean(h0), [dim, dim]);
    
    C0_nho = cov(h0);
    C1_nho = cov(h1);
    C_nho = 0.5 * (C0_nho + C1_nho);
    %observer_nho = (C_nho \ (signal(:)))';
    %C_inv = pinv_adap(C_nho, frac);
    C_inv = pinv(C_nho, tol);
    observer_nho = (C_inv * signal(:))';
    
    thresh_nho = test*observer_nho';
    [x_nho, y_nho, t_nho, auc_nho] = perfcurve(labels_test, thresh_nho, 1);
    scores_nho(i) = auc_nho;
    ts_nho(i,:) = thresh_nho;
end
plot(num_images_list, scores_nho, '-^','linewidth', 2)

for i=1:length(num_images_list)
    fid = fopen(sprintf('%s/fitroc_nho_%iim.txt', folder, num_images_list(i)),'w');
    fprintf(fid,'LABROC\n');
    fprintf(fid,'Large\n');
    fprintf(fid,'%f\n',ts_nho(i,1:val_offset)); %H0
    fprintf(fid,'*\n');
    fprintf(fid,'%f\n',ts_nho(i,val_offset+1:2*val_offset));  %H1
    fprintf(fid,'*');
    fclose(fid);
end


%%
scores_matched = zeros(length(num_images_list), 1);
ts_matched = zeros(length(num_images_list), size(test, 1));

for i=1:length(num_images_list)
    num_images = num_images_list(i);
    h0 = train(1:num_images, :);
    h1 = train(offset+1:offset+num_images, :);
    signal = reshape(mean(h1) - mean(h0), [dim, dim]);
    observer_matched = signal(:);
    
    thresh_matched = test*observer_matched;
    [x_matched, y_matched, t_matched, auc_matched] = perfcurve(labels_test, thresh_matched, 1);
    scores_matched(i) = auc_matched;
    ts_matched(i,:) = thresh_matched;
end
plot(num_images_list, scores_matched, '-v','linewidth', 2)

for i=1:length(num_images_list)
    fid = fopen(sprintf('%s/fitroc_matched_%iim.txt', folder, num_images_list(i)),'w');
    fprintf(fid,'LABROC\n');
    fprintf(fid,'Large\n');
    fprintf(fid,'%f\n',ts_matched(i,1:val_offset)); %H0
    fprintf(fid,'*\n');
    fprintf(fid,'%f\n',ts_matched(i,val_offset+1:2*val_offset));  %H1
    fprintf(fid,'*');
    fclose(fid);
end

%%
scores_ae = zeros(length(num_images_list), 1);
ts_ae = zeros(length(num_images_list), size(test, 1));

for i=1:length(num_images_list)
    num_images = num_images_list(i);
    h0 = train(1:num_images, :);
    h1 = train(offset+1:offset+num_images, :);
    signal = reshape(mean(h1) - mean(h0), [dim, dim]);
    base_dir = sprintf('%s/TMI_%iLS_T_%i_%sStop/output/', model_path, ae_channels(i), num_images, stop_setting);
    
    W = fread(fopen([base_dir, 'W1.dat']), [ae_channels(i), dim*dim], 'float32');
    W = W';
    
    C0_v = cov(h0 * W);
    C1_v = cov(h1 * W);
    C_v = 0.5 * (C0_v + C1_v);
    %observer_ae = (C_v \ (W' * signal(:)))' * W';
    observer_ae = (pinv(C_v) * (W' * signal(:)))' * W';
    
    thresh_ae = test*observer_ae';
    [x_ae, y_ae, t_ae, auc_ae] = perfcurve(labels_test, thresh_ae, 1);
    scores_ae(i) = auc_ae;
    ts_ae(i,:) = thresh_ae;
end

plot(num_images_list, scores_ae, '-x','linewidth', 2);
xlabel('Number of Images')
ylabel('AUC')
title(fig_title, 'Fontweight', 'Normal')
legend('PLS', 'Conv', 'Numeric HO', 'Matched', 'AETSI', 'location', 'best')
set(gca, 'Fontsize', 20)
figname = ['images/',signal_name, '_auc','.png'];
saveas(gcf,figname)

for i=1:length(num_images_list)
    fid = fopen(sprintf('%s/fitroc_ae_%iim.txt', folder, num_images_list(i)),'w');
    fprintf(fid,'LABROC\n');
    fprintf(fid,'Large\n');
    fprintf(fid,'%f\n',ts_ae(i,1:val_offset)); %H0
    fprintf(fid,'*\n');
    fprintf(fid,'%f\n',ts_ae(i,val_offset+1:2*val_offset));  %H1
    fprintf(fid,'*');
    fclose(fid);
end

%%
figure;
imagesc(reshape(observer_ae, [dim, dim])); axis off;colormap gray;axis image;
set(gcf,'Position',[100 100 400 400])
figname = ['images/',signal_name, '_observer_ae','.eps'];
saveas(gcf,figname)
close all;

figure;
imagesc(reshape(observer_pls, [dim, dim])); axis off;colormap gray;axis image;
set(gcf,'Position',[100 100 400 400])
figname = ['images/',signal_name, '_observer_pls','.eps'];
saveas(gcf,figname)
close all;

figure;
imagesc(reshape(observer_nho, [dim, dim])); axis off;colormap gray;axis image;
set(gcf,'Position',[100 100 400 400])
figname = ['images/',signal_name, '_observer_nho','.eps'];
saveas(gcf,figname)
close all;

figure;
imagesc(reshape(tplimg2(:), [dim, dim])); axis off;colormap gray;axis image;
set(gcf,'Position',[100 100 400 400])
figname = ['images/',signal_name, '_observer_conv','.eps'];
saveas(gcf,figname)
close all;

figure;
imagesc(reshape(observer_matched, [dim, dim])); axis off;colormap gray;axis image;
set(gcf,'Position',[100 100 400 400])
figname = ['images/',signal_name, '_observer_matched','.eps'];
saveas(gcf,figname)
close all;

%% Hotelling Observer
load('../Data/Lumpy_archive/train.mat');
load('../Data/Lumpy_archive/val.mat');
load('../Data/Lumpy_SKS/train_signals.mat');
load('../Data/Lumpy_SKS/val_signals.mat');

%load('/home/jasonlg/Autoencoder_DualLoss/Data/Lumpy/train_clean.mat')
%load('/home/jasonlg/Autoencoder_DualLoss/Data/Lumpy/val_clean.mat')

C0_g = cov([train(1:end/2, :); val(1:end/2, :)]);
C1_g = cov([train(end/2+1:end, :); val(end/2+1:end, :)] + [train_signals; val_signals]);
C_g = 0.5 * (C0_g + C1_g);
% images_ho = [train(1:end/2,:); train(end/2+1:end,:) + signal(:)'];
% C_g = cov(images_ho);
[U, S, V] = svd(C_g);
S_n = 400 * ones(4096,1);
S_prime = 1./(diag(S) + S_n);
signal = reshape(mean([train_signals; val_signals]), [64, 64]);
hotelling_observer = (V * diag(S_prime) * U' * signal(:))';
thresh_ho = test*hotelling_observer';
[x_ho, y_ho, t_ho, auc_ho] = perfcurve(labels_test, thresh_ho, 1);
% 
% fid = fopen(sprintf('Test_Statistics_Elliptical_SKS/fitroc_ho_%iim.txt', num_images_list(end)),'w');
% fprintf(fid,'LABROC\n');
% fprintf(fid,'Large\n');
% fprintf(fid,'%f\n',thresh_ho(1:val_offset)); %H0
% fprintf(fid,'*\n');
% fprintf(fid,'%f\n',thresh_ho(val_offset+1:2*val_offset));  %H1
% fprintf(fid,'*');
% fclose(fid);
% scores_ae(i) = auc_pls;

% sc = 5;
% sr = ceil(num_filters / sc);
% 
% figure;
% for i = 1:num_filters
%     subplot(sr, sc, i);
%     imagesc(reshape(XL(:,i), [64, 64]));
%     colormap gray;
% end

% %%
% t = textread('test_file.txt');
% x = t(:,1);
% y = t(:,2);
% figure;
% plot(x, y);

figure;
imagesc(reshape(hotelling_observer, [dim, dim])); axis off;colormap gray;axis image;
set(gcf,'Position',[100 100 400 400])
figname = ['images/',signal_name, '_observer_ho','.eps'];
saveas(gcf,figname)
close all;