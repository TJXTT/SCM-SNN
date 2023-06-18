clc;clear;
warning('off');
% SCM execution example.
%% Train data
tr_file = './example/train_data.mat';
val_file = './example/test_data.mat';
train_data = load(tr_file);
tr_feature = train_data.feature; % N*T*d, N: nums of samples, d: feature dimension
tr_label = train_data.label; % label info
%% Test data
val_data = load(val_file);
val_feature = val_data.feature;
val_label = val_data.label;
%% Nums of train/test samples, time steps
tr_nums = length(tr_feature(:,1,1));
val_nums = length(val_feature(:,1,1));
timestep = length(tr_feature(1,:,1));
feat_dim = length(tr_feature(1,1,:));
%% Hyper Parameters
gamma = 0.01;
rho = 1.0;
beta = 0.01;
K=10; % Iterations
%% SCM
% SCM Training
z = (randn(feat_dim+1, max(tr_label)+1)); % classifier initialization
model = SCM_Train_MultiClass(tr_feature, tr_label, gamma, beta, rho, z, K);
% Testing
val_res = SCM_Test_MultiClass(val_feature, val_label, model.alpha_Itr, model.bias_Itr);
[val_acc, itr] = max(val_res.acc_Itr); % MP Acc
[val_acc_s, itr_s] = max(val_res.acc_s_Itr); % Spike Acc 
disp(['SCM: MP_Acc: ', num2str(val_acc),' Spike_Acc: ', num2str(val_acc_s)]);
%% Test on Baseline
base_fc_file = './example/baseline_classifier.mat';
mse_classifier = load(base_fc_file);
mse_weight = mse_classifier.weight;
mse_bias = mse_classifier.bias;
w_Itr = cell(1,1);
b_Itr = cell(1,1);
w_Itr{1} = mse_weight';
b_Itr{1} = mse_bias;
base_res = SCM_Test_MultiClass(val_feature, val_label, w_Itr, b_Itr);
base_acc = base_res.acc_Itr;
base_acc_s = base_res.acc_s_Itr;
disp(['Baseline: MP_Acc: ', num2str(base_acc),' Spike_Acc: ', num2str(base_acc_s)]);


