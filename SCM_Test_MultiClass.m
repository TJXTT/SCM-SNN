function[res] = SCM_Test_MultiClass(feature, label, alpha_Itr, bias_Itr)

    label = label + 1;
    timestep = length(feature(1,:,1));
    acc_Itr = [];
    acc_s_Itr =[];

    for i=1:length(alpha_Itr)
        pred = 0;
        pred_s = 0;
        alpha = alpha_Itr{i};
        bias = bias_Itr{i};
        acc_t_steps = [];
        acc_s_t_steps = [];
        for t=1:timestep
            pred_step = squeeze(feature(:,t,:))*alpha + bias;
            pred = pred + pred_step;
            pred_step_s = pred_step>0; % convert to binary spike, Vth=0
            pred_s = pred_s + pred_step_s;
            [M I] = max(pred,[],2);
            [MS IS] = max(pred_s,[],2);
            acc_t_steps = [acc_t_steps sum(I'==label)/length(label)*100];
            acc_s_t_steps = [acc_s_t_steps sum(IS'==label)/length(label)*100];
        end
        acc = acc_t_steps(timestep);
        acc_s = acc_s_t_steps(timestep);
        acc_Itr = [acc_Itr acc];
        acc_s_Itr = [acc_s_Itr acc_s];
    end
    res.acc_Itr = acc_Itr;
    res.acc_s_Itr = acc_s_Itr;
end