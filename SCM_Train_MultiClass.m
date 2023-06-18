function[Model] = SCM_Train_MultiClass(feature, label, gamma, beta, rho, z, maxItr)
    Vth = 1.0;
    converage = 0;
    iter = 1;
    lambda = 0.000001;
    alpha_Itr = cell(maxItr,1);
    bias_Itr = cell(maxItr, 1);
    row = length(feature(:,1,1));
    label = label + 1;
    one_hot_lb = zeros(row,max(label));
    feat_len = length(feature(1,1,:));
    D = (eye(feat_len+1));
    D(feat_len+1, feat_len+1)  = 0;

    for q = 1:row
        ids = label(q);
        one_hot_lb(q,ids) = 1;
    end
    one_hot_lb = 2*one_hot_lb - 1; % {0,1} to {-1,1}
    Y_split = cell(max(label),1);
    for c=1:max(label)
        ids = one_hot_lb(:,c);
        Y_split(c) = {diag(ids)};
    end
    timestep = length(feature(1,:,1));
    A_split = cell(max(label), timestep, 1);

    for c=1:max(label)
        for t=1:timestep
            A_split{c,t} = Y_split{c}*[squeeze(feature(:,t,:)) ones(row,1)];
        end
    end
    beta_split = cell(max(label),1);
    for c=1:max(label)
        beta_split{c} = beta*ones(row,1);
    end
    ATA_split = cell(max(label), 1);
    P_split = cell(max(label), 1);
    G_split = cell(max(label), 1);
    for c=1:max(label)
        for t=1:timestep
            if t == 1
                ATA_split{c} = A_split{c,t}'*A_split{c,t};
            else
                ATA_split{c} = ATA_split{c} + A_split{c,t}'*A_split{c,t};
            end
        end
        G_split{c} = gamma*D + rho*ATA_split{c};
        P_split{c} = (G_split{c}+ lambda*eye(feat_len+1))^(-1);
    end

    u_T = randn(row, timestep, max(label));
    zk = z;
    while ~converage == 1
        for c=1:max(label)
            for t = 1: timestep
                part1 = lambda*zk(:,c);
                if t == 1
                     part2 = rho*A_split{c,t}'*u_T(:,t,c);
                else
                    part2 = part2 + rho*A_split{c,t}'*u_T(:,t,c); 
                end
                part2 = part2 + A_split{c,t}'*beta_split{c};
            end

            z(:,c) = P_split{c} * (part1 + part2); % Eq. (17)

            for t = 1: timestep
                %Eq. (18)
                a = A_split{c,t}*z(:,c)-beta_split{c}./rho;
                pos = find(a <= (Vth) & a >(Vth-sqrt(2/rho)));
                u_T(:,t,c) = a;
                if ~isempty(pos)
                    u_T(pos,t,c) = Vth;
                end
            end
            zk(:,c) = z(:,c);
        end
        alpha_Itr{iter} = z(1:feat_len,:,:);
        bias_Itr{iter} = z(feat_len+1,:,:);
        if iter >= maxItr
            converage = 1;
        end
        iter = iter + 1;
    end
    % classifier
    Model.alpha_Itr = alpha_Itr;
    Model.bias_Itr = bias_Itr;
end