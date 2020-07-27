% Multi-user channel estimation with reconfigurable intelligent surfaces via deep learning.
% MATLAB Codes for the paper: A. M. Elbir, A Papazafeiropoulos, P. Kourtessis,
% and S. Chatzinotas, "Deep Channel Learning For Large Intelligent Surfaces
% Aided mm-Wave Massive MIMO Systems", IEEE Wireless Communications
% Letters, in press, 2020.
% Prepared by Ahmet M. Elbir, 2020, Please cite the above work if you use
% this code. e-mail me for questions via: ahmetmelbir@gmail.com
% This includes the data generation for the training dataset and model
% training.


clear all;
% addpath('./AltMin/Narrowband');
% addpath('./TwoStage vs ZF vs ACE vs AS_final - two-bit');
% addpath(genpath('./AltMin'));
% ----------------------------- System Parameters -------------------------
% Num_bands = 3;   % number of subcarriers
opts.Num_paths = 4; %Number of channel paths
M = 64; % number of Tx antennas.
% M2 = 2^8;
L = 10; % number of LIS antennas. L=100;
K = 8; % Number of users
% T = 1024;
Ns = 1; % number of data streams.
% P = L;
NtRF = 4;%opts.Ns*opts.Num_users; % number of PS at Tx.
NrRF = 4;%opts.Ns; % number of PS at Rx.
% Ns = NtRF/opts.Num_users; % number of PS per user
opts.fc = 60*10^9; %carrier frequency
opts.BW = 4*10^9; %bandwidth
opts.fs = opts.BW;  % sampling frequency  fs=2M(BW)
opts.selectOutputAsPhases = 1;
opts.snr_param = [0]; % SNR dB.
opts.Nreal = 30; % number of realizations.
opts.Nchannels = 1; % number of channel matrices for the input data, make it 100 for better sampling the data space.
opts.fixedUsers = 0;
opts.fixedChannelGain = 0;
% opts.generateNoisyChannels = 1;
% opts.noiseLevelHdB_HB = [20 25 25]; % dB.
opts.noiseLevelHdB_CE = [ 25 30 40 50]; % dB.
opts.inputH = 1;
opts.inputRy = 0;
timeGenerate = tic;
rng(4096);

Num_paths = opts.Num_paths;
Nch = opts.Nchannels;
Nreal = opts.Nreal;
N = Nreal*Nch*K;
% X_HB = zeros(Num_bands*Nr,Nt,3,N);
Z = repmat(struct('channel_dc',zeros(M,1,K)),1,N );
% Y_Temp = zeros(Num_users,Ns*(Nt + Nr));
snr = db2pow(opts.snr_param);
% gamma = 1/(Nt*NtRF);
% Mt = Nt;
% Mr = Nr;
% Xce = eye(Mt);
F = dftmtx(M);% DFT_R = dftmtx(Nr);
% Fce = DFT_T(:,1:M);
% Wce = DFT_R(:,1:Mr);
jjHB = 1;
jjCE = 1;
jjA2 = 1;
X = eye(M); % pilot data.
X2 = eye(M*L);
V = eye(L); % reflect beamforming data.
doaMismatchList = linspace(0,10,Nch);
for nch = 1:Nch
    %% Generate channels
    % H.
%     if nch == 1
        % H, G.
        [H, At, Ar, DoA, AoA, beta, delay] = generate_channel_H(L,M,Num_paths,opts.fs,opts.fc,1,1);
        paramH(nch,1).At = At;
        paramH(nch,1).Ar = Ar;
        paramH(nch,1).DoA = DoA;
        paramH(nch,1).AoA = AoA;
        paramH(nch,1).beta = beta;
        % H LIS, G.
        [h_lis, At, Ar, DoA, AoA, beta, delay] = generate_channel_H_LIS(1,L,Num_paths,opts.fs,opts.fc,1,K);
        paramH_LIS(nch,1).At = At;
        paramH_LIS(nch,1).Ar = Ar;
        paramH_LIS(nch,1).DoA = DoA;
        paramH_LIS(nch,1).AoA = AoA;
        paramH_LIS(nch,1).beta = beta;
        % H DC.
        [h_dc, At, Ar, DoA, AoA, beta, delay] = generate_channel_H_LIS(1,M,Num_paths,opts.fs,opts.fc,1,K); 
        paramH_DC(nch,1).At = At;
        paramH_DC(nch,1).Ar = Ar;
        paramH_DC(nch,1).DoA = DoA;
        paramH_DC(nch,1).AoA = AoA;
        paramH_DC(nch,1).beta = beta;
%     else   
%     end
   %% TEST
   
   %%
   % cascaded channel.
    G = zeros(M,L,K);
    for kk = 1:K
        G(:,:,kk) = H* diag(h_lis(:,1,kk));
    end
    
%     h_lis = h_lis/norm(h_lis(:)); G = G/norm(G(:)); h_dc = h_dc/norm(h_dc(:));
    %% Channel estimation
    timeGenerate = tic;
    for nr = 1:Nreal
        snrIndex_CE = ceil(nr/(Nreal/size(opts.noiseLevelHdB_CE,2)));
        snrChannel = opts.noiseLevelHdB_CE(snrIndex_CE);
        %         snrIndex_HB = ceil(nr/(Nreal/size(opts.noiseLevelHdB_HB,2)));
        %         snrH_HB = opts.noiseLevelHdB_HB(snrIndex_HB);
%         X_wop = orth(1/sqrt(2)*(randn(M) + 1i*randn(M))); % data symbols.
        S = 1/sqrt(2)*(randn(K,M) + 1i*randn(K,M));
%         X_wop = F(:,1:K)*S; %X_wop = X_wop/norm(X_wop);
        for kk = 1:K % number of users.
            y_dc(kk,:) = awgn( h_dc(:,1,kk)'*X, snrChannel,'measured'  ); % direct channel data.
%             y_dc_wop(kk,:) = awgn( h_dc(:,1,kk)'*X_wop, snrChannel,'measured'  );
            h_dc_e(:,kk) = (y_dc(kk,:)*pinv(X))'; % direct channel LS.
            %         nmse_d = rms(vec( h_d_e(:,k) - h_d(:,1,k) )  )
            vG = []; h_dc_kron = [];
            for p = 1:L % for each LIS components. estimate cascaded channel
                v = V(:,p);
                vG = [vG v'*G(:,:,kk)'];
                h_dc_kron = [h_dc_kron h_dc(:,1,kk)'];
            end
            
            y_cc(:,:,kk) = reshape(awgn( (h_dc_kron + vG )*X2  ,snrChannel,'measured'),[M,L]); % cascaded channel data.
%             for p = 1:L % for each LIS components. estimate cascaded channel
%                 v = V(:,p);
%                 y_cc(:,p,kk) = awgn( (h_dc(:,1,kk)' +  v'*G(:,:,kk)' )*X  ,snrChannel,'measured');
% %                 y_cc_wop(:,p,kk) = awgn( (h_dc(:,1,kk)' +  v'*G(:,:,kk)' )*X_wop,snrChannel,'measured');
%                 G_e(:,p,kk) = (y_cc(:,p,kk)'*pinv(X) - h_dc_e(:,kk).').';
%                 
% %                 G_e2(:,p,kk) = (y_cc2(:,p,kk)'*pinv(X) - h_dc_e(:,kk).').';
% %                         rmse_g = rms ( vec ( G_e(:,p,kk) -  G(:,p,kk) )  )
%             end
            
            %% test.

            %%
%             H*V*h_lis(:,1,kk) - G(:,:,kk)*diag(V)
            
            R_dc(:,:,nr,kk) = reshape(y_dc(kk,:),[sqrt(M) sqrt(M)]); % (direct channel) data to feed ChannelNet_k
            R_cc(:,:,nr,kk) = y_cc(:,:,kk); % (cascaded channel) data to feed ChannelNet_k
%             R_cc2(:,:,nr,kk) = y_cc2(:,:,kk); % (cascaded channel) data to feed ChannelNet_k
            
%             R_dc_wop(:,:,nr,kk) = reshape(y_dc_wop(kk,:),[sqrt(M) sqrt(M)]); % (direct channel) data to feed ChannelNet_k
%             R_cc_wop(:,:,nr,kk) = y_cc_wop(:,:,kk); % (cascaded channel) data to feed ChannelNet_k
            
%             R_dc_wop(:,:,nr,k) = reshape(y_dc_wop(k,:),[sqrt(M2) sqrt(M2)]); % (direct channel) data to feed ChannelNet_k
%             R_cc_wop(:,:,nr,k) = y_cc_wop(:,:,k); % (cascaded channel) data to feed ChannelNet_k
        end
%         toc(timeGenerate)
    end
    %% Channel Estimation. Training data for A3
    %     jj = 1;
    for kk = 1:K % input-output pair of the DL model.
        for nr = 1:Nreal
            CENet{1,1}.X_dc(:,:,1,jjCE) = real(R_dc(:,:,nr,kk)); % input.
            CENet{1,1}.X_dc(:,:,2,jjCE) = imag(R_dc(:,:,nr,kk)); % input.
            CENet{1,1}.X_cc(:,:,1,jjCE) = real(R_cc(:,:,nr,kk)); % input.
            CENet{1,1}.X_cc(:,:,2,jjCE) = imag(R_cc(:,:,nr,kk)); % input.
%             CENet{1,1}.X_cc2(:,:,1,jjCE) = real(R_cc2(:,:,nr,kk)); % input.
%             CENet{1,1}.X_cc2(:,:,2,jjCE) = imag(R_cc2(:,:,nr,kk)); % input.
            
%             ENet{1,1}.X_dc_wop(:,:,1,jjCE) = real(R_dc_wop(:,:,nr,kk)); % input.
%             CENet{1,1}.X_dc_wop(:,:,2,jjCE) = imag(R_dc_wop(:,:,nr,kk)); % input.
%             CENet{1,1}.X_cc_wop(:,:,1,jjCE) = real(R_cc_wop(:,:,nr,kk)); % input.
%             CENet{1,1}.X_cc_wop(:,:,2,jjCE) = imag(R_cc_wop(:,:,nr,kk)); % input.
            channel_dc = h_dc(:,1,kk); % output. dc
            channel_cc = G(:,:,kk);% output. cc
            CENet{1,1}.Y_dc(jjCE,:) = [real(channel_dc(:)); imag(channel_dc(:))]; % output.
            CENet{1,1}.Y_cc(jjCE,:) = [real(channel_cc(:)); imag(channel_cc(:))]; % output.
            
            Z(1,jjCE).h_dc = h_dc;
            Z(1,jjCE).G = G;
            jjCE = jjCE + 1;
            
            keepIndex(jjCE) = [nch]; 
        end
    end
    %%
    nch
end % nch
% sizeOfInput = size(X_HB)
timeGenerate = toc(timeGenerate);
% stopp
%% SFCNN
% fprintf(2,['Train SFCNN \n'])
% [netSFCNN] = train_SFCNN(X_A2,Y_A2);
%%
% fprintf(2,['Train MC-CENet A2 for Channel Estimation \n'])
% [CEnet_A2] = train_CENet_A2(X_A2,Y_A2);
%% ChannelNet
for kkt = 1
    fprintf(2,['Train MLP \n'])
%     [MLP{1,kkt}.net_dc] = train_MLP(CENet{1,kkt}.X_dc,CENet{1,kkt}.Y_dc,0.00021);
%     [MLP{1,kkt}.net_cc] = train_MLP(CENet{1,kkt}.X_cc,CENet{1,kkt}.Y_cc,0.00000211); %0000011
    
    fprintf(2,['Train SFCNN \n'])
%     [SFCNN{1,kkt}.net_dc] = train_SFCNN(CENet{1,kkt}.X_dc,CENet{1,kkt}.Y_dc,0.00021);
%     [SFCNN{1,kkt}.net_cc] = train_SFCNN(CENet{1,kkt}.X_cc,CENet{1,kkt}.Y_cc,0.000001511); % 0000011
%     
    
    
%     fprintf(2,['Train ChannelNet_DC_WOP{' num2str(k) '} \n'])
%     [CEnet{1,k}.net_dc] = train_ChannelNet(CENet{1,k}.X_dc_wop,CENet{1,k}.Y_dc,0.00021);
%     fprintf(2,['Train ChannelNet_CC_WOP{' num2str(k) '} \n'])
%     [CEnet{1,k}.net_cc] = train_ChannelNet(CENet{1,k}.X_cc_wop,CENet{1,k}.Y_cc,0.000011);
    
    fprintf(2,['Train ChannelNet_DC{' num2str(kkt) '} \n'])
    [CEnet{1,kkt}.net_dc] = train_ChannelNet(CENet{1,kkt}.X_dc,CENet{1,kkt}.Y_dc,0.00021);
    fprintf(2,['Train ChannelNet_CC{' num2str(kkt) '} \n'])
    [CEnet{1,kkt}.net_cc] = train_ChannelNet(CENet{1,kkt}.X_cc,CENet{1,kkt}.Y_cc,0.00000211); % 0000011
%     fprintf(2,['Train ChannelNet_CC_WP2{' num2str(kkt) '} \n'])
%     [CEnet{1,kkt}.net_cc2] = train_ChannelNet(CENet{1,kkt}.X_cc2,CENet{1,kkt}.Y_cc,0.0000011);
    
    
end
%%
% fprintf(2,['Train MLP \n'])
% [MLPNet] = train_MLP(X_HB,Y_HB);
%% Run performance test
% XTest_dc(:,:,1) = real(RTest_dc_wp(:,:,k)); % generate first the test data RTest_dc_wp similar to R_dc and R_cc
% XTest_dc(:,:,2) = imag(RTest_dc_wp(:,:,k));
% [YPred] = double(predict(CEnet{1,1}.net_dc,XTest_dc));
% %                         timeCNN0(iTrial) = toc(timeSFCNNTemp);
% h_e_woP(:,k) = reshape(YPred(1:M) + 1i*YPred(M + 1:2*M), M,1);
% estError_dc_wP = rms(h_e_woP(:,k) - h(:,k));
