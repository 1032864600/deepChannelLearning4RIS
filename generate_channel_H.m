% wideband_channel_extractDOAs.m
% Today
% 10:13 AM
% 
% You edited an item
% Objective C
% wideband_channel_extractDOAs.m
% Last week
% Nov 11
% 
% You uploaded an item
% Objective C
% wideband_channel_extractDOAs.m
function [H, At, Ar, DOA, AOA, BETA, delay] = generate_channel_H(Nt,Nr,L,fs,fc,M,Num_users)
% clear all;
% Nt=64;      %trasmit antennas
% Nr=1;       % receive antennas
% L=3;        %paths
% fc=60*10^9; %carrier frequency
% fs=3.5*10^9;  % sampling frequency  fs=2M(BW)
% BW= 35*10^6;
% M=3;
% K=1;
% Nt_rf=4;

%%% path delay
% delay_max = 20e-9;
% delay = delay_max*rand(1,L);
delay = [[1.78032951094980e-08,9.55821708146159e-09,7.35131305345536e-09,4.73002944592621e-09,3.13224555779004e-09,8.52959370256540e-09,3.91263028442759e-10,1.18337584383972e-08,8.14978983666684e-09,7.66654915622931e-10]];

lambda_c = 3e8/fc; % wavelength
dt = lambda_c/2;
dr = dt;
H = zeros(Nr,Nt,Num_users,M);
% C = zeros(Nt,Nt,Num_users,M);
% Ch = zeros(Nt,Nt,Num_users,M);

for u=1:1:Num_users
    beta(1:L) = exp(1i*2*pi*rand(1,L));
%     beta(1:L) = sqrt(1/2)*(randn(1,L)+1i*randn(1,L)); % of var 1.
%     beta(1:L) = ones(1,L);
    
    %%% DoA
    DoA_index = non_overlapbeam(Nt,L,1);
    set_t = (-(Nt-1)/2:1:(Nt-1)/2)/(Nt/2);
    DoA = ((2/Nt)*rand(1,L) - 1/Nt) + set_t(DoA_index);
    %% AoA
    % AoA_index = non_overlapbeam(Nr,L,1);
    % set_r = (-(Nr-1)/2:1:(Nr-1)/2)/(Nr/2);
    % AoA = ((2/Nr)*rand(1,L) - 1/Nr) + set_r(AoA_index);
    AoA = 2*rand(1,L) - 1;
    
    
    for m = 1 :  M
        %     f = fc;
        f(m) = fc + fs/M*(m-1-(M-1)/2);
        lambda(m) = 3e8/f(m);
        for l = 1 : L
            At(:,l,u) = array_respones(DoA(l),Nt,dt,lambda(1));
            Ar(:,l,u) = array_respones(AoA(l),Nr,dr,lambda(1));
%             H(:,:,u,m) = H(:,:,u,m) + sqrt(Nt*Nr)* Ar(:,l,u)* At(:,l,u)';
            H(:,:,u,m) = H(:,:,u,m) + beta(l)*exp(-1i*2*pi*f(m)*delay(l))* Ar(:,l,u)* At(:,l,u)';
%             C(:,:,u,m) = C(:,:,u,m) + (Nt*Nr)*At(:,l,u)* At(:,l,u)';
%              norm(C(:,:,u,m) - H(:,:,u,m)'*H(:,:,u,m))
%             Ch(:,:,u,m) = Ch(:,:,u,m) + H(:,:,u,m)'*H(:,:,u,m);
        end
        H(:,:,u,m) = sqrt(Nt*Nr)*H(:,:,u,m);
%         H(:,:,u,m) = H(:,:,u,m);
%         C(:,:,u,m) = (Nt*Nr)*C(:,:,u,m);
        
    end
    
%         norm(C(:,:,u,m) - H(:,:,u,m)'*H(:,:,u,m))

%     norm(C(:,:,u,m)) - norm(H(:,:,u,m)'*H(:,:,u,m))
    
    %% collect data
    DOA(u,:) = DoA;
    AOA(u,:) = AoA;
    BETA(u,:) = beta;
end