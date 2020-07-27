function [Rate] = calculateRate_WB_SU(Frf_CNN,Fbb_CNN,Wrf_CNN,Wbb_CNN,H,snr)
Rate = 0;
[Nr,Nt,~,Num_bands] = size(H);
Ns = size(Fbb_CNN,2);
for b = 1:Num_bands
    Wt = sqrt(1)*Wrf_CNN*Wbb_CNN(:,:,b);
    Ft = sqrt(1)*Frf_CNN*Fbb_CNN(:,:,b);
%     C = inv(Wt'*Wt + 0.0001*eye(size(Wt,2)))*Wt';
%     Rate = Rate + real( log2(det(eye(Ns) + snr*C*H(:,:,1,b)*(Ft*Ft')*H(:,:,1,b)'*Wt )))/Num_bands;
    Rate = Rate + real(helperComputeSpectralEfficiency(H(:,:,1,b),Ft,Wt,Ns,snr) )/(Num_bands);
end
