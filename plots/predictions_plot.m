data_dir = 'Y:\data\Accum_Data\2024_Antarctica_Ground2\compression_measurements2\digrx0';
save_dir = 'C:\Users\s712a650\Desktop\project2.0\decompression\data';

coregistration_lag = -3.885;
Mt = 2;
fc = 750e6;
win = 1590:11605;

% Load pair 1: log_delay_01
[~, d1, ~]   = basic_load_arena(fullfile(data_dir, '20260408_173252_accum3_digrx0_0000.dat'));
out1         = mean(d1{4}(:,2:end), 2);

[~, d1in, ~] = basic_load_arena(fullfile(data_dir, '20260408_165831_accum3_digrx0_0000.dat'));
in1          = mean(d1in{4}(:,2:end), 2);

Nt   = length(in1);
freq = 1/Nt * ifftshift(-floor(Nt/2):floor((Nt-1)/2)).';
in1  = ifft(fft(in1) .* exp(1i*2*pi*freq*coregistration_lag));

% Load pair 2: log_delay_02
[~, d2, ~]   = basic_load_arena(fullfile(data_dir, '20260408_173546_accum3_digrx0_0000.dat'));
out2         = mean(d2{4}(:,2:end), 2);

[~, d2in, ~] = basic_load_arena(fullfile(data_dir, '20260408_172606_accum3_digrx0_0000.dat'));
in2          = mean(d2in{4}(:,2:end), 2);

Nt2   = length(in2);
freq2 = 1/Nt2 * ifftshift(-floor(Nt2/2):floor((Nt2-1)/2)).';
in2   = ifft(fft(in2) .* exp(1i*2*pi*freq2*coregistration_lag));

% Sanity check plots
figure(1); clf;
plot(real(out1)./max(abs(real(out1))), 'b'); hold on;
plot(real(in1) ./max(abs(real(in1))),  'r');
legend('output 173252','input 165831');
grid on; title('Pair 1 - log\_delay\_01');

figure(2); clf;
plot(real(out2)./max(abs(real(out2))), 'b'); hold on;
plot(real(in2) ./max(abs(real(in2))),  'r');
legend('output 173546','input 172606');
grid on; title('Pair 2 - log\_delay\_02');