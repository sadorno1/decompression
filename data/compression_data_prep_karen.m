
cd('/kucresis/scratch/data/Accum_Data/2024_Antarctica_Ground2/compression_measurements/20250529');


if 0
  %% COREGISTRATION
  % Load chirp output (20250529_135329_config)

  chirp_output_fn = 'digrx0/20250529_135329_accum3_digrx0_0000.dat';

  [hdr,data,hdr_debug] = basic_load_arena(chirp_output_fn);

  data{4} = data{4}(:,2:end);
  data{4} = mean(data{4},2);

  % imagesc(imag(data{4}))

  % Load chirp input (20250529_155035_config)

  chirp_input_fn = 'digrx0/20250529_155035_accum3_digrx0_0000.dat';

  [hdr_in,data_in,hdr_debug] = basic_load_arena(chirp_input_fn);

  data_in{4} = data_in{4}(:,2:end);
  data_in{4} = mean(data_in{4},2);

  % Number of samples in fast-time

  % imagesc(imag(data_in{4}))

  %%
  h_fig = figure(1); clf(h_fig); set(h_fig,'WindowStyle','docked');
  plot(real(data{4}))
  hold on;
  plot(real(data_in{4}))
  grid on;

  %%
  h_fig = figure(2); clf(h_fig); set(h_fig,'WindowStyle','docked');
  [data_xcorr,lags] = xcorr(data{4}, data_in{4});
  plot(lags, 10*log10(abs(data_xcorr)),'b.')
  grid on;
  hold on;

  % Oversample by Mt-times
  Mt = 100;
  Nt = size(data_xcorr,1);
  Nt_Mt = Nt * Mt;
  data_xcorr_Mt = interpft(data_xcorr, Nt*Mt);

  lags_Mt = lags(1) + 1/Mt * (0:Nt_Mt-1).';

  plot(lags_Mt, 10*log10(abs(data_xcorr_Mt)),'r-')

  % 3.885 sample delay

  %%

  Nt = size(data_in{4},1);
  coregistration_lag = -3.885;
  freq = 1/Nt * ifftshift(-floor(Nt/2) : floor((Nt-1)/2)).';
  data_coregistered = ifft(fft(data_in{4}) .* exp(1i*2*pi*freq*coregistration_lag));

  h_fig = figure(3); clf(h_fig); set(h_fig,'WindowStyle','docked');
  plot(real(data{4}),'b')
  hold on;
  plot(real(data_in{4}),'r')
  plot(real(data_coregistered),'k')
  grid on;

  %%
  h_fig = figure(4); clf(h_fig); set(h_fig,'WindowStyle','docked');
  [data_xcorr,lags] = xcorr(data{4}, data_coregistered);
  plot(lags, 10*log10(abs(data_xcorr)),'b.')
  grid on;
  hold on;

  % Oversample by Mt-times
  Mt = 100;
  Nt = size(data_xcorr,1);
  Nt_Mt = Nt * Mt;
  data_xcorr_Mt = interpft(data_xcorr, Nt*Mt);

  lags_Mt = lags(1) + 1/Mt * (0:Nt_Mt-1).';

  plot(lags_Mt, 10*log10(abs(data_xcorr_Mt)),'r-')

else

  % Load random waveform output (20250529_153300_config)

  chirp_output_fn = 'digrx0/20250529_153300_accum3_digrx0_0000.dat';

  [hdr,data,hdr_debug] = basic_load_arena(chirp_output_fn);

  data{4} = data{4}(:,2:end);
  data{4} = mean(data{4},2);

  % imagesc(imag(data{4}))

  % Load random waveform input (20250529_153941_config)

  chirp_input_fn = 'digrx0/20250529_153941_accum3_digrx0_0000.dat';

  [hdr_in,data_in,hdr_debug] = basic_load_arena(chirp_input_fn);

  data_in{4} = data_in{4}(:,2:end);
  data_in{4} = mean(data_in{4},2);


  Nt = size(data_in{4},1);
  coregistration_lag = -3.885;
  freq = 1/Nt * ifftshift(-floor(Nt/2) : floor((Nt-1)/2)).';
  data_in{4} = ifft(fft(data_in{4}) .* exp(1i*2*pi*freq*coregistration_lag));


  % Number of samples in fast-time

  % imagesc(imag(data_in{4}))

  %%
  h_fig = figure(1); clf(h_fig); set(h_fig,'WindowStyle','docked');
  plot(real(data{4}), 'g')
  hold on;
  plot(real(data_in{4}) .* max(abs(real(data{4}))) ./ max(abs(real(data_in{4})))  )
  grid on;

  train_input = data_in{4}(1590:11605,1);
  train_output = data{4}(1590:11605,1);  %original waveform

  Nt = length(train_input);
  Mt = 2;
  Nt_new = Nt*Mt;
  train_input = interpft(train_input,Nt_new);
  train_output = interpft(train_output,Nt_new);
  fs = 500e6 * Mt;
  dt = 1/fs;
  fc = 750e6;
  time = dt * (0:Nt_new-1).';
  train_input_real = real(train_input) .* cos(2*pi*fc*time) - imag(train_input) .* sin(2*pi*fc*time);
  train_output_real = real(train_output) .* cos(2*pi*fc*time) - imag(train_output) .* sin(2*pi*fc*time);

end

% Save to .mat file
cd('/home/k051m093/Documents/radar_cresis/radar_cresis/data/')
save('compression_data_output.mat', 'train_output_real', 'train_input_real');
disp('Saved distorted radar dataset to compression_data_output.mat');