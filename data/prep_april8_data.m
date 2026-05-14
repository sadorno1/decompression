% 165831 = output (target, high amplitude)
% 173252 = input (attenuated, low amplitude)

p1_input  = p1_out  % 173252 direct cable = attenuated input
p1_target = p1_in   % 165831 through hardware = clean target

p2_input  = p2_out  % 173546
p2_target = p2_in   % 172606

train_input_real  = [p1_input;  p2_input];
train_output_real = [p1_target; p2_target];

fprintf('Total samples: %d\n', length(train_input_real));

save(fullfile(save_dir, 'compression_data_april8.mat'), ...
     'train_input_real', 'train_output_real');
disp('Saved compression_data_april8.mat');