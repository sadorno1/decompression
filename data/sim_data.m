
hchirp = dsp.Chirp( ...
    'InitialFrequency', 0,...
    'TargetFrequency', 10, ...
    'TargetTime', 10, ...
    'SweepTime', 100, ...
    'SampleRate', 50, ...
    'SamplesPerFrame', 1000);

chirpData = (step(hchirp))';
evenFlag = mod(minute(datetime('now')),2);
if evenFlag
    chirpData = fliplr(chirpData);
end

X_chirpData = chirpData;

h_fig = figure(1); clf(h_fig); set(h_fig,'WindowStyle','docked');
plot(X_chirpData, 'g')
hold on;

y_chirpData = X_chirpData * 2;

h_fig = figure(2); clf(h_fig); set(h_fig,'WindowStyle','docked');
plot(X_chirpData, 'g')
hold on;
plot(y_chirpData)
grid on;
legend('input', 'output');


% Save to .mat file
cd('/home/k051m093/Documents/Signal_neural_network/data/')
save('sim_data_out.mat', 'X_chirpData', 'y_chirpData');
disp('Saved dataset to sim_data_out.mat');