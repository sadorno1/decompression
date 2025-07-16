
h_fig = figure(1); clf(h_fig); set(h_fig,'WindowStyle','docked');
plot(y_train)
hold on;
plot(train_predictions)
grid on;
legend('y train', 'train predictions');



h_fig = figure(2); clf(h_fig); set(h_fig,'WindowStyle','docked');
plot(y_test)
hold on;
plot(test_predictions)
grid on;
legend('y test', 'test predictions');

