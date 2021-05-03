% Plotting Simulation Data
close all 
% 
% ('Oscillator_Results_1616877058.2367737.mat')

figure
subplot(1,2,1)
plot(0:1:length(x_i)-1,x_i(:,1))
hold on
plot(0:1:length(x_leader)-1,x_leader(:,1))
title('Position')
legend('x_i', 'x_leader')
ylabel('Position [m]')
xlabel('Time [s]')
grid on
xlim([0 length(x_i)-10])

subplot(1,2,2)
plot(0:1:length(x_i)-1,x_i(:,2))
hold on
plot(0:1:length(x_leader)-1,x_leader(:,2))
title('Velocity')
legend('x_i', 'x_leader')
ylabel('Velocity [m/s]')
xlabel('Time [s]')
grid on
xlim([0 length(x_i)-10])