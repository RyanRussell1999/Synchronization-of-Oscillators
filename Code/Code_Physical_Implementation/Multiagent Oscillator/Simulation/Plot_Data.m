% Plotting Simulation Data
close all 
clear all

load('Oscillator_Results_1619297991.7853346.mat')

figure
subplot(1,2,1)
plot(0:0.1:2000,x_i(1:20001,1))
hold on
plot(0:0.1:2000,x_j(1:20001,1))
hold on
plot(0:0.1:2000,x_leader(1:20001,1))
title('Position')
legend('x_i', 'x_j', 'x_leader')
ylabel('Position [m]')
xlabel('Time [s]')

subplot(1,2,2)
plot(0:0.1:2000,x_i(1:20001,2))
hold on
plot(0:0.1:2000,x_j(1:20001,2))
hold on
plot(0:0.1:2000,x_leader(1:20001,2))
title('Velocity')
legend('x_i', 'x_j', 'x_leader')
ylabel('Velocity [m/s]')
xlabel('Time [s]')

figure
subplot(1,2,1)
plot(0:0.1:100,x_i(1:1001,1))
hold on
plot(0:0.1:100,x_j(1:1001,1))
hold on
plot(0:0.1:100,x_leader(1:1001,1))
title('Position')
legend('x_i', 'x_j', 'x_leader')
ylabel('Position [m]')
xlabel('Time [s]')

subplot(1,2,2)
plot(0:0.1:100,x_i(1:1001,2))
hold on
plot(0:0.1:100,x_j(1:1001,2))
hold on
plot(0:0.1:100,x_leader(1:1001,2))
title('Velocity')
legend('x_i', 'x_j', 'x_leader')
ylabel('Velocity [m/s]')
xlabel('Time [s]')