close all
clear all 

T = 1;
t = 0:T:100;
x(1) = 0;
AMP = 0.5;
FREQ = pi/4;
x_dot = AMP * sin(t * FREQ + pi/2);
for i = 1:length(t)-1
    x(i+1) = x(i) + T*x_dot(i);
end

figure
plot(t, x_dot)
title('Control Input/ Velocity')
figure
plot(t, x)
title('Position')