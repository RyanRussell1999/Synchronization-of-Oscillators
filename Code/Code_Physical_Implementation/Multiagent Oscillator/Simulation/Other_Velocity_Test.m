close all
clear all 

T = 1;
t = 0:T:100;
x(1) = -0.8;
AMP = 0.2;
x_dot(1) = AMP;
u = AMP;
for i = 1:length(t)-1
    if mod(i,10) == 0
        u = -u;
    end 
    x(i+1) = x(i) + T*u;
    x_dot(i+1) = u;
end

figure
plot(t, x_dot)
title('Control Input/ Velocity')
figure
plot(t, x)
title('Position')