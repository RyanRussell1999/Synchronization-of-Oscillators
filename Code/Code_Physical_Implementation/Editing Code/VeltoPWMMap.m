%% PWM to Velocity Mapping
load("Oscillator_Results_1615921266.6321125.mat")
Motor_PWMs = 10:5:100;
k = 1;
m = 1;
n = 1;

% Segment Data
for i = 1:length(PWM)
    current_pwm = Motor_PWMs(k); 
    if PWM(i) == current_pwm
        PWM_seg(n,m) = PWM(i);
        m = m + 1;
    else
        n = n + 1;
        m = 1;
        PWM_seg(n,m) = PWM(i);
        k = k + 1;
    end
end

for i = 1:length(PWM_seg(1,:))
    if PWM_seg(1,i) ~= 0
        PWM_10(i) = PWM_seg(1,i);
    end
    if PWM_seg(2,i) ~= 0
        PWM_15(i) = PWM_seg(2,i);
    end 
    if PWM_seg(3,i) ~= 0
        PWM_20(i) = PWM_seg(3,i);
    end   
    if PWM_seg(4,i) ~= 0
        PWM_25(i) = PWM_seg(4,i);
    end
    if PWM_seg(5,i) ~= 0
        PWM_30(i) = PWM_seg(5,i);
    end
    if PWM_seg(6,i) ~= 0
        PWM_35(i) = PWM_seg(6,i);
    end
    if PWM_seg(7,i) ~= 0
        PWM_40(i) = PWM_seg(7,i);
    end
    if PWM_seg(8,i) ~= 0
        PWM_45(i) = PWM_seg(8,i);
    end
    if PWM_seg(9,i) ~= 0
        PWM_50(i) = PWM_seg(9,i);
    end
    if PWM_seg(10,i) ~= 0
        PWM_55(i) = PWM_seg(10,i);
    end
    if PWM_seg(11,i) ~= 0
        PWM_60(i) = PWM_seg(11,i);
    end
    if PWM_seg(12,i) ~= 0
        PWM_65(i) = PWM_seg(12,i);
    end
    if PWM_seg(13,i) ~= 0
        PWM_70(i) = PWM_seg(13,i);
    end
    if PWM_seg(14,i) ~= 0
        PWM_75(i) = PWM_seg(14,i);
    end
    if PWM_seg(15,i) ~= 0
        PWM_80(i) = PWM_seg(15,i);
    end
    if PWM_seg(16,i) ~= 0
        PWM_85(i) = PWM_seg(16,i);
    end
    if PWM_seg(17,i) ~= 0
        PWM_90(i) = PWM_seg(17,i);
    end
    if PWM_seg(18,i) ~= 0
        PWM_95(i) = PWM_seg(18,i);
    end
    if PWM_seg(19,i) ~= 0
        PWM_100(i) = PWM_seg(19,i);
    end
end

k = 1;
% Perform Velocity Calculation for each PWM
for i = 1:length(PWM_10)
    if i < (length(PWM_10) - 1)
        VEL_10(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end

for i = 1:length(PWM_15)
    if i < (length(PWM_15) - 1)
        VEL_15(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end

for i = 1:length(PWM_20)
    if i < (length(PWM_20) - 1)
        VEL_20(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end

for i = 1:length(PWM_25)
    if i < (length(PWM_25) - 1)
        VEL_25(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end

for i = 1:length(PWM_30)
    if i < (length(PWM_30) - 1)
        VEL_30(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end

for i = 1:length(PWM_35)
    if i < (length(PWM_35) - 1)
        VEL_35(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end

start_40 = k;
for i = 1:length(PWM_40)
    if i < (length(PWM_40) - 1)
        VEL_40(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_40 = k;
start_45 = k+1;
for i = 1:length(PWM_45)
    if i < (length(PWM_45) - 1)
        VEL_45(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_45 = k;
start_50 = k+1;
for i = 1:length(PWM_50)
    if i < (length(PWM_50) - 1)
        VEL_50(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_50 = k;
start_55 = k+1;
for i = 1:length(PWM_55)
    if i < (length(PWM_55) - 1)
        VEL_55(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_55 = k;
start_60 = k+1;
for i = 1:length(PWM_60)
    if i < (length(PWM_60) - 1)
        VEL_60(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_60 = k;
start_65 = k+1;
for i = 1:length(PWM_65)
    if i < (length(PWM_65) - 1)
        VEL_65(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_65 = k;
start_70 = k+1;
for i = 1:length(PWM_70)
    if i < (length(PWM_70) - 1)
        VEL_70(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_70 = k;
start_75 = k+1;
for i = 1:length(PWM_75)
    if i < (length(PWM_75) - 1)
        VEL_75(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_75 = k;
start_80 = k+1;

for i = 1:length(PWM_80)
    if i < (length(PWM_80) - 1)
        VEL_80(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_80 = k;
start_85 = k+1;
for i = 1:length(PWM_85)
    if i < (length(PWM_85) - 1)
        VEL_85(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_85 = k;
start_90 = k+1;
for i = 1:length(PWM_90)
    if i < (length(PWM_90) - 1)
        VEL_90(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_90 = k;
start_95 = k+1;
for i = 1:length(PWM_95)
    if i < (length(PWM_95) - 1)
        VEL_95(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_95 = k;
start_100 = k+1;
for i = 1:length(PWM_100)
    if i < (length(PWM_100) - 1)
        VEL_100(i) = (distance(k) - distance(k+1)) / (time(k+1) - time(k));
    end
    k = k + 1; 
end
end_100 = k;

avg_VEL_40 = norm(VEL_40)/length(VEL_40);
avg_VEL_45 = norm(VEL_45)/length(VEL_45);
avg_VEL_50 = norm(VEL_50)/length(VEL_50);
avg_VEL_55 = norm(VEL_55)/length(VEL_55);
avg_VEL_60 = norm(VEL_60)/length(VEL_60);
avg_VEL_65 = norm(VEL_65)/length(VEL_65);
avg_VEL_70 = norm(VEL_70)/length(VEL_70);
avg_VEL_75 = norm(VEL_75)/length(VEL_75);
avg_VEL_80 = norm(VEL_80)/length(VEL_80);
avg_VEL_85 = norm(VEL_85)/length(VEL_85);
avg_VEL_90 = norm(VEL_90)/length(VEL_90);
avg_VEL_95 = norm(VEL_95)/length(VEL_95);
avg_VEL_100 = norm(VEL_100)/length(VEL_100); 

start_40 = 7;
end_40 = 825;
start_45 = 826;
end_45 = 1513;
start_50 = 1514;
end_50 = 2087;
start_55 = 2088;
end_55 = 2378;
start_60 = 2379;
end_60 = 3132;
start_65 = 3133;
end_65 = 3373;
start_70 = 3374;
end_70 = 3916;
start_75 = 3917;
end_75 = 4160;
start_80 = 4161;
end_80 = 4435;
start_85 = 4436;
end_85 = 4612;
start_90 = 4613;
end_90 = 4832;
start_95 = 4833;
end_95 = 5036;
start_100 = 5037;
end_100 = 5249;

avg_VEL_40 = (distance(start_40) - distance(end_40))/(time(end_40) - time(start_40));
avg_VEL_45 = (distance(start_45) - distance(end_45))/(time(end_45) - time(start_45));
avg_VEL_50 = (distance(start_50) - distance(end_50))/(time(end_50) - time(start_50));
avg_VEL_55 = (distance(start_55+20) - distance(end_55))/(time(end_55) - time(start_55+20));
avg_VEL_60 = (distance(start_60) - distance(end_60))/(time(end_60) - time(start_60));
avg_VEL_65 = (distance(start_65) - distance(end_65))/(time(end_65) - time(start_65));
avg_VEL_70 = (distance(start_70+20) - distance(end_70))/(time(end_70) - time(start_70+20));
avg_VEL_75 = (distance(start_75+20) - distance(end_75))/(time(end_75) - time(start_75+20));
avg_VEL_80 = (distance(start_80+20) - distance(end_80))/(time(end_80) - time(start_80+20));
avg_VEL_85 = (distance(start_85+20) - distance(end_85))/(time(end_85) - time(start_85+20));
avg_VEL_90 = (distance(start_90) - distance(end_90))/(time(end_90) - time(start_90));
avg_VEL_95 = (distance(start_95) - distance(end_95))/(time(end_95) - time(start_95));
avg_VEL_100 = (distance(start_100) - distance(end_100))/(time(end_100) - time(start_100));

VEL_VEC = [avg_VEL_40; avg_VEL_45; avg_VEL_50; avg_VEL_60; avg_VEL_70; avg_VEL_80; avg_VEL_85; avg_VEL_90];
PWM_VEC = [40; 45; 50; 60; 70; 80; 85; 90];
xq = 40:100;
vel_map = interp1(PWM_VEC, VEL_VEC, xq, 'spline');

figure
plot(xq, vel_map)
hold on
plot(xq, (0.0135*xq).^2)
xlabel("PWM")
ylabel("Velocity [m/s]")

% PWM = sqrt(vel)/0.0135