clc;
clear all;
close all;

i = 128;k = 1;
while i <= 2048
    M(:,k) = csvread(['C:\Users\Bhargav04\Documents\distributed computing group 6\Project3-Problem2\CUDA-Experiments\problem_2\part 3 and 4\speed_up_tesla_k80m\speed_up_for_',char(int2str(i)),'_element_matrix.csv']);
    i = i*2;
    k = k + 1;
end

str = 'b-^r-ok-<c->g-*';

figure;
for i = 1:5
    semilogx(2.^(0:7), M(2:9,i), str((i-1)*3 + 1:i*3)); hold on;
end
title('Two-dimensional unrolling');
xlabel('Number of elements unrolled');
ylabel('Speed-up');
legend('128x128 matrix', '256x256 matrix', '512x512 matrix', '1024x1024 matrix', '2048x2048 matrix');
grid on;

figure;
for i = 1:5
    semilogx(2.^(0:7), M(2+8:9+8,i), str((i-1)*3 + 1:i*3)); hold on;
end
title('One-dimensional unrolling');
xlabel('Number of elements unrolled');
ylabel('Speed-up');
legend('128x128 matrix', '256x256 matrix', '512x512 matrix', '1024x1024 matrix', '2048x2048 matrix');
grid on;

figure;
for i = 1:5
    semilogx(2.^(0:7), M(2+8+8:9+8+8,i), str((i-1)*3 + 1:i*3)); hold on;
end
title('Strassen-Winograd Algorithm with one dimensional unrolling');
xlabel('Number of elements unrolled');
ylabel('Speed-up');
legend('128x128 matrix', '256x256 matrix', '512x512 matrix', '1024x1024 matrix', '2048x2048 matrix');
grid on;