data = xlsread('/Users/garima/Documents/WSU/WSU_Courses/Fall_2015/Neural Networks/spambase/spambase_randomized_orig_labels.xlsx' );
x= data(:,(1:57));
y= data(:,58);

covx= cov(x);
% figure(1);
% plot(covx);
[V,D]=eigs(covx,3);
e_val= diag(D);



accuracy = zeros(size(V,2),1);
sum_e=0;
variance_percent = zeros(size(V,2),1);
diag_vals=diag(covx);
size(diag_vals);

total_variance = sum(diag_vals);

for i =1:size(V,2)
    sum_e = sum_e + e_val(i);
    variance_percent(i) = sum_e/total_variance;
end

figure(1)
hold on
plot(variance_percent);
plot(variance_percent,'r*');
title('Principal Components vs Variance Percentage');
xlabel('Variance percentage');
ylabel('# of principal components');
hold off;
[W,D]=eigs(covx,2);


z= W'*x';
z=z';


figure(2)
hold on
for i =1:size(x,1)
    if(y(i) == 0)
        scatter(z(i,1),z(i,2),'r');
    end
    
    if(y(i) == 1)
        scatter(z(i,1),z(i,2),'g');
    end
    
 
        
end

hold off
title('PC1 vs PC2');
xlabel('Principal Component #1');
ylabel('Principal Component #2');

 

st = 2000;
sv = 2601;

sizez = size(z,1);
%%Trying with quadratic terms using validation set
bias = ones(size(x,1),1);
z= [bias z];




z_t =z(1:st,:);
z_v = z(st+1:size(z,1),:);

y74 = y(1:st,:);
y100 = y(st+1:size(z,1),:);

z1 = z_t(:,2);
z2 = z_t(:,3);


z1_val = z_v(:,2);
z2_val = z_v(:,3);

z1z1 = z1.*z1;
z2z2 = z2.*z2;
z1z2 = z1.*z2;


z1vz1v = z1_val.*z1_val;
z2vz2v = z2_val.*z2_val;
z1vz2v = z1_val.*z2_val;

z_b = ones(st,1);
z_v_b = ones(sv,1);
msqr = zeros(4,1);
Ein = zeros(4,1);
%%%%%%%%%%Linear%%%%%%%%%%%%%%%%%%

Z = [z_b z1 z2];


A=Z'*Z;
b=Z'*y74;
w=A\b;

Z_val = [z_v_b z1_val z2_val];




fit = Z_val*w;
fit2 = Z*w;

msqr(1) = ((fit -y100)'*(fit-y100));
Ein(1) = ((fit2 -y74)'*(fit2-y74));




%%%%%%%%%%%%%%%%%%z1*z1%%%%%%%%%%%%%%



Z = [z_b z1 z2 z1z1];
A=Z'*Z;
b=Z'*y74;
w=A\b;




Z_val = [z_v_b z1_val z2_val z1vz1v];



fit = Z_val*w;
fit2 = Z*w;

msqr(2) = ((fit -y100)'*(fit-y100));
Ein(2) = ((fit2 -y74)'*(fit2-y74));

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%adding z2*z2

Z = [z_b z1 z2 z1z1 z2z2];
A=Z'*Z;
b=Z'*y74;
w=A\b;

Z_val = [z_v_b z1_val z2_val z1vz1v z2vz2v];

fit = Z_val*w;
fit2 = Z*w;

msqr(3) = ((fit -y100)'*(fit-y100));
Ein(3) = ((fit2 -y74)'*(fit2-y74));
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%adding z1*z2
Z = [z_b z1 z2 z1z1 z2z2 z1z2];
A=Z'*Z;
b=Z'*y74;
w=A\b;

Z_val = [z_v_b z1_val z2_val z1vz1v z2vz2v z1vz2v];

fit = Z_val*w;
fit2 = Z*w;

msqr(4) = ((fit -y100)'*(fit-y100));
Ein(4) = ((fit2 -y74)'*(fit2-y74));


disp(Ein);
disp(msqr);

error = msqr;
figure(4)
hold on


title('Error and quadratic terms');
xlabel('quadratic terms');
ylabel('Eout Error');


set(gca, 'XTick' ,[1 2 3 4]);
set(gca, 'XTickLabel',{'Linear','z1z1','z2z2','z1z2'});
plot(msqr/sv);
plot(Ein/st);
hold off;


%Using full data to calculate in sample accuracy

% z1 = z(:,2);
% z2 = z(:,3);
% z1z1 = z1.*z1;
% z2z2 = z2.*z2;
% Z= [z z1z1 z2z2] ;
% 

Z= z;

A=Z'*Z;
b=Z'*y;
w=A\b;


fit = Z*w;

confusionInmatrix = zeros(2);
rows= size(fit,1);
ncorrectIn =0;
for i=1:rows
    bin = 1;
    if (fit(i) <= 0.4) 
        bin=1;
    end
   
    if(fit(i)> 0.4)
        bin =2;
    end
    if(bin -1 == y(i))
        ncorrectIn = ncorrectIn+1;
    end
    labelj=y(i)+1;
    labeli= bin;
    
    disp(confusionInmatrix(labeli,labelj));
    confusionInmatrix(labeli,labelj) = confusionInmatrix(labeli,labelj) +1;
end

Str = sprintf('Confusion matrix using optimal weights and linear');
disp(Str);
disp(confusionInmatrix);
accuracy_opt= ncorrectIn/size(z,1);
Str = sprintf('Accuracy =%d and Correct predictions =%d',accuracy_opt,ncorrectIn);

disp(Str);




