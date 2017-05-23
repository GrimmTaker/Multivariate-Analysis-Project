
%Using training and Testing data

data= xlsread('/Users/garima/Documents/WSU/WSU_Courses/Fall_2015/Neural Networks/spambase/spambase_labels_changed.xlsx');
size(data);
x= data(:,(1:57));
y= data(:,58);

x_train = x(1:1000,:);
y_train = y(1:1000);

x_test = x(1001:4061,:);
y_test = y(1001:4061,:);
bias = ones(size(x_train,1),1);
V=[ bias x_train];
A=V'*V;
b=V'*y_train;
w=A\b;
rows = size(x_train,1);
fit = V*w;
Ein = ((fit-y_train)'*(fit -y_train))/rows;
confusionInmatrix = zeros(2);
ncorrectIn =0;
for i=1:rows
    bin = 1;
    if (fit(i) <= 1.5 )
        bin=1;
    end
   
    if(fit(i)> 1.5)
        bin =2;
    end
    if(bin == y(i))
        ncorrectIn = ncorrectIn+1;
    end
    labelj=y_train(i);
    labeli= bin;
    
    %disp(confusionInmatrix(labeli,labelj));
    confusionInmatrix(labeli,labelj) = confusionInmatrix(labeli,labelj) +1;
end
 
bias_test = ones(size(x_test,1),1);
V_test=[ bias_test x_test];
    

fit_test= V_test*w;
row_test = size(x_test,1);
Eout = ((fit_test-y_test)'*(fit_test -y_test))/row_test;


confusionValmatrix = zeros(2);
ncorrectEval =0;

for i=1:row_test;
   
    
    
    if (fit_test(i) < 1.5 )
        bin=1;
    end
   
    if(fit_test(i)> 1.5)
        bin =2;
    end
    
    if(bin == y_test(i))
        ncorrectEval = ncorrectEval+1;
    end
  
    
    
    labeli = bin;
    labelj =y_test(i);
    confusionValmatrix(bin,y_test(i)) = confusionValmatrix(bin,y_test(i)) +1;
    
    
    
end
Str = sprintf('Confusion matrix for in sample');
disp(Str);
disp(confusionInmatrix);
accuracyIn = ncorrectIn/rows;
DispAccuracyIn= sprintf('\nAccurate predictions (in sample) = %d ,Accuracy (in sample) = %f',ncorrectIn,accuracyIn);
disp(DispAccuracyIn);
Ein_accuracy = 1- accuracyIn;
insampleerror=  sprintf('\nIn sample error (Square Residuals) = %f, In sample Accuracy Error Ein =%f',Ein,Ein_accuracy);
disp(insampleerror);
Str = sprintf('\nConfusion matrix for validation(leave one out)');
disp(Str);
disp(confusionValmatrix);
val_accuracy = ncorrectEval/row_test;
eout_accuracy = 1- val_accuracy;
DispAccuracyVal= sprintf('\nAccurate predictions (Training Set) = %d , Accuracy =%f',ncorrectEval,val_accuracy);
disp(DispAccuracyVal);


Valerror = sprintf('\n E Test is = %f',Eout);
disp(Valerror);
%Aout=ecvm;
EinA = 1- Ein_accuracy;
Eout = 1- eout_accuracy;
e= EinA-Eout;
epsilon = sprintf('\n Espilon = %f',e);
disp(epsilon);
delta = 2*exp(-2*e*e*size(x_test,1));
confidence =1-delta;
confidencestr = sprintf('\n Confidence = %f',confidence);

disp(confidencestr);



