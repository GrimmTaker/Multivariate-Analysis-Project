data= xlsread('/Users/garima/Documents/WSU/WSU_Courses/Fall_2015/Neural Networks/spambase/spambase_labels_changed.xlsx');
size(data);
x= data(:,(1:57));
y= data(:,58);

bias = ones(size(x,1),1);
V=[ bias x];
A=V'*V;
b=V'*y;
w=A\b;
rows = size(x,1);
fit = V*w;
Ein = ((fit-y)'*(fit -y))/rows;
confusionInmatrix = zeros(2);
ncorrectIn =0;
for i=1:rows
    bin = 1;
    if fit(i) <= 1.5 
        bin=1;
    end
   
    if(fit(i)> 1.5)
        bin =2;
    end
    if(bin == y(i))
        ncorrectIn = ncorrectIn+1;
    end
    labelj=y(i);
    labeli= bin;
    
    %disp(confusionInmatrix(labeli,labelj));
    confusionInmatrix(labeli,labelj) = confusionInmatrix(labeli,labelj) +1;
end
 
ecvm=0;
ncorrectEval=0;
confusionValmatrix = zeros(2);
for i=1:rows
    Vtemp=V;
    ytemp = y;
    Vm1 = removerows(Vtemp,i);
    ym1 = removerows(ytemp,i);
    
    A=Vm1'*Vm1;
    b=Vm1'*ym1;
    w=A\b;
    
    fiti= V(i,:)*w;
    size(fiti);
    
    risq= (fiti -y(i))^2;
    ecvm = ecvm +risq;
    bin = 1;
    if fiti <= 1.5 
        bin=1;
    end
   
    if(fiti> 1.5)
        bin =2;
    end
    if(bin == y(i))
        ncorrectEval = ncorrectEval+1;
    end
    labeli = bin;
    labelj =y(i);
    confusionValmatrix(labeli,labelj) = confusionValmatrix(labeli,labelj) +1;
    
    
    
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
val_accuracy = ncorrectEval/rows;
eout_accuracy = 1- val_accuracy;
DispAccuracyVal= sprintf('\nAccurate predictions (Leave one out) = %d , Accuracy =%f',ncorrectEval,val_accuracy);
disp(DispAccuracyVal);
ecvm=ecvm/rows;

crossValerror = sprintf('\n Leave one out Error(Square Residuals) is = %f, Leave one out error(Accuracy)Ecv-1 = %f',ecvm,eout_accuracy);
disp(crossValerror);
Aout=ecvm;
EinA = 1- Ein_accuracy;
Eout = 1- eout_accuracy;
e= EinA-Eout;
epsilon = sprintf('\n Espilon = %f',e);
disp(epsilon);
delta = 2*exp(-2*e*e*4601);
confidence =1-delta;
confidencestr = sprintf('\n Confidence = %f',confidence);

disp(confidencestr);



