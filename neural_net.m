
rng(10,'twister');

%Use this newdata = data(randperm(size(data,1)),:)
data = xlsread('/Users/garima/Documents/WSU/WSU_Courses/Fall_2015/Neural Networks/spambase/spambase_randomized_orig_labels.xlsx' );
st = 2000;
sv = 2601;

x_raw= data(:,(1:57));
y_raw= data(:,58);


x1 = x_raw(1:st,:);
r = y_raw(1:st)';
bias_x = ones(st,1);

x = [bias_x x1];

x1_val = x_raw(st+1:size(x_raw,1),:);
r_val = y_raw(st+1:size(y_raw,1),:)';

bias_val = ones(sv,1);

x_val = [bias_val,x1_val];


h=5;
Eval = zeros(h,1);
for i=5:5
nodes = i;


wmin=-.01;
wmax=.01;
w = wmin+ rand(nodes,58)*(wmax-wmin);
v = wmin+ rand(nodes+1,1)*(wmax-wmin);
z= zeros(size(x,1),1);
ch_w = zeros(nodes,58);
prev_ch_w = zeros(nodes,58);
etha = 0.0001;
alpha = 0;
epochs = 2000;
E = zeros(epochs,1);
E_val_m = zeros(epochs,1);
for m=1:epochs
    y = zeros(st,1);
    y_train_raw = y;
    
    y_val = zeros(sv,1);
    
    y_val_raw = y_val;
    for t=1:size(x,1)

        z(t,1)= 1;

        for k=1:nodes

            z(t,k+1)=1/(1+exp(-1*(x(t,:)*w(k,:)')));

        end

        y(t) = v'*z(t,:)';
        y_train_raw(t) = y(t);

        if(y(t) <= 0.4)
            y(t)=0;
        end

        if(y(t) > 0.4)
            y(t) = 1;
        end

        
        prev_ch_w = ch_w;
        for l =1:nodes

        ch_w(l,:) = 1*etha*(r(t)-y(t))*v(l+1)*z(t,l+1)*(1-z(t,l+1))*x(t,:);

        w(l,:) = w(l,:)+ch_w(l,:)+alpha *prev_ch_w(l,:);
        end
        ch_v = 1*etha*(r(t)-y(t))*z(t,:)';
        v = v + ch_v;

    end
    
    

     E(m) = sum((y-r').^2);
    z_val= zeros(size(x_val,1),1);
 for  p=1 : size(x_val,1)

    

    z_val(p,1) =1;

    for k=1:nodes
        z_val(p,k+1)=1/(1+exp(-1*x_val(p,:)*(w(k,:))'));
    end

    y_val(p) = v'*z_val(p,:)';
    y_val_raw(p) = y_val(p);

    if(y_val(p) <= 0.4)
        y_val(p)=0;
    end

        if(y_val(p) > 0.4)
            y_val(p) = 1;
        end

end
E_val_m(m) = sum((y_val-r_val').^2);
        




%disp(v);
%loglog(E);
%axis([0 1000 0 .001]);
%disp(E(m));
%z_val= zeros(100,nodes);
end
% for  p=1 : size(x_val,1)
% 
%     %z_val =zeros(100,10);
%     z_val(p,1) =1;
% 
%     for k=1:nodes
%         z_val(p,k+1)=1/(1+exp(-1*x_val(p,:)*(w(k,:))'));
%     end
% 
%     y_val(p) = (v'*z_val(p,:)');
% 
%         if(y_val(p) < 0.4)
%         y_val(p)=0;
%         end
% 
%         if(y_val(p) >= 0.4)
%             y_val(p) = 1;
%         end
% 
% 
%         if(y_val(p) == r_val(p))
%             Eval(i) = Eval(i) +1;
%         end
% 
% 
% 
% 
% 
% 
% 
% end
end
% figure(1);
% plot(Eval)
% xlabel(' Nodes (Excluding Bias)');
% ylabel('Accuracy');

figure(2);
hold on;
plot(E/st,'g');
plot(E_val_m/sv, 'r');
hold off;
xlabel('epochs');
ylabel('Mean sq error');

