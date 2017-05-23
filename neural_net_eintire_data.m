
rng(100,'twister');

%Use this newdata = data(randperm(size(data,1)),:)
data = xlsread('/Users/garima/Documents/WSU/WSU_Courses/Fall_2015/Neural Networks/spambase/spambase_randomized_orig_labels.xlsx' );
st = 2000;
sv = 2601;
newdata = data(randperm(size(data,1)),:);



nodes=4;


 x_raw= newdata(:,(1:57));
    y_raw= newdata(:,58);
    bias_x = ones(size(x_raw,1),1);

    x = [bias_x x_raw];

    r = y_raw;


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
    
    
    %y = zeros(size(x,1),1);
    %y_train_raw = y;
    
 
    for t=1:size(x,1)

        z(t,1)= 1;

        for k=1:nodes

            z(t,k+1)=1/(1+exp(-1*(x(t,:)*w(k,:)')));

        end

        y_z(t) = v'*z(t,:)';
        y_train_raw(t) = y_z(t);
        y(t)=1;
        if(y_z(t) < 0.4)
            y(t)=0;
        end

        if(y_z(t) >= 0.4)
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
    
   
%disp(v);
%loglog(E);
%axis([0 1000 0 .001]);
%disp(E(m));
%z_val= zeros(100,nodes);
end

% 

% figure(1);
% plot(Eval)
% xlabel(' Nodes (Excluding Bias)');
% ylabel('Accuracy');

figure(2);
hold on;
plot(E/st,'g');

hold off;
xlabel('epochs');
ylabel('Mean sq error');

