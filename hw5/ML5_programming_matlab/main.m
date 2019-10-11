%%
%load data
load('all_data.mat');
X=full(train_data);
y=train_targets;
Xt=full(test_data);
%smooth
alpha=1;

%%
%calculates the priors for each class
% Extract the class labels  
priors.class = unique(y);  
% Initialize the priors.value  
priors.value = zeros(1, length(priors.class));  
% Calculate the priors  
for i = 1 : length(priors.class)
    for j = 1 : length(y)
%     priors.value(y(i)+1) =   priors.value(y(i)+1)+1;
        if y(j) == priors.class(i)
            priors.value(i) = priors.value(i) + 1;
        end
%    priors.value(i) = (sum(y == class(i))) / (length(y));  
    end
    priors.p(i)=(priors.value(i)+alpha)/(length(y)+alpha*length(priors.class));
end

%%
% Learn the features by calculating likelihood  
%离散
%取值为0时各属性在各类别上的概率
% P_0_classi=zeros(length(priors.class),size(X,2)+1);
P_0_classi=zeros(length(priors.class),2500);
%取值为1时各属性在各类别上的概率
% P_1_classi=zeros(length(priors.class),size(X,2)+1);
P_1_classi=zeros(length(priors.class),2500);
%第一列为类别名字
% P_0_classi(:,1)=priors.class;
% P_1_classi(:,1)=priors.class;
for i = 1 : 2500
%     sumOfClass=zeros(1,length(priors.class));
    for j = 1 : size(X,1)  
%         for k = 1 : length(priors.class)
%             if y(j)==priors.class(k)
%                 sumOfClass(k)= sumOfClass(k)+1;
%             end
%         end
%         
        for k = 1 : length(priors.class)
            if X(j,i)==0
                if y(j)==priors.class(k)
                    P_0_classi(k,i)= P_0_classi(k,i)+1;
                end
            end
            if X(j,i)==1
                if y(j)==priors.class(k)
                    P_1_classi(k,i)= P_1_classi(k,i)+1;
                end
            end
        end
    end
    for k = 1 : length(priors.class)
        P_0_classi(k,i)=(P_0_classi(k,i)+alpha)/(priors.value(k)+alpha*2);
        P_1_classi(k,i)=(P_1_classi(k,i)+alpha)/(priors.value(k)+alpha*2);
    end
end

%连续
%方差
variance=zeros(length(priors.class),2500);
%均值
meanValue=zeros(length(priors.class),2500);
%第一列为类别名字
% variance(:,1)=priors.class;
% meanValue(:,1)=priors.class;
for i = 2501 : 5000
    position=zeros(1,length(priors.class));
    for k = 1 : length(priors.class)
            matrix.class(k).value=zeros(1,priors.value(k));
    end
    
    for j = 1 : size(X,1)
        for k = 1 : length(priors.class)
            if y(j)==priors.class(k)
                position(1,k)=position(1,k)+1;
                matrix.class(k).value(1,position(1,k))=X(j,i);
            end
        end
    end
    
    for k = 1 : length(priors.class)
            variance(k,i-2500)=var( matrix.class(k).value);
            meanValue(k,i-2500)=mean( matrix.class(k).value);
    end
end

%%
%预测
result=zeros(size(Xt,1),1);
%最大方差
maxVar=max(max(variance));
for i = 1 : size(Xt,1)
    ln_res=zeros(length(priors.class),1);
    for k = 1 : length(priors.class)
        ln_res(k,1)=log(priors.p(k));
        for j = 1 : 5000
            if j <= 2500
                if Xt(i,j)==0
    %                 if P_0_classi(k,j) == 0
    %                     ln_res(k,1)=0;
    %                     break;
    %                 end
                    if P_0_classi(k,j) ~= 0
                       ln_res(k,1) = ln_res(k,1) + log(P_0_classi(k,j));
                    end
                end
                if Xt(i,j)==1
    %                 if P_1_classi(k,j) == 0
    %                     ln_res(k,1)=0;
    %                     break;
    %                 end
                    if P_1_classi(k,j) ~= 0
                        ln_res(k,1) = ln_res(k,1) + log(P_1_classi(k,j));
                    end
                end
            end
            
            if j > 2500
                mean_value=meanValue(k,j-2500);
                var_value=variance(k,j-2500);
                if var_value==0
                    var_value=maxVar * eps;
                end
%                 if var_value ~= 0
%                     p = (1/(sqrt(2*pi*var_value))) * exp(-(Xt(i,j)-mean_value)^2/(2*var_value));
%                     p = exp(-(Xt(i,j)-mean_value)^2/(2*var_value)) /(sqrt(2*pi*var_value));
                    p = -(Xt(i,j)-mean_value)^2/(2*var_value)  - log(sqrt(2*pi*var_value));
%                     if p ~= 0
                        ln_res(k,1) = ln_res(k,1) + p;
%                     end
%                     if p == 0
%                         count = count + 1;
% %                         ln_res(k,1)=0;
% %                         break;
%                     end
%                 end
            end
        end
        
%         for j = 2501 : 5000
%             mean_value=meanValue(k,j-2500);
%             var_value=variance(k,j-2500);
%             if var_value ~= 0
%                 p = (1/(sqrt(2*pi)*var_value))*exp(-(Xt(i,j)-mean_value)^2/(2*(var_value^2)));
%                 if p ~= 0
%                     ln_res(k,1) = ln_res(k,1) + log(p);
%                 end
%             end
%         end
    end
    
    
    
    [ma,I]=max(ln_res);
    result(i,1)=priors.class(I); 
end

csvwrite('test_predictions.csv',result);
