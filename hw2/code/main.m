% old_l=0;    %记录上次计算的l
% n=0;    %计算迭代次数
%b=[0;0;1];  %初始参数 （自定义）
% b=zeros(31,1);
% b(31,1)=1;
% for i=1:31
%     fprintf('%d',b(1,i));
% end

data=csvread('data.csv');   %x
targets=csvread('targets.csv');   %y
%fprintf('%d ',size(data,2));
add=ones(1,size(data,1));
data=[data add'];
% fprintf('%d ',size(data,2));
% for i=1:31
%     fprintf('%d ',newData(1,i));
% end

% indices =crossvalind('Kfold', size(data,1), 10);

for t=1:10
     n=0;
     old_l=0;
     b=zeros(31,1);
     b(31,1)=1;
%     test=(indices==t);
%     train=~test;
%     data_train=data(train,:); %训练集
%     targets_train=targets(train,:);
%     data_test=data(test,:);  %测试集
%     targets_test=targets(test,:);

    num=size(data,1)/10;
    begin=(t-1)*56+1;
    en=begin+num-1;
    data_test=data;
    data_test=data_test(begin:en,:);  %测试集
    targets_test=targets;
    targets_test=targets_test(begin:en,:);
    data_train=data;
    data_train(begin:en,:)=[];
    targets_train=targets;
    targets_train(begin:en,:)=[];
    
    while(1)
       cur_l=0;
       bx=zeros(size(data_train,1),1);
       %计算当前参数下的l
       for i=1:size(data_train,1)
            bx(i) = b.'*data_train(i,:)';
            cur_l = cur_l + ((-targets_train(i)*bx(i)) )+log(1+exp(bx(i)));
       end

       %迭代终止条件
%        if abs(cur_l-old_l)<0.1  
%            break;
%        end
%fprintf('%d\n',n);
       if n==5
           break;
       end


       %更新参数(牛顿迭代法)以及保存当前l
       n=n+1;
       old_l = cur_l;
       p1=zeros(size(data_train,1),1);
       dl=0;
       d2l=0;

       for i=1:size(data_train,1)
            p1(i) = 1 - 1/(1+exp(bx(i)));
            dl = dl - data_train(i,:).'*(targets_train(i)-p1(i));
            d2l = d2l + data_train(i,:).' * data_train(i,:)*p1(i)*(1-p1(i));
%             if (d2l==0)
%                 fprintf('wrong\n');
%                 break;
%             end
       end
       b = b - pinv(d2l)*dl;
    end
    
     res=zeros(size(data_test,1),2);
    for i=1:size(data_test,1)
        res(i,1)=i+size(data_test,1)*(t-1);
        pre=1/(1+exp(-(b'*data_test(i,:)')));
%         if pre>0.5
%             res(i,2)=1;
%         end
%         if pre==0.5
%             res(i,2)=0.5;
%         end;
%         if pre<0.5
%             res(i,2)=0;
%         end
        if pre>=0.5
            res(i,2)=1;
        else
            res(i,2)=0;
        end
    end
    
%   res=1/(1+exp(-(b'*data_test')));
    csvwrite(['fold',num2str(t),'.csv'],res);
end