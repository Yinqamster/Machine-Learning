data=csvread('data.csv');   %x
targets=csvread('targets.csv');   %y
T=[1 5 10 100];
indices =crossvalind('Kfold', size(data,1), 10);
for i=1:4
    for j=1:10
        count=0;
%         num=floor(size(data,1)/10);
%         begin=(j-1)*num+1;
%         en=begin+num-1;
%         data_test=data;
%         data_test=data_test(begin:en,:);  %²âÊÔ¼¯
%         targets_test=targets;
%         targets_test=targets_test(begin:en,:);
%         data_train=data;
%         data_train(begin:en,:)=[];
%         targets_train=targets;
%         targets_train(begin:en,:)=[];

        D=ones(size(data,1),1);
        D(:,1)=1/size(data,1);
            
        test=(indices==j);
        train=~test;
        data_train=data(train,:);
        data_test=data(test,:);
        targets_train=targets(train,:);
        targets_test=targets(test,:);
        
        
        res=zeros(size(targets_test,1),2);
        pre=zeros(size(targets_test,1),2);
        [p,q]=find(indices==j);
        res(:,1)=p;
        for t=1:T(i)
            if t>30
                break;
            end
%             count=count+1;
            for k=1:size(data_train,1)
                data_train(i,:)=D(i)*data_train(i,:);
            end
            [out1,out2,b]=LR(data_train,targets_train,data);
%             B=glmfit(data_train,[targets_train ones(size(targets_train,1),1)],'binomial','link','logit');
%             pre=B(1)+data_test*B(2);
%             for k=1:size(pre,1)
%                 if pre>=0.5
%                     res(k)=1;
%                 else
%                     res(k)=0;
%                 end
%             end
            acc=sum(targets(:,1)==out1(:,1))/size(out1,1);
%             acc=sum(targets_train(:,1)==out(:,1))/size(out,1);
            e=1-acc;
            if e==0
                break;
            end
% e=1-acc;
            if e > 0.5
                break;
            end
%             if(e~=0)
                alpha=1/2 * (log(acc)-log(e));
%             else
%                 alpha=1;
%             end
            
            z=sum(D,'omitnan');
%             z=1;
            for k=1:size(targets_train,1)
                if targets_train(k)==out2(k)
                    D(k)=D(k)*exp(-alpha)/z;
                end
                if targets_train(k)~=out2(k)
                    D(k)=D(k)*exp(alpha)/z;
                end
                
            end
            
            for k=1:size(data_test,1)
                r=1/(1+exp(-(b'*data_test(k,:)')));
                if r>=0.5
                    r=1;
                end
                if r<0.5
                    r=-1;
                end
                pre(k)=pre(k)+alpha*r;
            end  
        end
        for k=1:size(data_test,1)
            res(k,2)=sign(pre(k));
            if res(k,2)==-1
                res(k,2)=0;
            end
        end  
        
%         out=LR(data_test,targets_test);
%         count
        csvwrite(['experiments/base',num2str(T(i)),'_fold',num2str(j),'.csv'],res);
    end
end