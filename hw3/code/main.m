%读取训练数据
input=csvread('train_data.csv');
output=csvread('train_targets.csv');

%把输出从1维变成10维  
for i=1:size(input,1)  %1为行数
    switch output(i)  
        case 0  
            output1(i,:)=[1 0 0 0 0 0 0 0 0 0];  
        case 1  
            output1(i,:)=[0 1 0 0 0 0 0 0 0 0];  
        case 2  
            output1(i,:)=[0 0 1 0 0 0 0 0 0 0];  
        case 3  
            output1(i,:)=[0 0 0 1 0 0 0 0 0 0];
        case 4
            output1(i,:)=[0 0 0 0 1 0 0 0 0 0];  
        case 5  
            output1(i,:)=[0 0 0 0 0 1 0 0 0 0];  
        case 6  
            output1(i,:)=[0 0 0 0 0 0 1 0 0 0];  
        case 7  
            output1(i,:)=[0 0 0 0 0 0 0 1 0 0];
        case 8 
            output1(i,:)=[0 0 0 0 0 0 0 0 1 0];  
        case 9  
            output1(i,:)=[0 0 0 0 0 0 0 0 0 1];
    end  
end

% 对训练的特征进行归一化  
% [trainInput,inputps]=mapminmax(input'); 
% [trainInput,mini,maxi]=premnmx(input');
trainInput=input';

% 参数的初始化  
inputNum = 400;%输入层的节点数  
hiddenNum = 100;%隐含层的节点数  
outputNum = 10;%输出层的节点数  
  
% 权重和偏置的初始化  
w1 = rands(inputNum,hiddenNum);  
b1 = rands(hiddenNum,1);  
w2 = rands(hiddenNum,outputNum);  
b2 = rands(outputNum,1);  
  
% 学习率  
yita = 0.1; 


% 网络的训练  
for r = 1:30    %训练次数
    E(r) = 0;% 统计误差  
    for m = 1:size(input,1)
        % 信息的正向流动  
        x = trainInput(:,m);  
        % 隐含层的输出  
        hidden=w1'*x+b1;
        hiddenOutput=g(hidden);
%         for j = 1:hiddenNum  
%             hidden(j,:) = w1(:,j)'*x+b1(j,:);  
%             hiddenOutput(j,:) = g(hidden(j,:));  
%         end  
        % 输出层的输出  
        outputOutput(:,m) = w2'*hiddenOutput+b2;  
          
        % 计算误差  
        e = output1(m,:)'-outputOutput(:,m);  
        E(r) = E(r) + sum(abs(e));  
          
        % 修改权重和偏置  
        % 隐含层到输出层的权重和偏置调整  
        dw2 = hiddenOutput*e';  
        db2 = e;  
          
        % 输入层到隐含层的权重和偏置调整  
%         partOne=hiddenOutput*(1-hiddenOutput);
        
%         hiddenOutput1=zeros(100,99);
%         hiddenOutput1=[hiddenOutput hiddenOutput1];
%         partOne=(hiddenOutput1*hiddenOutput)';
        partOne=(hiddenOutput.*(1-hiddenOutput))';
        partTwo=(w2*e)';
%         for j = 1:hiddenNum  
%             partOne(j) = hiddenOutput(j)*(1-hiddenOutput(j));  
%       %      partTwo(j) = w2(j,:)*e;  
%         end  
%         x1=zeros(400,99);
%         x1=[x x1];
        dw1=x*(partOne.*partTwo);
%         partOne1=zeros(100, 99);
%         partOne1=[partOne' partOne1];
        db1=(partOne.*partTwo)';
%         for i = 1:inputNum  
%             for j = 1:hiddenNum  
%                 dw1(i,j) = partOne(j)*x(i,:)*partTwo(j);  
%                 db1(j,:) = partOne(j)*partTwo(j);  
%             end  
%         end  
          
        w1 = w1 + yita*dw1;  
        w2 = w2 + yita*dw2; 
        b1 = b1 + yita*db1;  
        b2 = b2 + yita*db2; 
        
    end  
%     csvwrite(['outputw1.csv'],w1);
end



test_input=csvread('test_data.csv');
% [test_input1,testinputps]=mapminmax(test_input');
test_input1=test_input';
% [test_input1,mint,maxt]=premnmx(test_input');
for m = 1:size(test_input,1)
%     t = test_input1(m,:)';  
    hiddenTest=w1'*test_input1(:,m)+b1;
    hiddenTestOutput=g(hiddenTest);
    outputOfTest(:,m) = w2'*hiddenTestOutput+b2; 
%     for j = 1:hiddenNum  
%         hiddenTest(j,:) = w1(:,j)'*testInput(:,m)+b1(j,:);  
%         hiddenTestOutput(j,:) = g(hiddenTest(j,:));  
%     end     
end
% outputOfTest=mapminmax('reverse',test_input1,testinputps);
% csvwrite(['testw1.csv'],w1);
%     outputOfTest = w2'*hiddenTestOutput+b2; 
for m=1:size(test_input,1)
%     [maxoutput,index]=max(outputOfTest(:,m));
%     output_fore(m)=index-1;
%     output_max(m)=maxoutput;
   output_fore(m)=find(outputOfTest(:,m)==max(outputOfTest(:,m)))-1;  
end 
csvwrite(['test_predictions.csv'],output_fore');
% csvwrite(['outputTest.csv'],outputOfTest');






