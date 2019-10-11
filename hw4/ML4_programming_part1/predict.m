function pred = predict(Xt, model)

gamma = model.Parameters(4);
RBF = @(u,v)( exp(-gamma.*sum( (u-v).^2) ) );
len = length(model.sv_coef);
sizeXt = size(Xt,1);
y = zeros(sizeXt,1);
pred=zeros(sizeXt,1);
for j = 1:sizeXt
    for i = 1:len
        u = model.SVs(i,:);
        y(j,1) = y(j,1) + model.sv_coef(i)*RBF(u,Xt(j,:));
    end
end
b = -model.rho;
y = y + b;

pred(y >= 0) =  0;
pred(y <  0) =  1;

end