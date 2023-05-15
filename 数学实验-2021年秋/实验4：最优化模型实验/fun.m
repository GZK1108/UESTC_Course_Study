function [x0, y0] = fun
f = inline('-exp(x) * x * sin(x)');
fz = inline('exp(x) * x * sin(x)');
[x0, y0] = fminbnd(f, 8, 9);%fminbnd去最值只能在一定范围内，范围大了取不到所有的点
y0 = -y0;
ezplot(fz, [0, 9])
end