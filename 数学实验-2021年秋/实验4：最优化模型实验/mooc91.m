function mooc91
c = [-2.5; -5; -10];
a = [-2.5, -5, -10; -2, -2, 1];
b = [-50; 0];
aeq = [1, 1, 1];
beq = 100;
lb = [10, 0, 20];
ub = [30, 90, 80];
[x, y] = linprog(c, a, b, aeq, beq, lb, ub);
x
y = -y