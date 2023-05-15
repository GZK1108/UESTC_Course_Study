function lab4
c = [-17; -15];
a = [3, 5; 6, 5; 2, 6];
b = [150; 160; 180];
[x, y] = linprog(c, a, b);
x
y = -y