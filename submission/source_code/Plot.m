close all;
clear all;


%------------------------------------------
% Squelette de GRAPH MATLAB
%------------------------------------------


% DATALOAD
%------------------------------------------

% dataload du fichier
data=load('Ib_HallmV_I_4_35.txt');
I = data(:,1);
V = data(:,2);

data=load('Ib_HallmV_I_4_35_descente.txt');
Im = data(:,1);
Vm = data(:,2);

B = 0.072 * I;
Bm = 0.072 * Im;

V = V - 39.18;
Vm = Vm - 39.18;

Vtot = [Vm; V];
Btot = [Bm; B];


% CALCUL
%------------------------------------------





% INTERPOLATION
%------------------------------------------
if 1
interp_1 = Btot;
interp_2 = Vtot;
interp_step = abs( interp_1(end)-interp_1(1) ) / 200 ;   
poly = polyfit(interp_1,interp_2,1);                     % degré interpolation
x = min(interp_1): interp_step : max(interp_1);          % pts dessinés
y = polyval(poly,x);
interp_pente = ( y(end)-y(1) )/( x(end)-x(1) )        % pente regr. lin.
end




% GRAPH
%------------------------------------------
% taille des points et de la police

lw=1.2; fs=16;

figure
hold on;

plot(B,V,'xr', 'linewidth',lw);
plot(Bm,Vm,'xg', 'linewidth',lw);
%plot(x,y,'-b', 'linewidth',lw);  % plot interpol

set(gca,'fontsize',fs);
grid('on');
xlabel('$ B $ [T]', 'interpreter', 'latex');
ylabel('$ V_H $ [mV]', 'interpreter', 'latex');
h = legend('$B^+$', '$B^-$','domaine utile');
set(h,'interpreter','latex');
%axis([0, 2, 0, 0.2]);

hold off;




% GRAPH MULTIPLE Y-AXIS
%------------------------------------------
if 0
lw=1.2; fs=16;

figure
[AX,H1,H2] = plotyy(T,f,T,Q);
set(get(AX(1),'Xlabel'),'String','Legende X',...
    'fontsize',fs, 'interpreter', 'latex');
set(get(AX(1),'Ylabel'),'String','Legende Y1', ...
    'fontsize',fs, 'interpreter', 'latex');
set(get(AX(2),'Ylabel'),'String','Legende Y2', ...
    'fontsize',fs, 'interpreter', 'latex');
set(H1(1),'Marker','x', 'LineStyle', 'none');
set(H2(1),'Marker','x', 'LineStyle', 'none');
set(gca,'fontsize',fs);
%set(get(AX(1),'XTick'), = [20 40]);
set(AX, 'XTick', [0 20 40 100], 'fontsize',fs);
%set(AX(2), 'XTick', 'none');
end

% OPTIONS
%------------------------------------------

% legend('courbe a', 'courbe b', ...)
% text(x,y, 'legende en ce point')
% loglog(x,y)
% semilogx(x,y)
% semilogy(x,y)
% errorbar(T,Etab,dEtab,'xb');