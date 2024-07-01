% > Bio-Heat Equation PDE without observer
%
% 
% Author: Eugenio Bugli
% June/July 2024
% 
%
clear all
clc
syms t x real
% parameters of the Bio-Heat Equation for a muscolar tissue

rho = 1050; % tissue density [kg/(m^3)]
rho_b = 1043; % blood density [kg/(m^3)]
c_tissue = 3639; % specific heat of the tissue [J/(kg*K)]
c_b = 3825; % specific heat of blood [J/(kg*K)]
k_eff = 5; % termal conductivity of the tissue [W/(m*K)]
T_a = 37; % arterial blood temperature [C]
Q = 0; % no internal source of heat
w_b = 2.22*10^-3; % blood perfusion rate [s^-1]
w_min = 0.43*10^-4; % perfusion rate min value [s^-1]
w_max = 3.8*10^-3; % perfusion rate max value [s^-1]

q_0 = 16;
beta = 15;
L_0 = 0.05;
tf = 1800;
T_M_a = 45-37;

% Bio-Heat Equation after transformations:

% 
%  (rho*c_tissue*L_0^2)                                  (rho_b*c_b*w_b*L_0^2)                  Q * L_0^2
% ---------------------- * d_tau(theta) = d_xx(theta) - ----------------------- * theta + -----------------------
%      (tf*k_eff)                                                k_eff                       (T_M - T_a)*k_eff
% 
%                     a1 * d_tau(theta) = d_xx(theta) - a2*theta + a3

a1 = rho*c_tissue*(L_0^2)/(tf*k_eff); % 1.0614 
a2 = - rho_b*c_b*w_b*(L_0^2)/k_eff; % - 4.4283
a3 = 0; % since Q = 0

% use pdepe routine to solve a system of parabolic and elliptic PDEs with one spatial variable x and time t
% parameters:

x = linspace(0,1,100);
t = linspace(0,1,100);

% routine :
% (for reference write on terminal "edit pdepe" and read the comments)

m = 0; % slab
res = pdepe(m, @bio_heat_eq, @bio_heat_ic, @bio_heat_bc, x, t);

T = res(:,:,1);

% numerical solution:

surf(x,t,T);
xlabel('Distance x')
ylabel('Time t')
zlabel('T(x,t)')
view([150 25])


function [c, s, f] = bio_heat_eq(x, t, u, dudx)
    %  a1 * d_tau(theta) = d_xx(theta) - a2*theta + a3
    c = 1.0614;
    f = dudx;
    s = - 4.4283*u;
end

function u0 = bio_heat_ic(x)
    % this function returns the initial condition
    u0 = (2/4)*(x^4) + 15*((x - 1)^2)*x/8; % initial cond eq and observer
end


function [pl,ql,pr,qr] = bio_heat_bc(xl,ul,xr,ur,t)
    % this function return the boundary conditions
    % xl, Tl --> input for the left boundary (i.e. x = 0)
    % xr, Tr --> input for the right boundary (i.e. x = 1)
    % t = time variable
    pl = ul;
    ql = 0; % since there is no flux
    pr = - 2;
    qr = 1;
end