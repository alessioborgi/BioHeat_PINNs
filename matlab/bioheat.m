% > Bio-Heat Equation PDE 
%
% 
% Author: Eugenio Bugli
% June 2024
% 
%
clear all
clc
syms t x real
% parameters of the Bio-Heat Equation for a muscolar tissue

rho = 1050; % tissue density [kg/(m^3)]
rho_b = 1043; % blood density [kg/(m^3)]
c = 3639; % specific heat of the tissue [J/(kg*K)]
c_b = 3825; % specific heat of blood [J/(kg*K)]
k_eff = 5; % termal conductivity of the tissue [W/(m*K)]
T_a = 37; % arterial blood temperature [C]
Q = 0; % no internal source of heat
w_b = 2.22*10^-3; % blood perfusion rate [s^-1]
w_min = 0.43*10^-4; % perfusion rate min value [s^-1]
w_max = 3.8*10^-3; % perfusion rate max value [s^-1]

q_0 = 16;
rho = 1050;
rho_b = 1043;
c_tissue = 3639;
k_eff = 5;
L_0 = 0.05;
c_b = 3825;
w_b = 2.22*10^-3;
tf = 1800;
T_M_a = 45-37;
alpha = rho*c_tissue/k_eff;
a1 = tf/(alpha*L_0^2)
a2 = c_b*L_0^2/k_eff
W = rho_b*w_b
% use pdepe routine to solve a system of parabolic and elliptic PDEs with one spatial variable x and time t
% parameters:

% > m: scalar that represents the symmetry of the problem (slab, cylindrical, or spherical). 
% > pdefun: equations
% > icfun: initial value
% > bcfun: boundary condition

% > xmesh: The pdepe function returns values of the solution on a mesh provided in xmesh.
% > tspan: The ordinary differential equations (ODEs) resulting from discretization in space are integrated to obtain approximate solutions at the times specified in tspan.

x = linspace(0,1,100);
t = linspace(0,1,100); % tf = 1800s

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
    % this 3 elements define the equation expected from the solver
    rho = 1050;
    rho_b = 1043;
    c_tissue = 3639;
    k_eff = 5;
    L_0 = 0.05;
    c_b = 3825;
    w_b = 2.22*10^-3;
    tf = 1800;
    % -------

    c = 1.061375;% (tf*k_eff/(rho*c_tissue*L_0^2));
    f = dudx;
    s = 1.9125*2.48*u/4;% - (rho_b*w_b*c_b/(rho*c)) * T;
end

function T0 = bio_heat_ic(x)
    % this function returns the initial condition
    q_0 = 16;
    Beta = 15;
    L_0 = 0.05;
    T_M_a = 45-37; 
    % -------
    T0 = (q_0*x^4)/(4*T_M_a) + ( 15*x/(T_M_a) )*(x - 1)^2; % initial cond eq and observer
end


function [pl,ql,pr,qr] = bio_heat_bc(xl,ul,xr,ur,t)
    % this function return the boundary conditions
    % xl, Tl --> input for the left boundary (i.e. x = 0)
    % xr, Tr --> input for the right boundary (i.e. x = 1)
    % t = time variable
    T_M_a = 45-37; 
    q_0 = 16;
    % -------
    pl = ul;
    ql = 0; % since there is no flux
    pr = - q_0/(T_M_a);
    qr = 1;
end