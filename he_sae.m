% This matlab code solves the time-independent Schrodinger equation (TISE) for
% for helium (he) atom using a single-active electron (sae) approximation Ref.[1], 
% in which the pseudospectral method is used [2].  
%
% The coefficients in the SAE potential are obtained from Ref. [1].
% Ionization energy for ground and excited states of the He atom is obtained and compared with those in Ref.[1]. 
%
% The atomic unit (au) is used in the calculation.
% Note that to get more accurate results, one would increase numerical parameters (N, a, b, etc., )
%
% Refs: [1] R.Reiff, T.Joyce, A.Jaroń-Becker and A. Becker, J. Phys. Commun. 4, 065011  (2020);
%       [2] Tsogbayar (PhD) thesis, York University, Canada (2014), available at https://yorkspace.library.yorku.ca/xmlui/handle/10315/28160
%
% Written by Tsogbayar Tsednee (PhD)
% Email: tsog215@gmail.com
% Dec 19, 2024 & University of North Dakota 
%%% 
function [] = he_sae
%
clc;
%
C0 = 1.;
Zc = 1.;
c = 2.0329;
a1 = 0.3953;
b1 = 6.1805;
%
N = 4.*128;  % number of grid points along radial distance r; you may change it
a = 0.0;     % starting point od coordinate r
b = 80.;     % end point of cooridnate r; you may change it
%
[En_1s, En_2s, En_3s, rho_1s, wr, r ] = he_sae_ham(N,a,b,C0, Zc, c, a1, b1, 0., 1); % s orbital 
[En_1s, En_2s, En_3s, rho_2s, wr, r ] = he_sae_ham(N,a,b,C0, Zc, c, a1, b1, 0., 2); % s orbital
[En_2p, En_3p, En_4p, rho_2p, wr, r ] = he_sae_ham(N,a,b,C0, Zc, c, a1, b1, 1., 1); % p orbital
[En_3d, En_4d, En_5d, rho_3d, wr, r ] = he_sae_ham(N,a,b,C0, Zc, c, a1, b1, 2., 1); % d orbital

%
rho_total = 2.*rho_1s + 0.*rho_2s + 0.*rho_2p;
%
plot(r, 4.*pi.*r.^2.*rho_total, 'b', 'LineWidth',1.5)
hold off
xlabel('$r\,(au)$', 'Interpreter','latex')
ylabel('$4\pi r^{2} \rho$', 'Interpreter','latex')
set(gca, 'FontSize', 18)
axis([0 5 0 2])
box on
%
[En_1s, En_2s, En_2p, En_3s, En_3p, En_3d] % ionization energies 
% [En_1s, En_2s, En_2p, En_3s, En_3p, En_3d]
%  -0.9444   -0.1597   -0.1285   -0.0650   -0.0567   -0.0556 vs 
%  -0.944    -0.15969  -0.12847  -0.064999 -0.056679 -0.055581 from Ref.[1]      

average_r2 = sum(wr.*rho_total.*r.^2.*r.^2.*4.*pi); % 2.313182481893269


%%%
return
end

%%%
function [En_1, En_2, En_3, rho, wr, r ] = he_sae_ham(N,a,b,C0, Zc, c, a1, b1, ell, n_ind)
%
[r,wr,D]=legDC2(N,a,b);
%
wr = wr(2:N);
r = r(2:N);
D2 = (2/(b-a))^2*D^2;
D2 = D2(2:N,2:N);
%
V_sae = -(C0./r) - (Zc.*exp(-c.*r)./r) - a1.*exp(-b1.*r);
%
H0_ham = -0.5.*D2 + diag(ell*(ell+1)./(2.*r.^2)) + diag(V_sae);
%
[Vec,En] = eig(H0_ham);                                     % Eigenvalue problem
En = diag(En);
[foo, ij] = sort(En);
En = En(ij);
[En(1),En(2),En(3),En(4),En(5)];
%
%           
Vec = Vec(:,ij);                       % The unnormalized eigenfunctions
V1 = Vec(:,n_ind);                         % The unnormalized eigenfunction for the ground state,
%V1 = [0.,;V1,;0.];
n_c = sum(wr.*V1.*V1.*4.*pi);
V1 = 1./sqrt(n_c).*V1;
R = V1./r;
rho = R.*R; 
%%%
En_1 = En(1);
En_2 = En(2);
En_3 = En(3);
%%%
return
end


 function [xi,w,D]=legDC2(N,a,b)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            % legDc.m
            %
            % Computes the Legendre differentiation matrix with collocation at the
            % Legendre-Gauss-Lobatto nodes.
            %
            % Reference:
            %   C. Canuto, M. Y. Hussaini, A. Quarteroni, T. A. Tang, "Spectral Methods
            %   in Fluid Dynamics," Section 2.3. Springer-Verlag 1987
            %
            % Written by Greg von Winckel - 05/26/2004
            % Contact: gregvw@chtm.unm.edu
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            % Truncation + 1
            N1=N+1;
            
            % CGL nodes
            xc=cos(pi*(0:N)/N)';
            
            % Uniform nodes
            xu=linspace(-1,1,N1)';
            
            % Make a close first guess to reduce iterations
            if N<3
                x=xc;
            else
                x=xc+sin(pi*xu)./(4*N);
            end
            
            % The Legendre Vandermonde Matrix
            P=zeros(N1,N1);
            
            % Compute P_(N) using the recursion relation
            % Compute its first and second derivatives and
            % update x using the Newton-Raphson method.
            
            xold=2;
            while max(abs(x-xold))>eps
                
                xold=x;
                
                P(:,1)=1;    P(:,2)=x;
                
                for k=2:N
                    P(:,k+1)=( (2*k-1)*x.*P(:,k)-(k-1)*P(:,k-1) )/k;
                end
                
                x=xold-( x.*P(:,N1)-P(:,N) )./( N1*P(:,N1) );
            end
            
            X=repmat(x,1,N1);
            Xdiff=X-X'+eye(N1);
            
            L=repmat(P(:,N1),1,N1);
            L(1:(N1+1):N1*N1)=1;
            D=(L./(Xdiff.*L'));
            D(1:(N1+1):N1*N1)=0;
            D(1)=(N1*N)/4;
            D(N1*N1)=-(N1*N)/4;
            
            % Linear map from[-1,1] to [a,b]
            xi=(a*(1-x)+b*(1+x))/2;        % added by Tsogbayar Tsednee
            
            % Compute the weights
            w=(b-a)./(N*N1*P(:,N1).^2);    % added by Tsogbayar Tsednee
            
  end