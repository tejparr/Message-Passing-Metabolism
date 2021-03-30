function DEMO_MessagePassingMetabolism
% This demo reproduces the figures presented in the paper 'Message passing
% and metabolism,' written for the Entropy special issue 'Applying the 
% Free-Energy Principle to Complex Adaptive Systems.' It includes a series
% of demonstrations, first illustrating an interpretation of a randomly
% generated master equation as a gradient descent on a free energy
% functional, then moves to setting out a set of conditions that, when
% imposed on the master equation, lead to mass action kinetics of the sort
% encountered in biochemical metabolic networks. From here, we can move to
% Michaelis-Menten kinetics, which summarise enzymatic kinetics, and use
% these to express a simple biochemical network and the consequences of
% inducing enzymatic disconnection.
%__________________________________________________________________________

% Preliminaries
%==========================================================================
rng default % for reproducibility
close all
warning('off','MATLAB:delaunay:DupPtsDelaunayWarnId')
warning('off','MATLAB:plot:IgnoreImaginaryXYZPart')

% Solenoidal and dissipative decomposition of a master equation
%==========================================================================
% Randomly generate probability transition matrix
%--------------------------------------------------------------------------
L = abs(randn(3,3));            % Randomly generate matrix with positive values
L = L - diag(diag(L));          % Set diagonal elements to zero
L = L - diag(sum(L));           % Ensure conservation of probability

% Find steady state
%--------------------------------------------------------------------------
[~, S, V] = svd(L);             % Singular value decomposition
k = find(abs(diag(S))<1e-8);    % Find zero singular value (up to 1e-8 tolerance)
K = V(:,k)/sum(V(:,k));         % Steady state distribution

% Decompose L in terms of steady state
%--------------------------------------------------------------------------
A = L*diag(K);                  % Re-express so rows sum to zero
R = - 0.5*(A + A');             % Dissipative term
Q = A + R;                      % Solenoidal term

% Simulate trajectories
%--------------------------------------------------------------------------
for i = 1:20
   % Initial conditions
   %-----------------------------------------------------------------------
   r = rand(3,1);
   r = r/sum(r);
   p_L{i} = r;
   p_Q{i} = r;
   p_R{i} = r;
   
   % Free energies
   %-----------------------------------------------------------------------
   F_L{i} = zeros(1,100); F_L{i}(1) = mpm_free_energy(K,p_L{i});
   F_Q{i} = zeros(1,100); F_Q{i}(1) = mpm_free_energy(K,p_Q{i});
   F_R{i} = zeros(1,100); F_R{i}(1) = mpm_free_energy(K,p_R{i});
end

for i = 1:100
    for j = 1:length(p_L)
        p_L{j}(:,end+1) = expm( L/16)*p_L{j}(:,end);
        p_Q{j}(:,end+1) = expm( Q/diag(K)/16)*p_Q{j}(:,end);
        p_R{j}(:,end+1) = expm(-R/diag(K)/16)*p_R{j}(:,end);
        F_L{j}(i+1)     = mpm_free_energy(K,p_L{j}(:,i+1));
        F_Q{j}(i+1)     = mpm_free_energy(K,p_Q{j}(:,i+1));
        F_R{j}(i+1)     = mpm_free_energy(K,p_R{j}(:,i+1));
    end
end

% Free energy landscape
%--------------------------------------------------------------------------
[x,y] = meshgrid(0:1/128:1,0:1/128:1);
x     = fliplr(triu(fliplr(x)));
y     = fliplr(triu(fliplr(y)));

tri   = delaunay(x,y);
G     = zeros(size(x));
for i = 1:size(x,1)
    for j = 1:size(x,2)
        G(i,j) = mpm_free_energy(K,[x(i,j),y(i,j),1-x(i,j)-y(i,j)]');
    end
end

% Plot free energy landscape
%--------------------------------------------------------------------------
figure('Color','w','Name','Solenoidal and dissipative flows');
subplot(1,3,1)
trisurf(tri,x,y,G,'EdgeColor','none'), colormap bone, axis square, hold on
for i = 1:length(p_L)
    plot3(p_L{i}(1,:),p_L{i}(2,:),F_L{i}+exp(-4),'w','LineWidth',1.5)
end
ax = gca; ax.FontName = 'Times New Roman';
title('Dissipitive + Solenoidal')

subplot(1,3,2)
trisurf(tri,x,y,G,'EdgeColor','none'), colormap bone, axis square, hold on
for i = 1:length(p_L)
    plot3(p_Q{i}(1,:),p_Q{i}(2,:),F_Q{i}+exp(-4),'w','LineWidth',1.5)
end
ax = gca; ax.FontName = 'Times New Roman';
title('Solenoidal')

subplot(1,3,3)
trisurf(tri,x,y,G,'EdgeColor','none'), colormap bone, axis square, hold on
for i = 1:length(p_L)
    plot3(p_R{i}(1,:),p_R{i}(2,:),F_R{i}+exp(-4),'w','LineWidth',1.5)
end
ax = gca; ax.FontName = 'Times New Roman';
title('Dissipitive')

clear

% Construct chemical reaction
%==========================================================================
% Here we specify a generative model for a reaction:
%      x(1) + x(2) <-> x(3) + x(4)

alpha = 1/4;

% Prior for first molecule - P(x(1))
%--------------------------------------------------------------------------
D{1} = [alpha;1-alpha];

% Empirical (conditional) prior for second molecule - P(x(2)|x(1))
%--------------------------------------------------------------------------
D{2} = eye(2);      % As on the same side of the reaction, if molecule 1 is present, so is molecule 2

% Empirical (conditional) prior for third molecule - P(x(3)|x(2))
%--------------------------------------------------------------------------
D{3} = [0 1;1 0];   % As on the opposite side of the reaction, if molecule 2 is present, molecule 3 is not

% Empirical (conditional) prior for fourth molecule - P(x(4)|x(3))
%--------------------------------------------------------------------------
D{4} = eye(2);      % As on the same side of the reaction, if molecule 3 is present, so is molecule 4

% Construct steady state distribution
%--------------------------------------------------------------------------
for f1 = 1:2
    for f2 = 1:2
        for f3 = 1:2
            for f4 = 1:2
                K(f1,f2,f3,f4) = D{1}(f1)*D{2}(f2,f1)*D{3}(f3,f2)*D{4}(f4,f3);
            end
        end
    end
end

% Construct master equation
%--------------------------------------------------------------------------
b    = alpha/(1 - alpha);                              % This can be obtained by solving the master equation when the marginals are consistent with steady state
b    = b^2;                                            % The ^2 is a correction for the mean-field approximation
V    = b*kron([1 1],kron([1 1],kron([1 0],[1 0])))...  % Right singular vector (orthogonal to steady state)
       - kron([1 0],kron([1 0],kron([1 1],[1 1])));
a    = [1 -1]';                                        % Left singular vector (ensures conservation of probability)

% Transition rate matrix for each species
L{1} =  a*V;
L{2} =  a*V;
L{3} = -a*V;
L{4} = -a*V;

% Initial conditions
p{1} = [1/2;1/2];
p{2} = [1/2;1/2];
p{3} = [1/2;1/2];
p{4} = [1/2;1/2];

G(1) = mpm_free_energy(K(:),mpm_kroni(p,1));

% Simulate system through numerical integration
%--------------------------------------------------------------------------
for i = 1:64
    P = kron(p{1}(:,i),p{2}(:,i));
    P = kron(P,p{3}(:,i));
    P = kron(P,p{4}(:,i));
    for j = 1:length(p)
        p{j}(:,i+1) = p{j}(:,i) + L{j}*P/16;
    end
    G(i) = mpm_free_energy(K(:),mpm_kroni(p,i));
end

% Plot reaction kinetics
%--------------------------------------------------------------------------
figure('Color','w','Name','Kinetics of simple reaction'); clf
colormap gray
subplot(2,2,1)
plot(p{1}(1,:)), hold on, plot(p{2}(1,:)), plot(p{3}(1,:)), plot(p{4}(1,:));
title('Normalised concentrations')
xlim([1,size(p{1},2)])
ax = gca; ax.FontName = 'Times New Roman';
subplot(2,2,2)
imagesc(1-[p{1};p{2};p{3};p{4}]), caxis([0 1]), title('Probabilities')
ax = gca; ax.FontName = 'Times New Roman';
subplot(2,2,3)
plot(G-G(1)), caxis([0 1]), title('Free energy (change)')
xlim([1,size(p{1},2)])
ax = gca; ax.FontName = 'Times New Roman';

clear 

% Construct chemical reaction system
%==========================================================================
% Here we specify a generative model for a reaction system:
%      x(1) + x(2) <-> x(3) 
%             x(3) <-> x(4) + x(5)


alpha(1) = 0.5;
alpha(2) = 0.7;

% Prior for first and second molecules - P(x(1),x(2))
%--------------------------------------------------------------------------
D{1} = [alpha(1);0;0;1-alpha(1)];

% Empirical (conditional) prior for third molecule - P(x(3)|x(1),x(2))
%--------------------------------------------------------------------------
D{2} = [0 0 0 (alpha(2)-alpha(1))/(alpha(2)*(1-alpha(1)));
        1 0 0 (alpha(1)-alpha(1)*alpha(2))/(alpha(2)*(1-alpha(1)))];

% Empirical (conditional) prior for fourth and fifth molecules - P(x(4),x(5)|x(3))
%--------------------------------------------------------------------------
D{3} = [0 1-alpha(2); 0 0; 0 0; 1 alpha(2)];
   
% Construct steady state distribution
%--------------------------------------------------------------------------
for f1 = 1:2
    for f2 = 1:2
        for f3 = 1:2
            K(f1,f2,f3) = D{1}(f1)*D{2}(f2,f1)*D{3}(f3,f2);
        end
    end
end

% Construct master equation
%--------------------------------------------------------------------------
z    = 0.5; % Free parameter

% Define right singular vectors as weighted sum of the following vectors
u{1} = [kron([1;0],kron([1;1],[1;1])) kron([1;1],kron([1;0],[1;0]))];
u{2} = [kron([1;0],kron([1;0],kron([1;1],kron([1;1],[1;1]))))...
        kron([1;1],kron([1;1],kron([1;0],kron([1;1],[1;1]))))...
        kron([1;1],kron([1;1],kron([1;1],kron([1;0],[1;0]))))];
u{3} = [kron([1;0],kron([1;0],[1;1])) kron([1;1],kron([1;1],[1;0]))];

% Coefficients for above (obtained through solving master equation at
% steady state).
b1   = z*(alpha(2) - alpha(1))/(alpha(2)*alpha(1)^2);
b2   = (1-z)*alpha(2)*(alpha(2) - alpha(1))/(alpha(1)*(1-alpha(2)))^2;
b3   = z; 

% Right singular vectors
V{1} = (u{1}*[b3;-b1])';
V{2} = (u{2}*[b2;-1;b1])';
V{3} = (u{3}*[-b2;1-b3])';

% Left singular vector
a = [1;-1];

% Transition rate matrices for each species
L{1} = a*V{1};
L{2} = a*V{1};
L{3} = a*V{2};
L{4} = a*V{3};
L{5} = a*V{3};

% Initial conditions
p{1} = [1/3;2/3];
p{2} = [1/3;2/3];
p{3} = [1/3;2/3];
p{4} = [1/3;2/3];
p{5} = [1/3;2/3];

% Numerically integrate to simulate kinetics
%--------------------------------------------------------------------------
for i = 1:64
    for j = 1:length(p)
        if j == 1 || j == 2
            P = kron(p{2}(:,i),p{1}(:,i));
            P = kron(p{3}(:,i),P);
        elseif j == 3
            P = kron(p{2}(:,i),p{1}(:,i));
            P = kron(p{3}(:,i),P);
            P = kron(p{4}(:,i),P);
            P = kron(p{5}(:,i),P);
        else
            P = kron(p{4}(:,i),p{3}(:,i));
            P = kron(p{5}(:,i),P);
        end
        p{j}(:,i+1) = p{j}(:,i) + L{j}*P/16;
    end
end

% Plot reaction kinetics
%--------------------------------------------------------------------------
figure('Color','w','Name','Kinetics of simple reaction system'); clf
colormap gray
subplot(3,1,1)
plot(p{1}(1,:)), hold on, plot(p{2}(1,:)), plot(p{3}(1,:)), plot(p{4}(1,:)), plot(p{5}(1,:));
xlim([1,size(p{1},2)])
title('Normalised concentrations')
ax = gca; ax.FontName = 'Times New Roman';
subplot(3,1,2)
imagesc(1-[p{1};p{2};p{3};p{4};p{5}]), caxis([0 1]), title('Probabilities')
ax = gca; ax.FontName = 'Times New Roman';

clear

% Construct enzymatic system
%==========================================================================
% Here we specify a generative model for a reaction:
%      x(1) + x(3) <-> x(2) <-> x(4) + x(1)
% where x(1) is an enzyme and x(2) is an enzyme-substrate complex

alpha(1) = 0.6;
alpha(2) = 0.1;

% Prior for enzyme versus complex - P(x(1),x(2))
%--------------------------------------------------------------------------
D{1} = [0;alpha(1);1-alpha(1);0];

% Empirical (conditional) prior for substrate molecule - P(x(3)|x(1))
%--------------------------------------------------------------------------
D{2} = [0 alpha(2)   0 alpha(2);
        1 1-alpha(2) 1 1-alpha(2)];

% Empirical (conditional) prior for product molecules - P(x(4)|x(1))
%--------------------------------------------------------------------------
D{3} = [0 1-alpha(2) 0 1-alpha(2);
        1 alpha(2)   1 alpha(2)];
    
% Construct steady state distribution
%--------------------------------------------------------------------------
for f1 = 1:4
    for f2 = 1:2
        for f3 = 1:2
            K(f1,f2,f3) = D{1}(f1)*D{2}(f2,f1)*D{3}(f3,f2);
        end
    end
end

% Construct master equation
%--------------------------------------------------------------------------
z    = 0.5;                     % free parameter
c    = (1 - z)*(1-alpha(1));    % supply rate for S and loss rate of P
L    = mpm_enzyme_L(alpha,z,c); % transition rate matrices

% Initial conditions
p{1} = [1/2;1/2];
p{2} = [1/2;1/2];
p{3} = [1/4;3/4];
p{4} = [1/4;3/4];

% Numerically integrate to simulate kinetics
%--------------------------------------------------------------------------
for i = 1:128
    for j = 1:length(p)
        if j == 3
            P = kron(p{2}(:,i),p{1}(:,i));
            P = kron(p{3}(:,i),P);
        elseif j == 1 || j == 2
            P = kron(p{2}(:,i),p{1}(:,i));
            P = kron(p{3}(:,i),P);
            P = kron(p{4}(:,i),P);
        else
            P = kron(p{2}(:,i),p{1}(:,i));
            P = kron(p{4}(:,i),P);
        end
        p{j}(:,i+1) = p{j}(:,i) + L{j}*P/16;
    end
end

% Plot kinetics
%--------------------------------------------------------------------------
figure('Color','w','Name','Enzyme (Michaelis Menten) kinetics'); clf
colormap gray
subplot(2,2,1)
title('Normalised concentrations')
plot(p{1}(1,:)), hold on, plot(p{2}(1,:)), plot(p{3}(1,:)), plot(p{4}(1,:));
xlim([1,size(p{1},2)])
ax = gca; ax.FontName = 'Times New Roman';
subplot(2,2,3)
imagesc(1-[p{1};p{2};p{3};p{4}]), caxis([0 1]), title('Probabilities')
ax = gca; ax.FontName = 'Times New Roman';

% Illustrate inferential perspective
%==========================================================================
% Under the steady state distribution, the relationship between the
% marginals for the substrate and product is linear. 

q{3} = alpha(1) - p{4}(1,:); % Beliefs about substrate given by product
q{4} = alpha(1) - p{3}(1,:); % Beliefs about product given by substrate

subplot(2,2,2)
plot(q{3}), hold on, plot(q{4})
xlim([1,size(p{1},2)])
ax = gca; ax.FontName = 'Times New Roman';

% Calculate free energies expected under enzyme distribution
%--------------------------------------------------------------------------
K1 = sum(K,3);          % Marginalise out product
K2 = squeeze(sum(K,2)); % Marginalise out substrate

for i = 1:length(q{3})
    F1(1,i) = mpm_free_energy(K1(2,:)',[q{3}(i);1-q{3}(i)]);
    F1(2,i) = mpm_free_energy(K1(3,:)',[q{3}(i);1-q{3}(i)]);
    F2(1,i) = mpm_free_energy(K2(2,:)',[q{4}(i);1-q{4}(i)]);
    F2(2,i) = mpm_free_energy(K2(3,:)',[q{4}(i);1-q{4}(i)]);
end

% Average under enzyme distribution
F1 = sum(p{2}.*F1,1);
F2 = sum(p{2}.*F2,1);

% Plot free energy changes
%--------------------------------------------------------------------------
subplot(2,2,4)
plot(F1'), hold on
plot(F2')
xlim([1,size(p{1},2)])
title('(Average) free energy')
ax = gca; ax.FontName = 'Times New Roman';

clear

% Construct metabolic network 
%==========================================================================
% This part of the demo illustrates the notion of a disconnection and a
% diaschisis - drawing from canonical network pathologies in neurobiology. 

% Stoichiometry matrix
%--------------------------------------------------------------------------
W = zeros(7,12);
W(1,3)  = 1;  W(1,4)  = -1; W(1,5)  = -1; W(1,6) = 1; W(1,9) = -1; W(1,10) = 1;
W(2,2)  = 1;  W(2,3)  = -1; W(2,4)  =  1;
W(3,1)  = 1;  W(3,2)  = -1;
W(4,5)  = 1;  W(4,6)  = -1; W(4,7)  = -1;
W(5,7)  = 1;  W(5,8)  = -1;
W(6,9)  = 1;  W(6,10) = -1; W(6,11) = -1;
W(7,11) = 1;  W(7,12) = -1;

c       = 1/8;

% Specify marginals directly
%--------------------------------------------------------------------------
alpha   = rand(7,1);
alpha   = alpha/sum(alpha);
km(1)   = 0;
km(2:size(W,2)) = rand(1,size(W,2)-1);

% Ensure that at steady state, mass is conserved
%--------------------------------------------------------------------------
vmax(1)         = c;
r(1)            = c;
vmax(2)         = c*(km(2)+alpha(3))/alpha(3);
r(2)            = mpm_Michaelis_Menten(vmax(2),km(2),alpha(3));
vmax(3)         = vmax(2) + 0.4;
r(3)            = mpm_Michaelis_Menten(vmax(3),km(3),alpha(2));
vmax(4)         = (r(3) - r(2))*(km(4) + alpha(1))/alpha(1);
r(4)            = mpm_Michaelis_Menten(vmax(4),km(4),alpha(1));
vmax(6)         = 0.1;
r(6)            = mpm_Michaelis_Menten(vmax(6),km(6),alpha(4));
vmax(10)        = 0.2;
r(10)           = mpm_Michaelis_Menten(vmax(10),km(10),alpha(6));
vmax(5)         = 0.3*(r(3)-r(4)+r(6)+r(10))*(km(5) + alpha(1))/alpha(1);
r(5)            = mpm_Michaelis_Menten(vmax(5),km(5),alpha(1));
vmax(9)         = (r(3)-r(4)+r(6)+r(10)-r(5))*(km(9) + alpha(1))/alpha(1);
r(9)            = mpm_Michaelis_Menten(vmax(9),km(9),alpha(1));
vmax(7)         = (r(5)-r(6))*(km(7) + alpha(4))/alpha(4);
r(7)            = mpm_Michaelis_Menten(vmax(7),km(7),alpha(4));
vmax(8)         = r(7)*(km(8) + alpha(5))/alpha(5);
r(8)            = mpm_Michaelis_Menten(vmax(8),km(8),alpha(5));
vmax(11)        = (r(9)-r(10))*(km(11) + alpha(6))/alpha(6);
r(11)           = mpm_Michaelis_Menten(vmax(11),km(11),alpha(6));
vmax(12)        = r(11)*(km(12) + alpha(7))/alpha(7);
r(12)           = mpm_Michaelis_Menten(vmax(12),km(12),alpha(7));

[ind,~] = find(W<0);

% Reactions vector function
%--------------------------------------------------------------------------
r = @(u) [c;mpm_Michaelis_Menten(vmax(2:end),km(2:end),u(ind,1))]; % Reactions (assumed to obey Michaelis-Menten kinetics)

% Solve reaction system
%--------------------------------------------------------------------------
u = exp(-16)*ones(size(W,1),1);

for i = 1:512
    u(:,i+1) = u(:,i) + W*r(u(:,i))/16;
end

% Plot 'healthy' steady state
%--------------------------------------------------------------------------
figure('Color','w','Name','Reaction network'); clf
subplot(3,2,1)
plot(u');
xlim([1,size(u,2)])
ax = gca; ax.FontName = 'Times New Roman';
subplot(3,2,[3 5])

for i = 1:size(u,2)
    plot([0,0],[-3 3],'k'), hold on
    plot([0,3],[0 0],'k')
    plot(0,0,'.b','MarkerSize',256*u(1,i))
    plot(0,1,'.b','MarkerSize',256*u(2,i))
    plot(0,2,'.b','MarkerSize',256*u(3,i))
    plot(0,-1,'.b','MarkerSize',256*u(6,i))
    plot(0,-2,'.b','MarkerSize',256*u(7,i))
    plot(1,0,'.b','MarkerSize',256*u(4,i))
    plot(2,0,'.b','MarkerSize',256*u(5,i))
    hold off
    axis equal, axis off
    drawnow
end

vmax(5) = 0; % Induce disconnection

% Reactions vector function
%--------------------------------------------------------------------------
r = @(u) [c;mpm_Michaelis_Menten(vmax(2:end),km(2:end),u(ind,1))]; % Reactions (assumed to obey Michaelis-Menten kinetics)

% Solve reaction system
%--------------------------------------------------------------------------
u = exp(-16)*ones(size(W,1),1);

for i = 1:512
    u(:,i+1) = u(:,i) + W*r(u(:,i))/8;
end

% Plot lesioned steady state
%--------------------------------------------------------------------------
subplot(3,2,2)
plot(u');
xlim([1,size(u,2)])
ax = gca; ax.FontName = 'Times New Roman';
subplot(3,2,[4 6])

for i = 1:size(u,2)
    plot([0,0],[-3 3],'k'), hold on
    plot([0,3],[0 0],'k')
    plot(0,0,'.b','MarkerSize',256*u(1,i))
    plot(0,1,'.b','MarkerSize',256*u(2,i))
    plot(0,2,'.b','MarkerSize',256*u(3,i))
    plot(0,-1,'.b','MarkerSize',256*u(6,i))
    plot(0,-2,'.b','MarkerSize',256*u(7,i))
    plot(1,0,'.b','MarkerSize',256*u(4,i))
    plot(2,0,'.b','MarkerSize',256*u(5,i))
    hold off
    axis equal, axis off
    drawnow
end


function P = mpm_kron(p)
% Kronecker Tensor product of all matrices in array p.
if numel(p)>2
    p{2} = kron(p{2},p{1});
    q    = p(2:end);
    P    = mpm_kron(q);
else
    P = kron(p{2},p{1});
end

function Pi = mpm_kroni(p,i)
% Kronecker Tensor product of ith column vectors of each matrix in p.
qi = cell(size(p));
for j = 1:numel(p)
    qi{j} = p{j}(:,i);
end
Pi = mpm_kron(qi);

function F = mpm_free_energy(K,p)
% Variational free energy
F = p'*(log(p+1e-8) - log(K+1e-8));

function r = mpm_Michaelis_Menten(vmax,km,u)
% Rate function under Michaelis Menten kinetics
r = vmax(:).*(u./(km(:) + u(:)));

function L = mpm_enzyme_L(alpha,z,c)
% Probability rate matrix for enzymatic system

% Left singular vector
%--------------------------------------------------------------------------
a = [1;-1];

% Combinations of states
%--------------------------------------------------------------------------
u{1} = [kron([1;1],kron([1;1],[1;0])), kron([1;0],kron([1;0],[1;1]))]; % Complex, enzyme x substrate [omit product]
u{2} = [kron(kron([1;1],kron([1;1],[1;1])),[1;0]),...                  % Complex
        kron(kron([1;1],kron([1;0],[1;0])),[1;1]),...                  % Enzyme x substrate
        kron(kron([1;0],kron([1;1],[1;0])),[1;1])];                    % Enzyme x product
u{3} = [kron([1;1],kron([1;1],[1;0])) kron([1;0],kron([1;0],[1;1]))];  % Complex, enzyme x product [omit substrate]

% Coefficients
%--------------------------------------------------------------------------
b1   = (z + c - alpha(1)*z)/(alpha(1)^2*alpha(2));
b2   = (alpha(1) + z + c - alpha(1)*z - 1)/(alpha(1)^2*(alpha(2) - 1));
b3   = z; 

% Right singular vectors
%--------------------------------------------------------------------------
V{1} = (u{1}*[b3;-b1])'   + c;   % For substrate
V{2} = (u{2}*[-1;b1;b2])';       % For complex/enzyme
V{3} = (u{3}*[1-b3;-b2])' - c;   % For product

% Transition rate matrices
%--------------------------------------------------------------------------
L{1} =  a*V{2}; % C
L{2} = -a*V{2}; % E
L{3} =  a*V{1}; % S
L{4} =  a*V{3}; % P


