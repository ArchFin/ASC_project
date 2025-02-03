% Main folder paths
mainFolder1 = '/Users/mariakarampela/Downloads/DreemEEG'; mainFolder2 = '/Users/mariakarampela/Downloads/EEG-Group1'; % Define participant folders with yellow dots (DreemEEG) participantFolders1 = {
'1425',
'1733_BandjarmasinKomodoDragon',
'1871',
'1991_MendozaCow',
'2222_JiutaiChicken',
'2743_HuaianKoi'
};
% Define participant folders to include (EEG-Group1)
participantFolders2 = {
'184_WestYorkshireWalrus',
'1465_WashingtonQuelea',
'1867_BucharestTrout',
'1867_GoianiaCrane',
'3604_LichuanHookworm',
'3604_ShangquiHare',
'3614_BrisbaneHornet',
'3614_VientianeWhippet',
'3938_YingchengSeaLion',
'4765_NouakchottMoose',
'5644_AkesuCoral',
'5892_LvivRooster',
'5892_NonthaburiHalibut',
'7135_TampicoWallaby',
'8681_NanchangAlbatross',
'8725_SishouMosquito',
'8725_YangchunCobra'
};

% Load data from both groups
[allTETData1, participantIDs1, groups1, meditationStyles1, sessionIDs1] = loadTETData(mainFolder1, participantFolders1, 'Group1');
[allTETData2, participantIDs2, groups2, meditationStyles2, sessionIDs2] = loadTETData(mainFolder2, participantFolders2, 'Group2');

% ensure both datasets have the same number of columns
if size(allTETData1, 2) == size(allTETData2, 2)

%concatenate data from both groups
allTETData = [allTETData1; allTETData2];
participantIDs = [participantIDs1; participantIDs2];
groups = [groups1; groups2];
meditationStyles = [meditationStyles1; meditationStyles2];
sessionIDs = [sessionIDs1; sessionIDs2];

%normalize the data
allTETData = zscore(allTETData);
%set the random seed for reproducibility
rng(12345); % Ensuring the same clusters are used every time

% set the number of clusters
optimal_k = 4;
% Apply K-Means clustering
[idx, C] = kmeans(allTETData, optimal_k, 'Replicates', 1000);
% Initialize storage for averaging
numRepetitions = 2;
 allTransProbs = zeros(optimal_k, optimal_k, numRepetitions);
allEmissionMeans = zeros(optimal_k, size(allTETData, 2), numRepetitions);
allEmissionCovs = zeros(size(allTETData, 2), size(allTETData, 2), optimal_k, numRepetitions);
allFs = zeros(size(allTETData, 1), optimal_k, numRepetitions);
allStateSeqs = zeros(size(allTETData, 1), numRepetitions);
% Folder to save repetitions
saveFolder = '/Users/mariakarampela/Downloads/repetitions';
if ~exist(saveFolder, 'dir')
mkdir(saveFolder);
end

%Repeat the HMM process
for rep = 1:numRepetitions
fprintf('Repetition %d of %d\n', rep, numRepetitions);
%Set a fixed random seed before initializing random parameters
rng(12345 + rep);
%Initializing HMM parameters
numStates = optimal_k;
numEmissions = size(allTETData, 2);
%Initializing transition probabilities
transProb = rand(numStates);
transProb = transProb ./ sum(transProb, 2); % Normalize to make it stochastic
%Initializing emission probabilities (using a Student's t-distribution)
emissionMean = rand(numStates, numEmissions);
emissionCov = repmat(eye(numEmissions), [1, 1, numStates]);
nu = repmat(5, 1, numStates); % Initial guess for degrees of freedom
%Train HMM using Baum-Welch algorithm (custom implementation)
[transProb, emissionMean, emissionCov, nu] = trainHMM(allTETData(:, 1:12), numStates, emissionMean, emissionCov, transProb, nu);
%Decode the sequence using the Viterbi algorithm
[stateSeq, logProb] = decodeHMM(allTETData(:, 1:12), transProb, emissionMean, emissionCov, nu); %Calculate state probabilities using the custom forward-backward algorithm
[alpha, beta, fs, logLik] = forwardBackward(allTETData(:, 1:12), transProb, emissionMean, emissionCov, nu);
%Store results for averaging
allTransProbs(:, :, rep) = transProb;
allEmissionMeans(:, :, rep) = emissionMean;
allEmissionCovs(:, :, :, rep) = emissionCov;
allFs(:, :, rep) = fs;
allStateSeqs(:, rep) = stateSeq';
% Save the results of this repetition
saveFile = fullfile(saveFolder, sprintf('repetition_%d.mat', rep));
save(saveFile, 'transProb', 'emissionMean', 'emissionCov', 'nu', 'stateSeq', 'fs', 'logProb', 'logLik');
end
%Average the results across repetitions
avgTransProb = mean(allTransProbs, 3);
avgEmissionMeans = mean(allEmissionMeans, 3);
avgEmissionCovs = mean(allEmissionCovs, 4);
avgFs = mean(allFs, 3);
avgStateSeq = mode(allStateSeqs, 2);
%Set minimum state duration
minStateLength = 10;
%Calculate the distribution of changes in state probabilities
changes = abs(diff(avgFs, 1, 1));
meanChange = mean(changes(:));
stdChange = std(changes(:));

 %Defining an even more lenient threshold
threshold = meanChange + 0.05 * stdChange;
%Initializing labels and transition windows
labels = cell(size(allTETData, 1), 1);
transitionalWindows = cell(size(allTETData, 1), 1);
%Define the initial buffer size increment for checking purposes
initialBufferSize = 5;
% Identify transitions and label them dynamically
sessionLength = 101;
for i = 2:length(avgStateSeq)
if avgStateSeq(i) ~= avgStateSeq(i-1) && mod(i, sessionLength) ~= 1
% Check the duration of the previous state
prevState = avgStateSeq(i-1);
prevStateLength = 1;
for j = i-2:-1:1
if avgStateSeq(j) == prevState
prevStateLength = prevStateLength + 1;
else
break;
end
end
% Check the duration of the new state
newState = avgStateSeq(i);
newStateLength = 1;
for j = i+1:length(avgStateSeq)
if avgStateSeq(j) == newState
newStateLength = newStateLength + 1;
else
break;
end
end
% Skip labeling if either the previous or the new state is shorter than minStateLength
if prevStateLength < minStateLength || newStateLength < minStateLength
continue;
end
% skip transitions if the new state has a duration of 15 rows or fewer
if newStateLength <= 10
continue;
end
% check if there is a short state between two valid states and skip it
if newStateLength < minStateLength && prevStateLength >= minStateLength && newStateLength >= minStateLength
continue;
end
startIdx = i;
% move backwards to find the start of the transition
while startIdx > 1 && abs(avgFs(startIdx, newState) - avgFs(startIdx-1, newState)) > threshold
startIdx = startIdx - 1;
end
endIdx = i;
% move forward to find the end of the transition
while endIdx < length(avgStateSeq) && abs(avgFs(endIdx, newState) - avgFs(endIdx+1, newState)) > threshold
endIdx = endIdx + 1;
end

 % Initialise dynamic buffer sizes
dynamicBufferSizeStart = initialBufferSize;
dynamicBufferSizeEnd = initialBufferSize;
% dynamically adjust the buffer size for the start of the transition
while startIdx - dynamicBufferSizeStart > 0 && abs(avgFs(max(1, startIdx - dynamicBufferSizeStart), newState) - avgFs(max(1, startIdx - dynamicBufferSizeStart + 1), newState)) > threshold / 2 dynamicBufferSizeStart = dynamicBufferSizeStart + 1;
end
% Dynamically adjust the buffer size for the end of the transition
while endIdx + dynamicBufferSizeEnd <= length(avgStateSeq) && abs(avgFs(min(length(avgStateSeq), endIdx + dynamicBufferSizeEnd), newState) - avgFs(min(length(avgStateSeq), endIdx + dynamicBufferSizeEnd - 1), newState)) > threshold / 2
dynamicBufferSizeEnd = dynamicBufferSizeEnd + 1;
end
% Update the start and end indices with dynamic buffers
startIdx = max(1, startIdx - dynamicBufferSizeStart);
endIdx = min(length(avgStateSeq), endIdx + dynamicBufferSizeEnd);
%Additional check to avoid counting transitions between the same states
if avgStateSeq(startIdx) == avgStateSeq(endIdx)
continue;
end
%Label the transition
for j = startIdx:endIdx
if j <= length(labels)
labels{j} = sprintf('Transition from S%d to S%d', avgStateSeq(startIdx), avgStateSeq(endIdx)); transitionalWindows{j} = sprintf('Start: %d, End: %d', startIdx, endIdx);
end
end
end
end
% Count the number of transitions
numTransitions = sum(~cellfun('isempty', labels));
disp(['Number of transitions labeled: ', num2str(numTransitions)]);
% create results table using the averaged results
resultsTable = table(allTETData, avgStateSeq, participantIDs, labels, transitionalWindows, groups, meditationStyles, sessionIDs, ...
'VariableNames', {'TETData', 'State', 'ParticipantID', 'Label', 'TransitionalWindow', 'Group', 'MeditationStyle', 'SessionID'});
disp(resultsTable);
% display the transition matrix
disp('Averaged Transition Matrix:');
disp(avgTransProb);
% Visualize clusters and transitions using PCA for dimensionality reduction visualizeClustersAndTransitions(allTETData, avgStateSeq, labels);
% Visualize the first session (rows 1 to 101)
visualizeSession(resultsTable, 1, 101);
else
error('The number of columns in the datasets from both groups do not match.');
end
%% Function to load TET data from participant folders
function [allTETData, participantIDs, groups, meditationStyles, sessionIDs] = loadTETData(mainFolder, participantFolders, groupLabel)
allTETData = [];
participantIDs = {}; % initializing participant IDs
groups = {}; % initializing groups

 meditationStyles = []; % initializing meditation styles
sessionIDs = {}; % initializing session IDs
for i = 1:length(participantFolders)
participantFolder = fullfile(mainFolder, participantFolders{i}, '20-SubjExp'); files = dir(fullfile(participantFolder, '*_TET.mat'));
for j = 1:length(files)
fileName = fullfile(files(j).folder, files(j).name);
fprintf('Now reading %s\n', fileName);
data = load(fileName);
if isfield(data, 'Subjective')
TETData = data.Subjective(:, 1:12);
if isempty(allTETData)
allTETData = TETData;
elseif size(TETData, 2) == size(allTETData, 2)
allTETData = [allTETData; TETData];
else
fprintf('Dimension mismatch in %s\n', fileName);
end
%Add participant IDs
participantIDs = [participantIDs; repmat({participantFolders{i}}, size(TETData, 1), 1)]; %Add group label
groups = [groups; repmat({groupLabel}, size(TETData, 1), 1)];
%Add meditation styles (assuming the 13th column contains this)
meditationStyles = [meditationStyles; data.Subjective(:, 13)];
%Add session IDs (using filename as session ID)
sessionIDs = [sessionIDs; repmat({files(j).name}, size(TETData, 1), 1)];
else
fprintf('Subjective not found in %s\n', fileName);
end
end
end
%Add group label to participant IDs
participantIDs = strcat(participantIDs, '_', groupLabel);
end
%% Function to visualize clusters and transitions using PCA
function visualizeClustersAndTransitions(data, clusters, labels)
% Perform PCA for dimensionality reduction
[coeff, score, ~] = pca(data);
% Using the first two principal components for visualization
pc1 = score(:, 1);
pc2 = score(:, 2);
figure;
hold on;
colors = lines(max(clusters));
%Plotting data points with cluster colors
for i = 1:max(clusters)
clusterIdx = clusters == i;
scatter(pc1(clusterIdx), pc2(clusterIdx), 10, colors(i, :), 'filled');
end
%Plotting transition lines
for i = 2:length(labels)
if ischar(labels{i}) && contains(labels{i}, 'Transition')
plot([pc1(i-1), pc1(i)], [pc2(i-1), pc2(i)], 'k--', 'LineWidth', 1.5);
end
end

 xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('Clusters and Transitions (PCA)');
legend(arrayfun(@(x) sprintf('Cluster %d', x), 1:max(clusters), 'UniformOutput', false)); hold off;
end
%% Function to train HMM using Baum-Welch algorithm with Student's t-distribution
function [transProb, means, covariances, nu] = trainHMM(data, numStates, means, covariances, transProb, nu)
maxIter = 100;
tol = 1e-4;
numEmissions = size(data, 2);
for iter = 1:maxIter
% E-step: calculate responsibilities using current parameters
[gamma, xi] = eStep(data, numStates, transProb, means, covariances, nu);
% M-step: update parameters using responsibilities
transProb = updateTransitionProbabilities(xi);
[means, covariances, nu] = updateEmissionParameters(data, gamma, numStates);
% Check for convergence (e.g., based on log-likelihood change)
%% ...
end
end
%% Function to compute responsibilities (gamma) and pairwise probabilities (xi) in the E-step
function [gamma, xi] = eStep(data, numStates, transProb, means, covariances, nu)
numData = size(data, 1);
gamma = zeros(numData, numStates);
xi = zeros(numStates, numStates, numData - 1);
alpha = zeros(numData, numStates);
for t = 1:numData
for j = 1:numStates
if t == 1
alpha(t, j) = mvtpdf(data(t, :), means(j, :), covariances(:, :, j), nu(j)) * (1/numStates);
else
alpha(t, j) = mvtpdf(data(t, :), means(j, :), covariances(:, :, j), nu(j)) * sum(alpha(t-1, :) .* transProb(:, j)'); end
end
alpha(t, :) = alpha(t, :) / sum(alpha(t, :));
end
beta = zeros(numData, numStates);
beta(numData, :) = 1;
for t = numData-1:-1:1
for i = 1:numStates
beta(t, i) = sum(beta(t+1, :) .* transProb(i, :) .* arrayfun(@(j) mvtpdf(data(t+1, :), means(j, :), covariances(:, :, j), nu(j)), 1:numStates));
end
beta(t, :) = beta(t, :) / sum(beta(t, :));
end
for t = 1:numData
gamma(t, :) = alpha(t, :) .* beta(t, :);
gamma(t, :) = gamma(t, :) / sum(gamma(t, :));
end
for t = 1:numData-1
for i = 1:numStates
for j = 1:numStates
xi(i, j, t) = alpha(t, i) * transProb(i, j) * mvtpdf(data(t+1, :), means(j, :), covariances(:, :, j), nu(j)) * beta(t+1, j);

 end
end
xi(:, :, t) = xi(:, :, t) / sum(sum(xi(:, :, t)));
end
end
%% Function to update transition probabilities in the M-step
function transProb = updateTransitionProbabilities(xi)
transProb = sum(xi, 3);
transProb = transProb ./ sum(transProb, 2);
end
%% Function to update means, covariances, and degrees of freedom for Student's t-distribution in the M-step
function [means, covariances, nu] = updateEmissionParameters(data, gamma, numStates)
[numData, numEmissions] = size(data);
means = zeros(numStates, numEmissions);
covariances = zeros(numEmissions, numEmissions, numStates);
nu = zeros(1, numStates);
for j = 1:numStates
gamma_j = gamma(:, j);
sum_gamma_j = sum(gamma_j);
means(j, :) = sum(data .* gamma_j) / sum_gamma_j;
S = zeros(numEmissions, numEmissions);
for t = 1:numData
S = S + gamma_j(t) * (data(t, :) - means(j, :))' * (data(t, :) - means(j, :));
end
covariances(:, :, j) = S / sum_gamma_j;
covariances(:, :, j) = makePositiveDefinite(covariances(:, :, j));
nu(j) = estimateNu(gamma_j, data, means(j, :), covariances(:, :, j));
end
end
%% Function to compute multivariate t-distribution probability density function
function p = mvtpdf(x, mu, Sigma, nu)
d = length(mu);
x_mu = x(:) - mu(:);
p = gamma((nu + d) / 2) / (gamma(nu / 2) * (nu * pi)^(d / 2) * sqrt(det(Sigma)) * (1 + (x_mu' / Sigma) * x_mu / nu)^((nu + d) / 2));
end
%% Function to estimate degrees of freedom for Student's t-distribution
function nu = estimateNu(gamma, data, mean, covariance)
nu = 5;
tol = 1e-3;
maxIter = 100;
for iter = 1:maxIter
oldNu = nu;
% Update nu based on some iterative method, e.g., Newton-Raphson
% For simplicity, we'll keep it constant here
if abs(nu - oldNu) < tol
break;
end
end
end
%% Function to decode the sequence using the trained HMM
function [stateSeq, logProb] = decodeHMM(data, transProb, means, covariances, nu)
numData = size(data, 1);
numStates = size(transProb, 1);

 delta = zeros(numData, numStates); psi = zeros(numData, numStates); for j = 1:numStates
if numel(covariances(:, :, j)) > 2 covMat = covariances(:, :, j);
else
covMat = covariances(:, j);
end
delta(1, j) = mvtpdf(data(1, :), means(j, :), covMat, nu(j)) * (1/numStates);
end
delta(1, :) = delta(1, :) / sum(delta(1, :));
for t = 2:numData
for j = 1:numStates
[delta(t, j), psi(t, j)] = max(delta(t-1, :) .* transProb(:, j)');
delta(t, j) = delta(t, j) * mvtpdf(data(t, :), means(j, :), covMat, nu(j)); end
delta(t, :) = delta(t, :) / sum(delta(t, :));
end
stateSeq = zeros(1, numData);
[~, stateSeq(numData)] = max(delta(numData, :));
for t = numData-1:-1:1
stateSeq(t) = psi(t+1, stateSeq(t+1));
end
logProb = sum(log(max(delta, [], 2)));
end
%% Function to compute forward-backward probabilities and state probabilities
function [alpha, beta, gamma, logLik] = forwardBackward(data, transProb, means, covariances, nu) numData = size(data, 1);
numStates = size(transProb, 1);
alpha = zeros(numData, numStates);
beta = zeros(numData, numStates);
gamma = zeros(numData, numStates);
logLik = 0;
for t = 1:numData
for j = 1:numStates
if numel(covariances(:, :, j)) > 2
covMat = covariances(:, :, j);
else
covMat = covariances(:, j);
end
if t == 1
alpha(t, j) = mvtpdf(data(t, :), means(j, :), covMat, nu(j)) * (1/numStates);
else
alpha(t, j) = mvtpdf(data(t, :), means(j, :), covMat, nu(j)) * sum(alpha(t-1, :) .* transProb(:, j)');
end
end
alpha(t, :) = alpha(t, :) / sum(alpha(t, :));
end
beta(numData, :) = 1;
for t = numData-1:-1:1
for i = 1:numStates
beta(t, i) = sum(beta(t+1, :) .* transProb(i, :) .* arrayfun(@(j) mvtpdf(data(t+1, :), means(j, :), covMat, nu(j)), 1:numStates));
end
beta(t, :) = beta(t, :) / sum(beta(t, :));

end
for t = 1:numData
gamma(t, :) = alpha(t, :) .* beta(t, :);
gamma(t, :) = gamma(t, :) / sum(gamma(t, :));
end
logLik = sum(log(sum(alpha, 2)));
end
%Function to ensure covariance matrices are positive definite
function covMatrix = makePositiveDefinite(covMatrix)
[V, D] = eig(covMatrix);
D(D < 1e-6) = 1e-6;
covMatrix = V * D / V;
end
function visualizeSession(resultsTable, startRow, endRow)
%Extract the relevant segment from the results table
segment = resultsTable(startRow:endRow, :);
%Define the state colors
stateColors = lines(max(segment.State));
%Plot the data for each dimension
figure;
hold on;
numDimensions = size(segment.TETData, 2);
offset = 5;
for i = 1:numDimensions
plot(segment.TETData(:, i) + offset * (i - 1));
end
%Add shaded areas for each state
for i = 1:max(segment.State)
idx = segment.State == i;
if any(idx)
area(find(idx), repmat(max(max(segment.TETData)) + 2 * numDimensions, 1, sum(idx)), ... 'FaceColor', stateColors(i, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end
end
%Add transition windows
for i = 1:length(segment.TransitionalWindow)
if ~isempty(segment.TransitionalWindow{i})
transWindow = sscanf(segment.TransitionalWindow{i}, 'Start: %d, End: %d');
x = transWindow(1):transWindow(2);
if ~isempty(x) && all(x <= endRow)
plot(x, repmat(max(max(segment.TETData)) + 2 * numDimensions, size(x)), 'k-', 'LineWidth', 2); end
end
end
title(sprintf('EEG Data Visualization from Row %d to %d', startRow, endRow));
xlabel('Time');
ylabel('Amplitude (offset for each dimension)');
hold off;
end
