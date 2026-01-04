clc;
clear;
close all;
scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = scriptDir;  % if this file sits at repo root
inputFolder  = fullfile(repoRoot, 'data', 'input');
outputFolder = fullfile(repoRoot, 'data', 'output', 'Processed_PODC');
if ~exist(outputFolder, 'dir'); mkdir(outputFolder); end
% Parameters
mu0        = -0.25;
sigma0     = 0.2;
alpha_base = 0.25;
win_size   = 15;
epsilon    = 1e-6;

lambda  = 0.6;
eta0    = 0.35;
sigma_g = 8;
if ~exist(inputFolder, 'dir')
    error('Input folder not found: %s\nCreate it and put images under data/input/.', inputFolder);
end

extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'};
fileList = [];
for k = 1:numel(extensions)
    fileList = [fileList; dir(fullfile(inputFolder, extensions{k}))]; %#ok<AGROW>
end
fileList = fileList(~[fileList.isdir]);

if isempty(fileList)
    error('No images found in: %s\nSupported: jpg/jpeg/png/bmp/tif/tiff', inputFolder);
end

fprintf('Found %d images.\n', numel(fileList));
fprintf('Input : %s\n', inputFolder);
fprintf('Output: %s\n', outputFolder);
for i = 1:numel(fileList)
    fileName = fileList(i).name;
    inPath   = fullfile(inputFolder, fileName);

    try
        imgRaw = imread(inPath);
    catch
        warning('Cannot read: %s', inPath);
        continue;
    end

    imgDouble = im2double(imgRaw);

    % Convert to grayscale if RGB
    if ndims(imgDouble) == 3
        I0 = rgb2gray(imgDouble);
    else
        I0 = imgDouble;
    end
    p_low  = prctile(I0(:), 5);
    p_high = prctile(I0(:), 98);
    I = (I0 - p_low) / (p_high - p_low + eps);
    I = min(max(I, 0), 1);
    local_mean   = imguidedfilter(I, I, 'NeighborhoodSize', [win_size win_size]);
    local_sqmean = imguidedfilter(I.^2, I, 'NeighborhoodSize', [win_size win_size]);
    local_var    = max(local_sqmean - local_mean.^2, 0);
    local_std    = sqrt(local_var + epsilon);
    I_norm = (I - local_mean) ./ (local_std + epsilon);
    I_norm = I_norm * sigma0 + mu0;

    std_mean = mean(local_std(:));
    alpha_local = alpha_base * (1 + 0.35 * (local_std ./ (std_mean + eps)));
    alpha_local = min(alpha_local, 2.5 * alpha_base);
    I_nonlin = 1 ./ (1 + exp(-alpha_local .* (I_norm - 0.5)));
    Gx = imfilter(I, [-1 0 1; -2 0 2; -1 0 1]/4, 'replicate');
    Gy = imfilter(I, ([-1 0 1; -2 0 2; -1 0 1]' )/4, 'replicate');
    grad_mag = sqrt(Gx.^2 + Gy.^2);
    g_low  = prctile(grad_mag(:), 60);
    g_high = prctile(grad_mag(:), 99);
    W = (grad_mag - g_low) / (g_high - g_low + eps);
    W = min(max(W, 0), 1);
    W = imgaussfilt(W, 1.0);

    I_struct = W .* I_nonlin + (1 - W) .* I;

    I_LF_ref = imgaussfilt(I, sigma_g);
    I_LF2    = imgaussfilt(I_struct, sigma_g);
    R = I_LF_ref - I_LF2;
    I_illum = I_struct + lambda * R;
    Delta = I_illum - I;
    s_low  = prctile(local_std(:), 20);
    s_high = prctile(local_std(:), 90);
    eta_map = (local_std - s_low) / (s_high - s_low + eps);
    eta_map = min(max(eta_map, 0), 1);
    eta = eta0 * (0.5 + 0.5 * eta_map);
    I_mix = I + eta .* Delta;
    low_p2  = prctile(I_mix(:), 0.5);
    high_p2 = prctile(I_mix(:), 99.5);
    I_final = mat2gray(I_mix, [low_p2 high_p2]);
    [~, baseName, ~] = fileparts(fileName);
    outPath = fullfile(outputFolder, [baseName, '.png']);
    try
        imwrite(I_final, outPath);
    catch
        imwrite(im2uint8(I_final), outPath);
    end

    if mod(i, 10) == 0 || i == numel(fileList)
        fprintf('Processed %d / %d\n', i, numel(fileList));
    end
end
fprintf('Done. Results saved to: %s\n', outputFolder);
