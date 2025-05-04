clc
clear
%IMG-20250414-WA0177.jpg - 2 8 inverted background issue
%carplate.jpg - 2 8
%ev.jpg - 2 8 - very close (1 character wrong); 1 5 - can, cnanot
%IMG-20250414-WA0204.jpg - 2 8 - inverted issue
%mercedes.jpg - 2 8
%car2.jpg - 2 8 detect plate but not ocr
%sms.jpg - 2 8
%KJ8-HondaCity - 2 8 detect len is 2 (filtered out)
%UKM - 2 8 cannot
%car3 - 2 8 cnanot
%vgr - 2 8
%black - 2 8 close, first char wrong
%big - 2 8 ocr got problem
%test2 - 2 6 ocr lighting on image
%test - 2 8
%337564373_1159416691422375_9123487902255978681_n - 2 8
%337166832_5904455639676740_285992518671105305_n - 2 8 
%grey - 2 8
%IMG_20250419_1323352 - 2 8
%IMG_20250424_0812332 - 2 8
%IMG_20250310_1049013 - 2 8
%126061387_4167529469940194_3888307349537178725_n2 - 2 8
%156608911_3726984024022394_7157110332085937238_n - 2 8 d looks like 0
%ZC3353 - 2 8 Z looks like 4

% Read the input image
originalImage = imread('Bus_5.HEIC');

% Convert to grayscale
grayImage = rgb2gray(originalImage);

% Apply preprocessing techniques
% 1. Contrast enhancement
enhancedImage = imadjust(grayImage); % Boost contrast
enhancedImage = adapthisteq(enhancedImage);

% 2. Noise reduction
denoisedImage = medfilt2(enhancedImage);

% 3. Edge detection (Sobel operator)
edgeImage = edge(denoisedImage, 'sobel');

% 4. Morphological operations for plate region detection
dilatedEdges = imdilate(edgeImage, strel('rectangle', [2 8]));
filledRegions = imfill(dilatedEdges, 'holes');

% 5. Candidate region selection
regionProps = regionprops(filledRegions, 'BoundingBox', 'Area', 'Extent');
plateRegions = [];
imageArea = size(originalImage, 1) * size(originalImage, 2);
rectangleImage = originalImage;

% Define a list of colors for isClose boxes
colorList = {'blue', 'yellow', 'cyan', 'magenta', 'white', 'black', 'red'};
colorCount = length(colorList);
colorIndex = 1;
disp(imageArea * 0.001)
disp(imageArea * 0.1)
for i = 1:length(regionProps)
    bbox = regionProps(i).BoundingBox;
    aspectRatio = bbox(3) / bbox(4);
    area = regionProps(i).Area;

    % Draw rectangle for all detected regions
    %rectangleImage = insertShape(rectangleImage, 'Rectangle', bbox, 'Color', 'red', 'LineWidth', 1, LineWidth=5);
    
    % Filter conditions for potential plate regions
    if (aspectRatio >= 3 && aspectRatio <= 4.5) && (area > 0.001 * imageArea && area < 0.1 * imageArea)
        plateRegions = [plateRegions, i];
        rectangleImage = insertShape(rectangleImage, 'Rectangle', regionProps(i).BoundingBox, 'Color', 'green', 'LineWidth', 2, LineWidth=5);
    else
        isClose = (aspectRatio >= 2.5 && aspectRatio <= 5) && ...
                  (area > 0.0005 * imageArea && area < 0.15 * imageArea);

       if isClose
            % Pick a color from the list and cycle
            currentColor = colorList{colorIndex};
            rectangleImage = insertShape(rectangleImage, 'Rectangle', bbox, ...
                'Color', currentColor, 'LineWidth', 2);
            if strcmp(currentColor, 'magenta')
                disp(currentColor)
                disp(aspectRatio);
                disp(area);
            end
            % Cycle color index
            colorIndex = mod(colorIndex, colorCount) + 1;
        end
    end
end

% 6. Plate extraction and manual OCR integration
validCandidates = {};
plateBoundingBoxes = {};
plateIntensities = {};

for idx = 1:length(plateRegions)
    i = plateRegions(idx);
    bbox = regionProps(i).BoundingBox;
    padding = 2;
    % Add padding: expand width and height, shift x and y
    paddedBBox = [
        bbox(1) - padding, ...        % Shift x left
        bbox(2) - padding, ...        % Shift y up
        bbox(3) + 2 * padding, ...    % Increase width
        bbox(4) + 2 * padding         % Increase height
    ];
    
    % Ensure padded bounding box stays within image boundaries
    paddedBBox(1) = max(1, paddedBBox(1)); % x >= 1
    paddedBBox(2) = max(1, paddedBBox(2)); % y >= 1
    paddedBBox(3) = min(paddedBBox(3), size(originalImage, 2) - paddedBBox(1) + 1); % width
    paddedBBox(4) = min(paddedBBox(4), size(originalImage, 1) - paddedBBox(2) + 1); % height
    
    % Round to integer values for imcrop
    paddedBBox = round(paddedBBox);

    plateImage = imcrop(originalImage, paddedBBox);
    
    figure;
    imshow(plateImage);

    % Save for debugging (optional)
    %if idx == 2
        %imwrite(plateImage, "result21.png"); % Use PNG for lossless
    %end
    
    % Histogram-based filtering
    isValidPlate = true; % Assume valid unless proven otherwise
    
    % Compute histograms for each RGB channel
    r = plateImage(:,:,1);
    g = plateImage(:,:,2);
    b = plateImage(:,:,3);
    
    [histR, bins] = imhist(r);
    [histG, ~] = imhist(g);
    [histB, ~] = imhist(b);
    
    % Calculate standard deviation of intensities for each channel
    stdR = std(double(r(:)));
    stdG = std(double(g(:)));
    stdB = std(double(b(:)));
    
    % Define intensity range for "middle" (e.g., 50-200 out of 0-255)
    middleRange = (bins >= 50) & (bins <= 200);
    freqMiddleR = sum(histR(middleRange)) / sum(histR); % Fraction of pixels in middle range
    freqMiddleG = sum(histG(middleRange)) / sum(histG);
    freqMiddleB = sum(histB(middleRange)) / sum(histB);
    
    % Filter criteria:
    % - Low standard deviation indicates low contrast (uniform color)
    % - High frequency in middle range indicates a single dominant color
    stdThreshold = 30; % Adjust based on valid plate images (e.g., 30-50 for contrast)
    freqThreshold = 0.8; % Adjust: fraction of pixels in middle range (e.g., 0.8 means 80%)
    
    if (stdR < stdThreshold && stdG < stdThreshold && stdB < stdThreshold) || ...
       (freqMiddleR > freqThreshold && freqMiddleG > freqThreshold && freqMiddleB > freqThreshold)
        isValidPlate = false; % Image is too uniform (single color, normal-like histogram)
        disp(['Skipping plate candidate ', num2str(idx), ': Histogram indicates uniform color']);
    end
    
    % Visualize histogram for debugging
    %figure;
    %subplot(3,1,1); bar(bins, histR); title('red');
    %subplot(3,1,2); bar(bins, histG); title('Green Channel Histogram');
    %subplot(3,1,3); bar(bins, histB); title('Blue Channel Histogram');
    
    % Proceed only if the plate is valid
    if isValidPlate
        % Apply manual OCR approach and get first character bounding box
        addpath("functions");
        [text, firstCharBBox] = plateDetect(plateImage);
        disp(text);
        % Check if the detected text meets criteria
        if ~isempty(text) && any(isstrprop(text, 'digit')) && strlength(text) > 2 && strlength(text) <= 7
            bottom_y = bbox(2) + bbox(4); % Bottom of the bounding box
            validCandidates{end+1} = {text, bbox, bottom_y};
            
            % Adjust first character bounding box to original image coordinates
            if ~isempty(firstCharBBox)
                % Scale factors from resized image (300x500) to plateImage
                plateHeight = size(plateImage, 1);
                plateWidth = size(plateImage, 2);
                scaleX = plateWidth / 500;
                scaleY = plateHeight / 300;

                % Adjust bounding box: [x, y, width, height]
                adjustedFirstCharBBox = [
                    bbox(1) + firstCharBBox(1) * scaleX, ...  % x
                    bbox(2) + firstCharBBox(2) * scaleY, ...  % y
                    firstCharBBox(3) * scaleX, ...           % width
                    firstCharBBox(4) * scaleY                % height
                ];

                % Draw rectangle around first character on rectangleImage
                rectangleImage = insertShape(rectangleImage, 'Rectangle', adjustedFirstCharBBox, ...
                    'Color', 'yellow', 'LineWidth', 2);
            end
        end
    end
end

% 7. Select the best plate region
if ~isempty(validCandidates)
    disp(validCandidates)
    bottom_ys = cellfun(@(x) x{3}, validCandidates);
    [~, max_idx] = max(bottom_ys);
    bestPlateText = validCandidates{max_idx}{1};
    bestPlateBBox = validCandidates{max_idx}{2};

    % Extract the best plate image
    bestPlateImage = imcrop(originalImage, bestPlateBBox);

    % 8. Visualization
    figure('Position', [100, 100, 1200, 800]);

    subplot(2, 3, 1), imshow(originalImage), title('Original Image');
    subplot(2, 3, 2), imshow(dilatedEdges), title('DilatedEdges');
    subplot(2, 3, 3), imshow(filledRegions), title('Filled Regions');
    subplot(2, 3, 4), imshow(rectangleImage), title('Detected Regions');
    subplot(2, 3, 5), imshow(bestPlateImage), title(['Selected Plate: ', bestPlateText]);
else
    disp('No license plate with recognizable text detected');
    figure('Position', [100, 100, 1200, 800]);
    subplot(2, 3, 1), imshow(originalImage), title('Original Image');
    subplot(2, 3, 2), imshow(dilatedEdges), title('DilatedEdges');
    subplot(2, 3, 3), imshow(filledRegions), title('Filled Regions');
    subplot(2, 3, 4), imshow(rectangleImage), title('Detected Regions');
    subplot(2, 3, 5), title('No Plate Detected');
end

