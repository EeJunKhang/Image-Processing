originalImage = imread('Motor/Motor_46.jpg');

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
% Modified: Use two different structuring elements for car and motorcycle plates
carStrel = strel('rectangle', [2 8]);      % Horizontal emphasis for car plates
motoStrel = strel('rectangle', [3 3]);     % More square for motorcycle plates

dilatedEdgesCar = imdilate(edgeImage, carStrel);
dilatedEdgesMoto = imdilate(edgeImage, motoStrel);

% Combine both dilated images to catch all potential plates
dilatedEdges = dilatedEdgesCar | dilatedEdgesMoto;

filledRegions = imfill(dilatedEdges, 'holes');

% Optional: Clean up noise with morphological opening
filledRegions = imopen(filledRegions, strel('disk', 1));

% 5. Candidate region selection with expanded criteria
regionProps = regionprops(filledRegions, 'BoundingBox', 'Area', 'Extent');
plateRegions = [];
imageArea = size(originalImage, 1) * size(originalImage, 2);
rectangleImage = originalImage;

% Define a list of colors for isClose boxes
colorList = {'blue', 'yellow', 'cyan', 'magenta', 'white', 'black'};
colorCount = length(colorList);
colorIndex = 1;

% Define different aspect ratio ranges for different vehicle types
% For Malaysian plates:
carAspectRatioMin = 3.0;    % Standard car plates
carAspectRatioMax = 4.5;
motoAspectRatioMin = 1.8;   % Motorcycle plates are more square
motoAspectRatioMax = 3.0;

% Area constraints as percentage of image (adjusted to be more flexible)
minAreaRatio = 0.0005;  % Smaller to catch motorcycle plates
maxAreaRatio = 0.1;     % Upper bound remains the same

for i = 1:length(regionProps)
    bbox = regionProps(i).BoundingBox;
    aspectRatio = bbox(3) / bbox(4);
    area = regionProps(i).Area;
    
    % Draw rectangle for all detected regions
    rectangleImage = insertShape(rectangleImage, 'Rectangle', bbox, 'Color', 'red', 'LineWidth', 1);
    
    % Combined filter for both car and motorcycle plates
    isCar = (aspectRatio >= carAspectRatioMin && aspectRatio <= carAspectRatioMax);
    isMoto = (aspectRatio >= motoAspectRatioMin && aspectRatio <= motoAspectRatioMax);
    
    hasValidArea = (area > minAreaRatio * imageArea && area < maxAreaRatio * imageArea);
    
    % Check for standard car or motorcycle plate
    if (isCar || isMoto) && hasValidArea
        plateRegions = [plateRegions, i];
        
        % Color code: Green for car plates, Blue for motorcycle plates
        if isCar
            rectangleImage = insertShape(rectangleImage, 'Rectangle', regionProps(i).BoundingBox, 'Color', 'green', 'LineWidth', 2);
        else
            rectangleImage = insertShape(rectangleImage, 'Rectangle', regionProps(i).BoundingBox, 'Color', 'blue', 'LineWidth', 2);
        end
    else
        % For near-miss cases, mark with different colors
        isClose = ((aspectRatio >= motoAspectRatioMin-0.3 && aspectRatio <= carAspectRatioMax+0.5) && ...
                  (area > minAreaRatio * imageArea * 0.5 && area < maxAreaRatio * imageArea * 1.5));

        if isClose
            % Pick a color from the list and cycle
            currentColor = colorList{colorIndex};
            rectangleImage = insertShape(rectangleImage, 'Rectangle', bbox, ...
                'Color', currentColor, 'LineWidth', 2);
            if strcmp(currentColor, 'magenta')
                disp(['Aspect ratio: ', num2str(aspectRatio)]);
                disp(['Area: ', num2str(area)]);
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
    
    % Adjust padding based on the plate type (car vs motorcycle)
    aspectRatio = bbox(3) / bbox(4);
    if aspectRatio >= carAspectRatioMin
        padding = 2;  % Less padding for car plates
    else
        padding = 3;  % More padding for motorcycle plates
    end
    
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
    title(['Candidate ', num2str(idx), ' - AR: ', num2str(bbox(3)/bbox(4))]);
    
    % Histogram-based filtering with adjusted parameters for different plate types
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
    
    % Adjust thresholds based on plate type
    aspectRatio = bbox(3) / bbox(4);
    if aspectRatio >= carAspectRatioMin
        % Car plate thresholds
        stdThreshold = 30;
        freqThreshold = 0.8;
    else
        % Motorcycle plate thresholds - more lenient
        stdThreshold = 25;
        freqThreshold = 0.85;
    end
    
    if (stdR < stdThreshold && stdG < stdThreshold && stdB < stdThreshold) || ...
       (freqMiddleR > freqThreshold && freqMiddleG > freqThreshold && freqMiddleB > freqThreshold)
        isValidPlate = false; % Image is too uniform (single color, normal-like histogram)
        disp(['Skipping plate candidate ', num2str(idx), ': Histogram indicates uniform color']);
    end
    
    % Proceed only if the plate is valid
    if isValidPlate
        % Apply manual OCR approach and get first character bounding box
        addpath("functions");
        [text, firstCharBBox] = plateDetect2(plateImage);
        disp(['Detected text: ', text]);
        
        % Adjust validation criteria for Malaysian plates
        % Malaysian plates typically have 1-7 characters
        hasValidFormat = ~isempty(text) && ...
                         (any(isstrprop(text, 'digit')) || any(isstrprop(text, 'alpha'))) && ...
                         strlength(text) >= 1 && strlength(text) <= 7;
                         
        if hasValidFormat
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

% 7. Select the best plate region using a comprehensive scoring system
if ~isempty(validCandidates)
    disp('Valid candidates found:');
    
    % Initialize scores array
    scores = zeros(1, length(validCandidates));
    
    % Get image dimensions for normalization
    imgHeight = size(originalImage, 1);
    imgWidth = size(originalImage, 2);
    
    for i = 1:length(validCandidates)
        % Extract info from candidate
        candidateText = validCandidates{i}{1};
        candidateBBox = validCandidates{i}{2};
        bottom_y = validCandidates{i}{3};
        
        % Get candidate properties
        aspectRatio = candidateBBox(3) / candidateBBox(4);
        centerX = candidateBBox(1) + candidateBBox(3)/2;
        centerY = candidateBBox(2) + candidateBBox(4)/2;
        plateArea = candidateBBox(3) * candidateBBox(4);
        
        % 1. Text length score (Malaysian plates typically have 5-7 characters)
        textLengthScore = 0;
        textLength = strlength(candidateText);
        if textLength >= 5 && textLength <= 7
            textLengthScore = 1.0;
        elseif textLength >= 3
            textLengthScore = 0.7;
        else
            textLengthScore = 0.3;
        end
        
        % 2. Text format score (Malaysian plates have specific format)
        textFormatScore = 0;
        % Check if text matches Malaysian plate pattern: letters followed by numbers
        hasLetters = any(isstrprop(candidateText, 'alpha'));
        hasNumbers = any(isstrprop(candidateText, 'digit'));
        if hasLetters && hasNumbers
            textFormatScore = 1.0;
        elseif hasLetters || hasNumbers
            textFormatScore = 0.5;
        end
        
        % 3. Position score (center of image is better than edges)
        centerXNorm = abs((centerX / imgWidth) - 0.5) * 2;  % 0 = center, 1 = edge
        centerYNorm = centerY / imgHeight;  % Normalized y-position (0-1)
        positionScore = (1 - centerXNorm) * 0.5 + centerYNorm * 0.5;  % Higher for centered and lower plates
        
        % 4. Aspect ratio score (how well it matches expected plate ratios)
        aspectRatioScore = 0;
        if (aspectRatio >= 3.0 && aspectRatio <= 4.2)  % Car plate
            aspectRatioScore = 1.0;
        elseif (aspectRatio >= 1.8 && aspectRatio <= 3.0)  % Motorcycle plate
            aspectRatioScore = 0.9;
        elseif (aspectRatio >= 1.5 && aspectRatio <= 5.0)  % Close enough
            aspectRatioScore = 0.5;
        end
        
        % 5. Area score (larger plates are better, up to a reasonable size)
        areaNorm = plateArea / (imgWidth * imgHeight);  % Normalized area (0-1)
        areaScore = 0;
        if (areaNorm >= 0.005 && areaNorm <= 0.05)
            areaScore = 1.0;
        elseif (areaNorm >= 0.001 && areaNorm <= 0.1)
            areaScore = 0.7;
        else
            areaScore = 0.3;
        end
        
        % Calculate weighted final score
        scores(i) = textLengthScore * 0.25 + ...
                    textFormatScore * 0.3 + ...
                    positionScore * 0.15 + ...
                    aspectRatioScore * 0.2 + ...
                    areaScore * 0.1;
        
        disp(['Candidate ', num2str(i), ': ', candidateText, ...
             ' - Score: ', num2str(scores(i), '%.2f'), ...
             ' (Text: ', num2str(textLengthScore), ...
             ', Format: ', num2str(textFormatScore), ...
             ', Pos: ', num2str(positionScore, '%.2f'), ...
             ', AR: ', num2str(aspectRatioScore), ...
             ', Area: ', num2str(areaScore), ')']);
    end
    
    % Select the candidate with the highest score
    [~, max_idx] = max(scores);
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
    % Add information about plate type
    plateAR = bestPlateBBox(3) / bestPlateBBox(4);
    if plateAR >= carAspectRatioMin
        plateTypeText = 'Car Plate';
    else
        plateTypeText = 'Motorcycle Plate';
    end
    subplot(2, 3, 6), text(0.1, 0.5, {['Aspect Ratio: ', num2str(plateAR)], plateTypeText}, 'FontSize', 14), axis off;
else
    disp('No license plate with recognizable text detected');
    figure('Position', [100, 100, 1200, 800]);
    subplot(2, 3, 1), imshow(originalImage), title('Original Image');
    subplot(2, 3, 2), imshow(dilatedEdges), title('DilatedEdges');
    subplot(2, 3, 3), imshow(filledRegions), title('Filled Regions');
    subplot(2, 3, 4), imshow(rectangleImage), title('Detected Regions');
    subplot(2, 3, 5), title('No Plate Detected');
end