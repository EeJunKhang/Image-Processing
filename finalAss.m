clc
clear
%IMG-20250414-WA0177.jpg - 2 8
%carplate.jpg - 2 8
%ev.jpg - 2 8 - very close (1 character wrong); 1 5 - can

% Read the input image
originalImage = imread('IMG-20250414-WA0204.jpg');

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

for i = 1:length(regionProps)
    bbox = regionProps(i).BoundingBox;
    aspectRatio = bbox(3) / bbox(4);
    area = regionProps(i).Area;

    % Draw rectangle for all detected regions
    rectangleImage = insertShape(rectangleImage, 'Rectangle', bbox, 'Color', 'red', 'LineWidth', 1, LineWidth=5);

    % Filter conditions for potential plate regions
    if (aspectRatio >= 3 && aspectRatio <= 4.5) && (area > 0.001 * imageArea && area < 0.1 * imageArea)
        plateRegions = [plateRegions, i];
        rectangleImage = insertShape(rectangleImage, 'Rectangle', regionProps(i).BoundingBox, 'Color', 'green', 'LineWidth', 2, LineWidth=5);
    end
end

% 6. Plate extraction and manual OCR integration
validCandidates = {};
plateBoundingBoxes = {};
plateIntensities = {};

for idx = 1:length(plateRegions)
    i = plateRegions(idx);
    bbox = regionProps(i).BoundingBox; % Add some padding
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
    
    %figure;
    %imshow(plateImage);
    %if(idx == 4)
        %imwrite(plateImage,"result7.png") % use png for loseless
    %end
    % Apply manual OCR approach and get first character bounding box
    addpath("functions");
    [text, firstCharBBox] = plateDetect(plateImage);
    
    % Perform OCR for debugging
    %t = ocr(plateImage);
    %disp("ocr")
    %disp(t.Text);
    
    if ~isempty(text) && any(isstrprop(text, 'digit')) && strlength(text) > 2
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
                bbox(2) + firstCharBBox(2) * scaleY, ... % y
                firstCharBBox(3) * scaleX, ...           % width
                firstCharBBox(4) * scaleY ...            % height
            ];

            % Draw rectangle around first character on rectangleImage
            rectangleImage = insertShape(rectangleImage, 'Rectangle', adjustedFirstCharBBox, ...
                'Color', 'yellow', 'LineWidth', 2 , LineWidth=5);
           
        end
    end
end
% 7. Select the best plate region
if ~isempty(validCandidates)
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

    % Add text explanation to the detected regions subplot
    %subplot(2, 3, 4);
    %text(10, size(rectangleImage,1) + 20, 'Rectangle Colors:', 'FontWeight', 'bold');
    %text(10, size(rectangleImage,1) + 40, 'Green: Potential Plate Regions', 'Color', 'green');
    %text(10, size(rectangleImage,1) + 60, 'Red: All Detected Regions', 'Color', 'red');
else
    disp('No license plate with recognizable text detected');
    figure('Position', [100, 100, 1200, 800]);
    subplot(2, 3, 1), imshow(originalImage), title('Original Image');
    subplot(2, 3, 2), imshow(dilatedEdges), title('DilatedEdges');
    subplot(2, 3, 3), imshow(filledRegions), title('Filled Regions');
    subplot(2, 3, 4), imshow(rectangleImage), title('Detected Regions');
    subplot(2, 3, 5), title('No Plate Detected');
end

