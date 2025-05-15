function [croppedPlate, plateType, stepsImages, plateText] = main(img, isSmallPlate)

    % Initialize stepsImages to store processing images and descriptions
    stepsImages = cell(6, 2); % 6 steps: Grayscale, Contrast, Denoise, Edge, Region, Cropped
    
    
    % Step 1: Convert to grayscale
    if size(img, 3) == 3
        grayImage = rgb2gray(img);
    else
        grayImage = img;
    end
    stepsImages{1, 1} = grayImage;
    stepsImages{1, 2} = 'Grayscale Conversion: Convert the RGB image to grayscale for processing.';
    
    % Apply preprocessing techniques
    % Step 2: Contrast enhancement
    enhancedImage = imadjust(grayImage); % Boost contrast
    enhancedImage = adapthisteq(enhancedImage);
    stepsImages{2, 1} = enhancedImage;
    stepsImages{2, 2} = 'Contrast Enhancement: Adjust contrast and apply adaptive histogram equalization.';
    
    % Step 3: Noise reduction
    denoisedImage = medfilt2(enhancedImage);
    stepsImages{3, 1} = denoisedImage;
    stepsImages{3, 2} = 'Noise Reduction: Apply median filtering to remove noise.';
    
    % Step 4: Edge detection (Sobel operator)
    edgeImage = edge(denoisedImage, 'sobel');
    stepsImages{4, 1} = edgeImage;
    stepsImages{4, 2} = 'Edge Detection: Use Sobel operator to detect edges of the license plate.';
    
    % Morphological operations for plate region detection
    if isSmallPlate
        dilatedEdges = imdilate(edgeImage, strel('rectangle', [3 3]));
    else
        dilatedEdges = imdilate(edgeImage, strel('rectangle', [2 8]));
    end
    stepsImages{5, 1} = dilatedEdges;
    stepsImages{5, 2} = 'Morphological operations: Connects edge fragments using dilation.';
    filledRegions = imfill(dilatedEdges, 'holes');
    stepsImages{6, 1} = filledRegions;
    stepsImages{6, 2} = 'Morphological operations: Fills the interiors of connected edge regions to get complete, filled candidate plate regions.';
    
    % Candidate region selection
    regionProps = regionprops(filledRegions, 'BoundingBox', 'Area', 'Extent');
    plateRegions = [];
    imageArea = size(img, 1) * size(img, 2);
    rectangleImage = img;
    
    % Define a list of colors for isClose boxes
    colorList = {'blue', 'yellow', 'cyan', 'magenta', 'white', 'black'};
    colorCount = length(colorList);
    colorIndex = 1;

    for i = 1:length(regionProps)
        bbox = regionProps(i).BoundingBox;
        aspectRatio = bbox(3) / bbox(4);
        area = regionProps(i).Area;
    
        % Draw rectangle for all detected regions
        rectangleImage = insertShape(rectangleImage, 'Rectangle', bbox, 'Color', 'red', 'LineWidth', 1, LineWidth=5);
    
        % Filter conditions for potential plate regions
        if(~isSmallPlate)
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
                    % Cycle color index
                    colorIndex = mod(colorIndex, colorCount) + 1;
                end
            end
        else % small plate
            if (aspectRatio >= 1.3 && aspectRatio <= 3) && (area > 0.0005 * imageArea && area < 0.1 * imageArea)
                plateRegions = [plateRegions, i];
                rectangleImage = insertShape(rectangleImage, 'Rectangle', regionProps(i).BoundingBox, 'Color', 'green', 'LineWidth', 2, LineWidth=5);
            else
                isClose = (aspectRatio >= 1.0 && aspectRatio <= 4.5) && ...
                    (area > 0.0001 * imageArea && area < 0.15 * imageArea);
    
                if isClose
                    % Pick a color from the list and cycle
                    currentColor = colorList{colorIndex};
                    rectangleImage = insertShape(rectangleImage, 'Rectangle', bbox, ...
                        'Color', currentColor, 'LineWidth', 2);

                    % Cycle color index
                    colorIndex = mod(colorIndex, colorCount) + 1;
                end
            end
        end
    end
    stepsImages{7, 1} = rectangleImage;
    stepsImages{7, 2} = 'Region Selection: Identify potential license plate regions with bounding boxes.';
    
    % 6. Plate extraction and manual OCR integration
    validCandidates = {};
    plateDetectSteps = {}; % To store steps from plateDetect5
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
        paddedBBox(3) = min(paddedBBox(3), size(img, 2) - paddedBBox(1) + 1); % width
        paddedBBox(4) = min(paddedBBox(4), size(img, 1) - paddedBBox(2) + 1); % height
    
        % Round to integer values for imcrop
        paddedBBox = round(paddedBBox);
    
        plateImage = imcrop(img, paddedBBox);
    
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
        end
    
        % Proceed only if the plate is valid
        if isValidPlate
            % Apply manual OCR approach and get first character bounding box
            addpath("functions");
            [text, background_is_white] = plateDetect5(plateImage);

            % Check if the detected text meets criteria
            if ~isempty(text) && any(isstrprop(text, 'alpha')) && any(isstrprop(text, 'digit')) && strlength(text) > 2 && strlength(text) <= 7
                bottom_y = bbox(2) + bbox(4); % Bottom of the bounding box
                validCandidates{end+1} = {text, bbox, bottom_y, background_is_white};
            end
        end
    end
    
    % 7. Select the best plate region
    if ~isempty(validCandidates)
        %disp(validCandidates)
        bottom_ys = cellfun(@(x) x{3}, validCandidates);
        [~, max_idx] = max(bottom_ys);
        bestPlateText = validCandidates{max_idx}{1};
        bestPlateBBox = validCandidates{max_idx}{2};
        background_is_white = validCandidates{max_idx}{4};
    
        % Extract the best plate image
        bestPlateImage = imcrop(img, bestPlateBBox);
        %disp(bestPlateBBox);
        
        croppedPlate = bestPlateImage;
        stepsImages{8, 1} = croppedPlate;
        stepsImages{8, 2} = 'Plate Cropping: Extract the detected license plate region.';
        
        [~, background_is_white, detectSteps] = plateDetect5(bestPlateImage);
        plateDetectSteps = detectSteps;
        plateText = bestPlateText;

        % Determine plate type
        if bestPlateText(1) == 'Z'
            plateType = 'Military  Plate';
        elseif background_is_white
            plateType = 'White Background Plate';
        else
            plateType = 'Normal Plate';
        end
    else
        croppedPlate = zeros(50, 50, 3, 'uint8'); % Empty image
        plateType = 'None';
        plateText = '';
        stepsImages{8, 1} = croppedPlate;
        stepsImages{8, 2} = 'Plate Cropping: No valid plate detected.';
        plateDetectSteps = cell(6, 2);
    end

    stepsImages = [stepsImages; plateDetectSteps];
end