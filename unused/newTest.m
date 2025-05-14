function [bestPlateText, bestPlateImage] = newTest2(inputImage)
    % DETECTLICENSEPLATE detects Malaysian license plates from various vehicle types
    % Input:
    %   inputImage - RGB image containing vehicle(s)
    % Output:
    %   bestPlateText - Text from the most likely license plate
    %   bestPlateImage - Cropped image of the detected license plate
    
    % Read input image if filename is provided
    if ischar(inputImage)
        originalImage = imread(inputImage);
    else
        originalImage = inputImage;
    end
    
    %% 1. Advanced Preprocessing
    % Convert to grayscale
    grayImage = rgb2gray(originalImage);
    
    % Apply CLAHE for adaptive contrast enhancement
    enhancedImage = adapthisteq(grayImage, 'ClipLimit', 0.02, 'Distribution', 'uniform');
    %disp(size(enhancedImage));
    % Apply bilateral filtering to preserve edges while removing noise
    %denoisedImage = imbilatfilt(enhancedImage, 2, 30);
    denoisedImage = enhancedImage
    
    %% 2. Multi-scale Edge Detection
    % Apply Sobel edge detection
    sobelEdges = edge(denoisedImage, 'sobel');
    
    % Apply Canny edge detection with sensitivity threshold
    cannyEdges = edge(denoisedImage, 'canny', [0.1 0.3]);
    
    % Combine edges from both detectors
    combinedEdges = sobelEdges | cannyEdges;
    
    %% 3. Color-based Segmentation (supplementary approach)
    % Convert to HSV color space for color segmentation
    hsvImg = rgb2hsv(originalImage);
    
    % Extract the saturation and value channels
    saturationChannel = hsvImg(:,:,2);
    valueChannel = hsvImg(:,:,3);
    
    % Create binary masks for common plate colors
    % White plates: high value, low saturation
    whiteMask = (valueChannel > 0.7) & (saturationChannel < 0.3);
    
    % Yellow plates: specific hue range, high saturation
    hueChannel = hsvImg(:,:,1);
    yellowMask = (hueChannel > 0.1) & (hueChannel < 0.2) & (saturationChannel > 0.4) & (valueChannel > 0.4);
    
    % Black plates: low value
    blackMask = (valueChannel < 0.3);
    
    % Combine all plate color masks
    colorMask = whiteMask | yellowMask | blackMask;
    
    % Clean up color mask with morphological operations
    colorMask = imclose(colorMask, strel('rectangle', [5, 20]));
    colorMask = imfill(colorMask, 'holes');
    colorMask = bwareaopen(colorMask, 1000);
    
    %% 4. Morphological Operations for Different Plate Types
    % Define structuring elements for different plate types
    carStrel = strel('rectangle', [3, 15]);      % Elongated for car plates
    motoStrel = strel('rectangle', [3, 5]);      % More square for motorcycle plates
    
    % Apply different dilations
    dilatedEdgesCar = imdilate(combinedEdges, carStrel);
    dilatedEdgesMoto = imdilate(combinedEdges, motoStrel);
    
    % Combine dilated edges
    dilatedEdges = dilatedEdgesCar | dilatedEdgesMoto;
    
    % Fill holes
    filledRegions = imfill(dilatedEdges, 'holes');
    
    % Clean up with opening to remove noise
    filledRegions = imopen(filledRegions, strel('disk', 2));
    
    %% 5. Enhanced Region Extraction
    % Extract region properties
    regionProps = regionprops(filledRegions, 'BoundingBox', 'Area', 'Extent', 'Solidity');
    
    % Image dimensions
    imgHeight = size(originalImage, 1);
    imgWidth = size(originalImage, 2);
    imageArea = imgHeight * imgWidth;
    
    % Initialize
    plateRegions = [];
    rectangleImage = originalImage;  % For visualization
    
    % Define aspect ratio ranges for different vehicle types
    carAspectRatioMin = 3.0;      % Standard car plates (wider)
    carAspectRatioMax = 4.5;
    motoAspectRatioMin = 1.8;     % Motorcycle plates (more square)
    motoAspectRatioMax = 3.0;
    comAspectRatioMin = 3.5;      % Commercial vehicle plates (widest)
    comAspectRatioMax = 5.0;
    
    % Area constraints as percentage of image
    minAreaRatio = 0.0005;  
    maxAreaRatio = 0.1;
    
    % Loop through all regions
    for i = 1:length(regionProps)
        bbox = regionProps(i).BoundingBox;
        aspectRatio = bbox(3) / bbox(4);
        area = regionProps(i).Area;
        solidity = regionProps(i).Solidity;  % Area/ConvexArea - measures "fullness"
        extent = regionProps(i).Extent;      % Area/BoundingBoxArea - measures "rectangularity"
        
        % Draw rectangle for all detected regions
        rectangleImage = insertShape(rectangleImage, 'Rectangle', bbox, 'Color', 'red', 'LineWidth', 1);
        
        % Check if it's within any of the vehicle plate aspect ratio ranges
        isCar = (aspectRatio >= carAspectRatioMin && aspectRatio <= carAspectRatioMax);
        isMoto = (aspectRatio >= motoAspectRatioMin && aspectRatio <= motoAspectRatioMax);
        isCom = (aspectRatio >= comAspectRatioMin && aspectRatio <= comAspectRatioMax);
        
        % Check area constraint
        hasValidArea = (area > minAreaRatio * imageArea && area < maxAreaRatio * imageArea);
        
        % Additional shape constraints
        hasValidShape = (solidity > 0.7) && (extent > 0.6);  % Reasonably solid and rectangular
        
        % Character density check (using edge density)
        bboxInt = round(bbox);
        regionEdges = combinedEdges(bboxInt(2):min(bboxInt(2)+bboxInt(4),size(combinedEdges,1)), ...
                                  bboxInt(1):min(bboxInt(1)+bboxInt(3),size(combinedEdges,2)));
        edgeDensity = sum(regionEdges(:)) / numel(regionEdges);
        hasValidEdgeDensity = (edgeDensity > 0.05) && (edgeDensity < 0.4);  % Not too sparse, not too dense
        
        % Check if it's a valid plate region
        if (isCar || isMoto || isCom) && hasValidArea && hasValidShape && hasValidEdgeDensity
            plateRegions = [plateRegions, i];
            
            % Color code based on plate type
            if isCar
                rectangleImage = insertShape(rectangleImage, 'Rectangle', bbox, 'Color', 'green', 'LineWidth', 2);
            elseif isMoto
                rectangleImage = insertShape(rectangleImage, 'Rectangle', bbox, 'Color', 'blue', 'LineWidth', 2);
            else  % Commercial
                rectangleImage = insertShape(rectangleImage, 'Rectangle', bbox, 'Color', 'cyan', 'LineWidth', 2);
            end
        end
    end
    
    %% 6. Plate Extraction and OCR
    validCandidates = {};
    
    for idx = 1:length(plateRegions)
        i = plateRegions(idx);
        bbox = regionProps(i).BoundingBox;
        
        % Adjust padding based on plate aspect ratio
        aspectRatio = bbox(3) / bbox(4);
        if aspectRatio >= 3.0
            padding = [5, 10];  % [vertical, horizontal] for car/commercial plates
        else
            padding = [8, 8];   % Equal padding for motorcycle plates
        end
        
        % Add padding around the plate
        paddedBBox = [
            bbox(1) - padding(2), ...        % Shift x left
            bbox(2) - padding(1), ...        % Shift y up
            bbox(3) + 2 * padding(2), ...    % Increase width
            bbox(4) + 2 * padding(1)         % Increase height
        ];
        
        % Ensure padded bounding box stays within image boundaries
        paddedBBox(1) = max(1, paddedBBox(1)); 
        paddedBBox(2) = max(1, paddedBBox(2));
        paddedBBox(3) = min(paddedBBox(3), imgWidth - paddedBBox(1) + 1);
        paddedBBox(4) = min(paddedBBox(4), imgHeight - paddedBBox(2) + 1);
        
        % Round to integer values for imcrop
        paddedBBox = round(paddedBBox);
        
        % Extract plate image
        plateImage = imcrop(originalImage, paddedBBox);
        
        % Process plate image for character extraction
        if size(plateImage, 3) == 3
            plateGray = rgb2gray(plateImage);
        else
            plateGray = plateImage;
        end
        
        % Apply adaptive threshold for binarization
        binaryPlate = imbinarize(plateGray, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.4);
        
        % Invert if background is dark (more white pixels than black)
        if sum(binaryPlate(:)) > numel(binaryPlate)/2
            binaryPlate = ~binaryPlate;
        end
        
        % Clean binary image
        binaryPlate = bwareaopen(binaryPlate, 30);  % Remove small objects
        
        % Apply OCR directly or use custom character recognition
        plateText = '';
        
        % Position in the original image (used for scoring)
        centerX = bbox(1) + bbox(3)/2;
        centerY = bbox(2) + bbox(4)/2;
        bottom_y = bbox(2) + bbox(4);
        
        % Apply OCR
        try
            ocrResults = ocr(plateImage, 'CharacterSet', 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789', ...
                            'TextLayout', 'Block');
            
            % Check if OCR returned results
            if ~isempty(ocrResults.Text)
                plateText = strtrim(ocrResults.Text);
                
                % Remove non-alphanumeric characters
                plateText = regexp(plateText, '[A-Z0-9]+', 'match');
                if ~isempty(plateText)
                    plateText = strjoin(plateText, '');
                else
                    plateText = '';
                end
                
                % Calculate character confidence
                if ~isempty(ocrResults.CharacterConfidences)
                    charConfidence = mean(ocrResults.CharacterConfidences(~isnan(ocrResults.CharacterConfidences)));
                else
                    charConfidence = 0;
                end
                
                % Add to candidates if text is not empty
                if ~isempty(plateText)
                    % Create a structure with all relevant information
                    candidateInfo = struct();
                    candidateInfo.text = plateText;
                    candidateInfo.bbox = bbox;
                    candidateInfo.bottom_y = bottom_y;
                    candidateInfo.centerX = centerX;
                    candidateInfo.centerY = centerY;
                    candidateInfo.aspectRatio = aspectRatio;
                    candidateInfo.area = bbox(3) * bbox(4);
                    candidateInfo.confidence = charConfidence;
                    
                    validCandidates{end+1} = candidateInfo;
                end
            end
        catch
            disp(['OCR failed for candidate ', num2str(idx)]);
        end
    end

    %% 7. Result Scoring and Selection
    bestPlateText = '';
    bestPlateImage = [];
    
    if ~isempty(validCandidates)
        disp('Valid candidates found:');
        
        % Initialize scores array
        scores = zeros(1, length(validCandidates));
        
        for i = 1:length(validCandidates)
            candidate = validCandidates{i};
            
            % 1. Text length score (Malaysian plates typically have 5-7 characters)
            textLengthScore = 0;
            textLength = strlength(candidate.text);
            if textLength >= 5 && textLength <= 7
                textLengthScore = 1.0;
            elseif textLength >= 3
                textLengthScore = 0.7;
            else
                textLengthScore = 0.3;
            end
            
            % 2. Text format score (Malaysian plates have specific format)
            textFormatScore = 0;
            
            % Check if text matches Malaysian plate pattern
            hasLetters = any(isstrprop(candidate.text, 'alpha'));
            hasNumbers = any(isstrprop(candidate.text, 'digit'));
            
            % Malaysian plates typically start with letters followed by numbers
            if hasLetters && hasNumbers
                % Try to match common Malaysian plate formats: ABC1234, AB123C, etc.
                if regexp(candidate.text, '^[A-Z]{1,3}\d{1,4}[A-Z]?$')
                    textFormatScore = 1.0;
                else
                    textFormatScore = 0.7;
                end
            elseif hasLetters || hasNumbers
                textFormatScore = 0.4;
            end
            
            % 3. Position score (center of image is better than edges)
            centerXNorm = abs((candidate.centerX / imgWidth) - 0.5) * 2;  % 0 = center, 1 = edge
            centerYNorm = candidate.centerY / imgHeight;  % Normalized y-position (0-1)
            positionScore = (1 - centerXNorm) * 0.6 + centerYNorm * 0.4;  % Weighted toward center and lower
            
            % 4. Aspect ratio score (how well it matches expected plate ratios)
            aspectRatioScore = 0;
            ar = candidate.aspectRatio;
            
            if (ar >= 3.0 && ar <= 4.5)  % Car plate
                aspectRatioScore = 1.0;
            elseif (ar >= 1.8 && ar < 3.0)  % Motorcycle plate
                aspectRatioScore = 0.9;
            elseif (ar > 4.5 && ar <= 5.0)  % Commercial vehicle
                aspectRatioScore = 0.85;
            elseif (ar >= 1.5 && ar <= 5.5)  % Close enough
                aspectRatioScore = 0.5;
            end
            
            % 5. Area score (larger plates are better, up to a point)
            areaNorm = candidate.area / (imgWidth * imgHeight);  % Normalized area (0-1)
            areaScore = 0;
            if (areaNorm >= 0.005 && areaNorm <= 0.05)
                areaScore = 1.0;
            elseif (areaNorm >= 0.001 && areaNorm <= 0.1)
                areaScore = 0.7;
            else
                areaScore = 0.3;
            end
            
            % 6. OCR confidence score
            confidenceScore = min(candidate.confidence * 1.5, 1.0);  % Scale confidence (may be low)
            
            % Calculate weighted final score
            scores(i) = textLengthScore * 0.2 + ...
                        textFormatScore * 0.25 + ...
                        positionScore * 0.15 + ...
                        aspectRatioScore * 0.15 + ...
                        areaScore * 0.1 + ...
                        confidenceScore * 0.15;
            
            disp(['Candidate ', num2str(i), ': ', candidate.text, ...
                 ' - Score: ', num2str(scores(i), '%.2f'), ...
                 ' (Text: ', num2str(textLengthScore), ...
                 ', Format: ', num2str(textFormatScore), ...
                 ', Pos: ', num2str(positionScore, '%.2f'), ...
                 ', AR: ', num2str(aspectRatioScore), ...
                 ', Area: ', num2str(areaScore), ...
                 ', Conf: ', num2str(confidenceScore, '%.2f'), ')']);
        end
        
        % Select the candidate with the highest score
        [~, max_idx] = max(scores);
        bestCandidate = validCandidates{max_idx};
        bestPlateText = bestCandidate.text;
        bestPlateBBox = bestCandidate.bbox;
        
        % Extract the best plate image
        bestPlateImage = imcrop(originalImage, bestPlateBBox);
        
        %% 8. Visualization
        figure('Position', [100, 100, 1200, 800]);

        subplot(2, 3, 1), imshow(originalImage), title('Original Image');
        subplot(2, 3, 2), imshow(enhancedImage), title('Enhanced Image');
        subplot(2, 3, 3), imshow(combinedEdges), title('Edge Detection');
        subplot(2, 3, 4), imshow(filledRegions), title('Region Detection');
        subplot(2, 3, 5), imshow(rectangleImage), title('Candidate Regions');
        subplot(2, 3, 6), imshow(bestPlateImage), title(['Best Plate: ', bestPlateText]);
    else
        disp('No license plate with recognizable text detected');
        figure('Position', [100, 100, 1200, 600]);
        subplot(2, 3, 1), imshow(originalImage), title('Original Image');
        subplot(2, 3, 2), imshow(enhancedImage), title('Enhanced Image');
        subplot(2, 3, 3), imshow(combinedEdges), title('Edge Detection');
        subplot(2, 3, 4), imshow(filledRegions), title('Region Detection');
        subplot(2, 3, 5), imshow(rectangleImage), title('Candidate Regions');
        subplot(2, 3, 6), title('No Valid Plate Detected');
    end
end

clc;
clear;
[file,path]=uigetfile({'*.jpg;*.bmp;*.png;*.tif'},'Choose an image');
s=[path,file];
picture=imread(s);
%test = ocr(picture, CharacterSet="ABCDEFGHJKLMNPQRSTUVWXYZ0123456789", LayoutAnalysis="character", Model="english");
%disp("main ocr")
%disp(test.Text);
[a,b] = newTest2(picture);
