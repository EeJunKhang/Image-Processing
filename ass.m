clc
    % Read the input image
    originalImage = imread('vois.jpg');
    
    % Convert to grayscale
    grayImage = rgb2gray(originalImage);
    
    % Apply preprocessing techniques
    % 1. Contrast enhancement
    enhancedImage = imadjust(grayImage); % Boost contrast
    %enhancedImage = imadjust(grayImage, stretchlim(grayImage), []);
    enhancedImage = adapthisteq(enhancedImage);
    
    % 2. Noise reduction
    denoisedImage = medfilt2(enhancedImage);
    
    % 3. Edge detection (Sobel operator)
    edgeImage = edge(denoisedImage, 'sobel');
    
    % 4. Morphological operations for plate region detection
    % Dilation to connect potential plate regions
    dilatedEdges = imdilate(edgeImage, strel('rectangle', [5 20]));
    
    % Fill holes
    filledRegions = imfill(dilatedEdges, 'holes');
    
    % 5. Candidate region selection
    % Use region properties to filter potential plate regions
    regionProps = regionprops(filledRegions, ...
        'BoundingBox', 'Area', 'Extent');
    
    % Criteria for Malaysian license plate:
    % - Typical aspect ratio: ~3:1 to 4:1
    % - Minimum and maximum area based on image size
    plateRegions = [];
    
    % Create a copy of the original image to draw rectangles
    rectangleImage = originalImage;
    
    for i = 1:length(regionProps)
        % Calculate aspect ratio manually
        bbox = regionProps(i).BoundingBox;
        aspectRatio = bbox(3) / bbox(4);
        
        % Area check (proportional to image size)
        area = regionProps(i).Area;
        imageArea = size(originalImage, 1) * size(originalImage, 2);

         % Draw rectangle for all detected regions
         rectangleImage = insertShape(rectangleImage, 'Rectangle', ...
            bbox, 'Color', 'red', 'LineWidth', 1, LineWidth=5);

        % Filter conditions
        if (aspectRatio >= 3 && aspectRatio <= 4.5) && ...
           (area > 0.001 * imageArea && area < 0.1 * imageArea)
            plateRegions = [plateRegions, i];
            
            % Draw rectangle for potential plate regions
            rectangleImage = insertShape(rectangleImage, 'Rectangle', ...
                regionProps(i).BoundingBox, 'Color', 'green', 'LineWidth', 2, LineWidth=5);
            
           
        else
            % Draw rectangle for non-plate regions in blue
            %rectangleImage = insertShape(rectangleImage, 'Rectangle', ...
                %bbox, 'Color', 'blue', 'LineWidth', 1);
        end
    end
    
    % 6. Plate extraction and OCR
    if ~isempty(plateRegions)
        % Select the most likely plate region
        bestPlateIndex = plateRegions(1);
        plateBoundingBox = round(regionProps(bestPlateIndex).BoundingBox);
        
        % Extract plate region
        plateImage = imcrop(grayImage, plateBoundingBox);
        
        % Binarization
        binaryPlate = imbinarize(plateImage, 'adaptive');
        
        % 7. Character Segmentation
        % Use connected component analysis
        connectedComponents = bwlabel(binaryPlate);
        charProps = regionprops(connectedComponents, ...
            'BoundingBox', 'Area', 'Extent');
        
        % Filter and sort characters
        validChars = [];
        for j = 1:length(charProps)
            % Character filtering criteria
            charBBox = charProps(j).BoundingBox;
            charAspectRatio = charBBox(3) / charBBox(4);
            
            if (charAspectRatio > 0.2 && charAspectRatio < 1) && ...
               (charProps(j).Area > 50)
                validChars = [validChars, j];
            end
        end
        
        % 8. Visualization
        figure('Position', [100, 100, 1200, 800]);

        plateImage = imcrop(originalImage, plateBoundingBox); % Crop from original color image
        
        % Subplots for different stages of processing
        subplot(2,2,1), imshow(originalImage), title('Original Image');
        subplot(2,2,2), imshow(dilatedEdges), title('Edge Detection');
        subplot(2,2,3), imshow(filledRegions), title('Plate Region');
        subplot(2,2,4), imshow(rectangleImage), title('Detected Regions');
        
        % Add text explanation to the rectangle visualization
        %subplot(2,2,4);
        %text(10, size(rectangleImage,1) + 20, ...
            %'Rectangle Colors:', 'FontWeight', 'bold');
        %text(10, size(rectangleImage,1) + 40, ...
            %'Green: Potential Plate Regions', 'Color', 'green');
        %text(10, size(rectangleImage,1) + 60, ...
            %'Red: Plate Region Boundaries', 'Color', 'red');
        %text(10, size(rectangleImage,1) + 80, ...
            %'Blue: Non-Plate Regions', 'Color', 'blue');
        
        % Optional: OCR (Note: Requires Computer Vision Toolbox)
        try
            %ocrResults = ocr(binaryPlate);
            %disp('Detected License Plate:');
            %disp(ocrResults.Text);
            
            % Display OCR results in the last subplot
            %subplot(2,3,5);
            %text(0.1, 0.5, ['OCR Result: ', ocrResults.Text], ...
                %'FontSize', 12, 'Interpreter', 'none');
            %title('OCR Results');
            axis off;
        catch
            disp('OCR could not be performed');
        end
    else
        disp('No license plate detected');
    end
