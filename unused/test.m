% Main script for car plate detection
% Load the image and detect plates
image = imread('proton.jpg');
minPlateArea = 4100;
maxPlateArea = 15000;
plates = find_possible_plates(image, minPlateArea, maxPlateArea);

if ~isempty(plates)
    for i = 1:length(plates)
        plate_img = plates{i}{1};
        characters = plates{i}{2};

        disp(i);
        % Display the detected plate
        figure;
        imshow(plate_img);
        title(['Detected Plate ', num2str(i)]);
        % Display segmented characters
        for j = 1:length(characters)
            figure;
            imshow(characters{j});
            title(['Character ', num2str(j)]);
        end
    end
else
    disp('No plates found');
end

function morph_n_thresholded_img = preprocess(input_img)
    % Step 1: Reduce noise with Gaussian blur and convert to grayscale
    blurred = imgaussfilt(input_img, 1.2); % Sigma approximated from 7x7 kernel
    gray = rgb2gray(blurred);
    
    % Step 2: Find vertical edges using Sobel operator in x-direction
    sobel_x = [1 0 -1; 2 0 -2; 1 0 -1];
    grad_x = imfilter(gray, sobel_x, 'conv');
    abs_grad_x = abs(grad_x);
    
    % Step 3: Binarize using Otsu's thresholding
    level = graythresh(abs_grad_x);
    threshold_img = imbinarize(abs_grad_x, level);
    
    % Step 4: Apply morphological closing
    element = strel('rectangle', [3, 22]); % Matches Python's (22,3) kernel
    morph_n_thresholded_img = imclose(threshold_img, element);
end

function isPlate = ratioCheck(area, width, height, minPlateArea, maxPlateArea)
    % Helper function to validate plate area and aspect ratio
    ratioMin = 3;
    ratioMax = 6;
    aspect_ratio = width / height;
    if aspect_ratio < 1
        aspect_ratio = 1 / aspect_ratio;
    end
    isPlate = (area >= minPlateArea && area <= maxPlateArea) && ...
              (aspect_ratio >= ratioMin && aspect_ratio <= ratioMax);
end

function [cleaned_plate, plateFound, coordinates] = clean_plate(plate, minPlateArea, maxPlateArea)
    % Step 7: Clean the plate by finding the largest contour
    gray = rgb2gray(plate);
    thresh = imbinarize(gray, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.4);
    stats = regionprops(thresh, 'Area', 'BoundingBox');
    
    if ~isempty(stats)
        [max_area, idx] = max([stats.Area]);
        if max_area >= minPlateArea && max_area <= maxPlateArea
            bbox = stats(idx).BoundingBox;
            x = floor(bbox(1));
            y = floor(bbox(2));
            w = floor(bbox(3));
            h = floor(bbox(4));
            if ratioCheck(max_area, w, h, minPlateArea, maxPlateArea)
                cleaned_plate = plate(y:y+h-1, x:x+w-1, :);
                plateFound = true;
                coordinates = [x, y, w, h];
                return;
            end
        end
    end
    cleaned_plate = plate;
    plateFound = false;
    coordinates = [];
end

function characters = segment_chars(plate_img, fixed_width)
    % Steps 8-13: Segment characters from the plate
    if nargin < 2
        fixed_width = 400;
    end
    
    % Step 8: Extract value channel from HSV
    hsv = rgb2hsv(plate_img);
    V = hsv(:,:,3);
    
    % Step 9: Apply adaptive thresholding
    thresh = imbinarize(V, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.4);
    
    % Step 10: Invert the thresholded image
    thresh = ~thresh;
    
    % Resize to canonical size
    scale = fixed_width / size(plate_img, 2);
    new_height = round(size(plate_img, 1) * scale);
    plate_img_resized = imresize(plate_img, [new_height, fixed_width]);
    thresh_resized = imresize(thresh, [new_height, fixed_width], 'nearest');
    
    % Step 11: Find connected components and construct character candidates
    labels = bwlabel(thresh_resized);
    stats = regionprops(thresh_resized, 'BoundingBox', 'Area', 'Extent');
    charCandidates = false(size(labels));
    
    for i = 1:length(stats)
        bbox = stats(i).BoundingBox;
        boxW = bbox(3);
        boxH = bbox(4);
        if boxW > 0 && boxH > 0
            aspectRatio = boxW / boxH;
            extent = stats(i).Extent; % Area / BoundingRectArea
            heightRatio = boxH / size(plate_img_resized, 1);
            if aspectRatio < 1.0 && extent > 0.15 && heightRatio > 0.5 && ...
               heightRatio < 0.95 && boxW > 14
                charCandidates(labels == i) = true;
            end
        end
    end
    
    % Step 12 & 13: Find contours in character candidates and extract regions
    char_stats = regionprops(charCandidates, 'BoundingBox', 'Area');
    characters = {};
    if ~isempty(char_stats)
        % Sort by x-coordinate (left to right)
        bboxes = cat(1, char_stats.BoundingBox);
        [~, order] = sort(bboxes(:,1));
        sorted_char_stats = char_stats(order);
        
        addPixel = 4;
        for i = 1:length(sorted_char_stats)
            bbox = sorted_char_stats(i).BoundingBox;
            x = floor(bbox(1));
            y = floor(bbox(2));
            w = floor(bbox(3));
            h = floor(bbox(4));
            y_start = max(y - addPixel, 1);
            x_start = max(x - addPixel, 1);
            y_end = min(y + h + addPixel, size(thresh_resized, 1));
            x_end = min(x + w + addPixel, size(thresh_resized, 2));
            temp = thresh_resized(y_start:y_end, x_start:x_end);
            characters{end+1} = uint8(temp) * 255; % Convert to uint8 for consistency
        end
    end
end

function plates = find_possible_plates(input_img, minPlateArea, maxPlateArea)
    % Steps 5-7: Find and validate plate regions
    after_preprocess = preprocess(input_img);
    % Step 5: Find contours
    stats = regionprops(after_preprocess, 'BoundingBox', 'Area');
    plates = {};
    
    for i = 1:length(stats)
        bbox = stats(i).BoundingBox;
        area = stats(i).Area;
        width = bbox(3);
        height = bbox(4);
        if ratioCheck(area, width, height, minPlateArea, maxPlateArea)
            min_col = ceil(bbox(1));
            max_col = min_col + floor(bbox(3)) - 1;
            min_row = ceil(bbox(2));
            max_row = min_row + floor(bbox(4)) - 1;
            plate = input_img(min_row:max_row, min_col:max_col, :);
            % Clean the plate and segment characters
            [cleaned_plate, plateFound, ~] = clean_plate(plate, minPlateArea, maxPlateArea);
            if plateFound
                characters = segment_chars(cleaned_plate);
                plates{end+1} = {cleaned_plate, characters}; % Store as nested cell array
            end
        end
    end
end