function [np, firstCharBBox] = plateDetectTest3(img)
    % Perform initial OCR with specified character set
    isMainOCR = false;

    if isMainOCR
        mainOcr = ocr(img, CharacterSet='ABCDEFGHJKLMNPQRSTUVWXYZ0123456789', LayoutAnalysis='block', Model='english');
        disp(mainOcr);
        % Initialize firstCharBBox
        firstCharBBox = [];
        
        % Check if OCR result is valid
        ocrText = strtrim(mainOcr.Text);
        if ~isempty(ocrText) && all(isstrprop(ocrText, 'alphanum')) && ...
           strlength(ocrText) > 2 && strlength(ocrText) <= 8
            % Valid OCR result: extract first character's bounding box
            if ~isempty(mainOcr.Words)
                firstCharBBox = mainOcr.WordBoundingBoxes(1, :);
            end
            np = ocrText;
            return;
        end
    end

    % Image preprocessing
    [~, cc] = size(img);
    picture = imresize(img, [300 500]);
    
    if size(picture, 3) == 3
        picture = rgb2gray(picture);  % Convert to grayscale (0-255)
    end

    figure;
    imshow(picture);
    title('Original Resized Image');

    % Binarize the image using adaptive thresholding
    picture = imbinarize(picture, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.5);
    figure;
    imshow(picture);
    title('After Adaptive Binarization');

    % Apply morphological operations to clean up the image
    se = strel('disk', 3);  % Structuring element for morphological operations
    picture = imclose(picture, se);  % Connect disjointed parts of characters
    picture = imopen(picture, se);   % Remove small noise
    figure;
    imshow(picture);
    title('After Morphological Operations');

    minArea = 250; % Reduced for smaller characters
    maxArea = 8000;

    picture = bwareafilt(picture, [minArea maxArea]);
    figure; imshow(picture); title('After Area Filtering');
    
    % Label and filter regions
    [L, Ne] = bwlabel(picture);
    props = regionprops(L, 'Centroid', 'BoundingBox', 'Area');
    aspect_ratios = [props.BoundingBox];
    aspect_ratios = aspect_ratios(3:4:end) ./ aspect_ratios(4:4:end); % width / height
    valid_idx = [props.Area] >= minArea & [props.Area] <= maxArea & ...
        aspect_ratios > 0.2 & aspect_ratios < 1.5;
    valid_props = props(valid_idx);
    Ne = length(valid_props); % Update number of detected characters
    disp(['Number of detected characters after filtering: ', num2str(Ne)]);
    
    % Redraw bounding boxes with filtered regions
    figure; imshow(picture); hold on;
    centroids = zeros(Ne, 2);
    for n = 1:Ne
        thisBB = valid_props(n).BoundingBox;
        rectangle('Position', thisBB, 'EdgeColor', 'r', 'LineWidth', 1.5);
        centroids(n, :) = valid_props(n).Centroid;
    end
    hold off;

    % Determine if the plate has one or two rows
    sorted_idx = 1:Ne;  % Default: no sorting
    if Ne > 2  % Need at least 3 characters to analyze tilt or rows
        % Fit a line to centroids to check for tilt (y = mx + b)
        x = centroids(:, 1);  % x-coordinates
        y = centroids(:, 2);  % y-coordinates
        p = polyfit(x, y, 1);  % Linear fit: y = p(1)*x + p(2)
        y_fit = polyval(p, x);  % Predicted y-values
        residuals = abs(y - y_fit);  % Distance from each centroid to the line
        
        % Debug: Print residuals
        disp('Residuals from linear fit:');
        disp(residuals);
        
        % Threshold to determine if it's a single tilted row
        residual_threshold = 20;  % Pixels; adjust based on plate size
        if max(residuals) < residual_threshold
            % Single row (possibly tilted): sort by x-coordinate (left to right)
            [~, sorted_idx] = sort(centroids(:, 1));
            disp('Detected single row (possibly tilted). Sorting left to right.');
        else
            % Two rows: group by y-coordinate and sort within rows
            y_coords = centroids(:, 2);
            [sorted_y, idx] = sort(y_coords);
            y_diff = diff(sorted_y);
            % Find gap to separate rows
            row_threshold = mean(y_diff) + std(y_diff);  % Heuristic
            row_groups = cumsum([1; y_diff > row_threshold]);
            
            % Map row groups back to original indices
            row_assignments = zeros(Ne, 1);
            for i = 1:Ne
                row_assignments(idx(i)) = row_groups(i);
            end
            
            % Sort by row (top to bottom), then x within row (left to right)
            [~, sorted_idx] = sortrows([row_assignments, centroids(:, 1)]);
            disp('Detected two rows. Sorting top to bottom, left to right.');
        end
    elseif Ne == 2
        % For 2 characters, check vertical separation
        y_diff = abs(centroids(1, 2) - centroids(2, 2));
        if y_diff > 30  % Adjust threshold based on plate size
            % Two rows
            [~, sorted_idx] = sort(centroids(:, 2));  % Sort by y (top to bottom)
            disp('Detected two rows (2 characters). Sorting top to bottom.');
        else
            % Single row
            [~, sorted_idx] = sort(centroids(:, 1));  % Sort by x (left to right)
            disp('Detected single row (2 characters). Sorting left to right.');
        end
    end
    
    firstCharBBox = [];
    final_output = [];
    padding = 5;
    for n = 1:Ne
        % Process characters in sorted order
        char_idx = sorted_idx(n);
        [r, c] = find(L == char_idx);
       
        r1 = max(min(r) - padding, 1);
        r2 = min(max(r) + padding, size(picture, 1));
        c1 = max(min(c) - padding, 1);
        c2 = min(max(c) + padding, size(picture, 2));

        n1 = picture(r1:r2, c1:c2);  % Crop the character
        n1 = imresize(n1, [100, 100]);  % Resize for OCR
        
        % Store bounding box for the first character in sorted order
        if n == 1
            firstCharBBox = [min(c), min(r), max(c) - min(c) + 1, max(r) - min(r) + 1];
        end
        n1 = imcomplement(n1);  % Invert for OCR (white text on black background)
        ans = ocr(n1, CharacterSet="ABCDEFGHJKLMNPQRSTUVWXYZ0123456789", LayoutAnalysis="character", Model="english").Text;
        charText = strtrim(ans);
        if ~isempty(charText) && all(isstrprop(charText, 'alphanum')) && strlength(charText) == 1
            final_output = [final_output charText];
        else
            disp('Skipped noisy OCR output');
        end
    end
    
    np = final_output;
end

% Main script to run the function
clc;
clear;
[file, path] = uigetfile({'*.jpg;*.bmp;*.png;*.tif'}, 'Choose an image');
s = [path, file];
picture = imread(s);
[a, b] = plateDetectTest3(picture);
disp('Recognized characters:');
disp(a);