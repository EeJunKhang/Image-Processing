clc;
clear;
[file,path]=uigetfile({'*.jpg;*.bmp;*.png;*.tif'},'Choose an image');
s=[path,file];
img=imread(s);

% Initialize stepsImages to store processing images and descriptions
stepsImages = cell(6, 2); % 6 steps
figure; imshow(img);

% Image correlation method
picture = imresize(img, [300 500]);

if size(picture, 3) == 3
    picture = rgb2gray(picture);  % Convert to grayscale (0-255)
end
stepsImages{1, 1} = picture;
stepsImages{1, 2} = 'Grayscale Conversion: Convert the cropped plate image to grayscale.';
figure; imshow(picture);

% Detect background color using border sampling and edge detection
[rows, cols] = size(picture);
border_size = round(0.1 * min(rows, cols)); % 10% of smaller dimension
top_border = picture(1:border_size, :);
bottom_border = picture(end-border_size+1:end, :);
left_border = picture(:, 1:border_size);
right_border = picture(:, end-border_size+1:end);

% Concatenate border regions
border_pixels = [top_border(:); bottom_border(:); left_border(:); right_border(:)];

% Apply edge detection to the entire image to exclude text regions
edge_map = edge(picture, 'canny'); % Canny edge detector
% Dilate edges slightly to ensure text regions are excluded
se_dilate = strel('disk', 2);
edge_map_dilated = imdilate(edge_map, se_dilate);

% Sample non-edge pixels from border regions
border_mask = false(size(picture));
border_mask(1:border_size, :) = true;
border_mask(end-border_size+1:end, :) = true;
border_mask(:, 1:border_size) = true;
border_mask(:, end-border_size+1:end) = true;
non_edge_border_pixels = picture(border_mask & ~edge_map_dilated);

% If no non-edge pixels (e.g., noisy edges), fall back to all border pixels
if isempty(non_edge_border_pixels)
    non_edge_border_pixels = border_pixels;
end

% Compute mean intensity of sampled pixels
mean_intensity = mean(double(non_edge_border_pixels));
background_is_white = mean_intensity > 128; % Threshold for white vs. black

% Binarize with adaptive thresholding, adjusting polarity based on background
if background_is_white
    % White background, dark text: foreground is dark
    picture = imbinarize(picture, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.5);
    disp("is white background");
else
    % Black background, white text: foreground is bright
    picture = imbinarize(picture, 'adaptive', 'ForegroundPolarity', 'bright', 'Sensitivity', 0.5);
end
stepsImages{2, 1} = picture;
stepsImages{2, 2} = 'Adaptive Binarization: Binarize the image to highlight text, adjusting for background color.';
figure; imshow(picture);

% Ensure white text (1) on black background (0)
% For 'dark' polarity, imbinarize makes dark text white (1), which is correct
% For 'bright' polarity, imbinarize makes bright text white (1), which is correct
% No imcomplement needed here, as both cases produce white text on black background
%figure; imshow(picture); title('After Adaptive Binarization');

% Apply morphological operations
se = strel('disk', 3);
picture = imclose(picture, se); % Connect disjointed characters
picture = imopen(picture, se);  % Remove small noise
stepsImages{3, 1} = picture;
stepsImages{3, 2} = 'Morphological Operations: Apply closing and opening to connect characters and remove noise.';
figure; imshow(picture);

% Separate linked characters using erosion
se_erode = strel('disk', 1);  % Small structuring element to break connections
picture = imerode(picture, se_erode);  % Erode to separate touching characters
stepsImages{4, 1} = picture;
stepsImages{4, 2} = 'Erosion: Erode the image to separate touching characters.';
figure; imshow(picture);


if background_is_white
    picture = imcomplement(picture);
end

% Adjust area filtering
minArea = 600; % Reduced for smaller characters
maxArea = 12000;
picture = bwareafilt(picture, [minArea maxArea]);
stepsImages{5, 1} = picture;
stepsImages{5, 2} = 'Area Filtering: Remove regions outside the expected size range for characters.';
figure; imshow(picture); title("Area Filtering");

% Label and filter regions
[L, Ne] = bwlabel(picture);
props = regionprops(L, 'Centroid', 'BoundingBox', 'Area');
aspect_ratios = [props.BoundingBox];
aspect_ratios = aspect_ratios(3:4:end) ./ aspect_ratios(4:4:end); % width / height
valid_idx = [props.Area] >= minArea & [props.Area] <= maxArea & ...
    aspect_ratios > 0.2 & aspect_ratios < 1.5;
valid_props = props(valid_idx);
Ne = length(valid_props); % Update number of detected characters

% Create an image with bounding boxes for visualization
segImage = uint8(picture) * 255; % Convert logical to uint8 (0 or 255)
segImage = cat(3, segImage, segImage, segImage); % Replicate to RGB
for n = 1:Ne
    thisBB = valid_props(n).BoundingBox;
    segImage = insertShape(segImage, 'Rectangle', thisBB, 'Color', 'red', 'LineWidth', 2);
end
stepsImages{6, 1} = segImage;
stepsImages{6, 2} = 'Character Segmentation: Label and filter character regions, shown with bounding boxes.';
figure; imshow(segImage); title("Character Segmentation");
centroids = zeros(Ne, 2);
for n = 1:Ne
    centroids(n, :) = valid_props(n).Centroid;
end

% Determine if the plate has one or two rows
sorted_idx = 1:Ne;  % Default: no sorting
if Ne > 2  % Need at least 3 characters to analyze tilt or rows
    % Fit a line to centroids to check for tilt (y = mx + b)
    x = centroids(:, 1);  % x-coordinates
    y = centroids(:, 2);  % y-coordinates
    p = polyfit(x, y, 1);  % Linear fit: y = p(1)*x + p(2)
    y_fit = polyval(p, x);  % Predicted y-values
    residuals = abs(y - y_fit);  % Distance from each centroid to the line

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
    else
        % Single row
        [~, sorted_idx] = sort(centroids(:, 1));  % Sort by x (left to right)
    end
end

% Initialize variables to store OCR results and images
ocr_results = cell(Ne, 1);  % Store recognized characters
cropped_images = cell(Ne, 1);  % Store cropped and processed images for visualization

% Character set for OCR
char_set = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789';

% Padding size (pixels)
padding_size = 4;  % Adjust as needed

% Process each valid region in the order specified by sorted_idx
for i = 1:Ne
    % Get the index of the valid_props corresponding to sorted_idx
    prop_idx = sorted_idx(i);
    % Get the bounding box for the current region
    thisBB = valid_props(prop_idx).BoundingBox;
    x = floor(thisBB(1));
    y = floor(thisBB(2));
    width = ceil(thisBB(3));
    height = ceil(thisBB(4));

    % Crop the region from the binary image
    cropped = imcrop(picture, [x, y, width, height]);

    % Resize to 150x150
    cropped_resized = imresize(cropped, [150 150]);

    % Add padding
    padded = padarray(cropped_resized, [padding_size padding_size], 0, 'both');

    %figure; imshow(padded); title(['Processed Character ', num2str(i)]);

    % Store the processed image
    cropped_images{i} = padded;
    padded = imcomplement(padded);
    figure; imshow(padded); title(['Processed Character ', num2str(i)]);
    % Apply OCR
    ocr_result = ocr(padded, 'CharacterSet', char_set, 'LayoutAnalysis', 'character', 'Model', 'english').Text;
    disp(ocr_result);
    % Filter OCR result: ensure it's a single character from the character set
    ocr_result = strtrim(ocr_result);  % Remove whitespace
    if ismember(ocr_result, char_set)
        ocr_results{i} = ocr_result;
    else
        ocr_results{i} = '';  % Store empty if invalid

    end
end

% Combine valid OCR results into a single string (ignoring empty results)
valid_ocr = ocr_results(~cellfun('isempty', ocr_results));
final_plate = strjoin(valid_ocr, '');
np = final_plate;

disp(np)