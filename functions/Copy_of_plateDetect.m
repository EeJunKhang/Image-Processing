function [np, firstCharBBox] = plateDetect(img)
    % Image correlation method
    % Matches 2 matrices
    load('imgfildata2.mat');
    [~, cc] = size(img);
    picture = imresize(img, [300 500]);
    
    if size(picture, 3) == 3
        picture = rgb2gray(picture);  % Convert to grayscale (0-255)
    end

    %figure;
    %imshow(picture);

    % Detect background color using histogram
    hist = imhist(picture);
    [~, peak_idx] = max(hist);  % Find the most frequent intensity
    peak_intensity = peak_idx - 1;  % Convert to 0-255 scale (imhist bins are 1-256)
    
    % Classify background: assume background is the dominant intensity
    background_is_white = peak_intensity > 128;  % Threshold for white vs. black
    
    % Binarize the image
    threshold = graythresh(picture);  % Otsu's threshold
    if background_is_white
        % White background, black text: invert to make text white
        picture = ~im2bw(picture, threshold);
        %disp('Detected white background. Inverted binary image.');
    else
        % Black background, white text: no inversion needed
        picture = im2bw(picture, threshold);
        %disp('Detected black background. No inversion needed.');
    end
    %figure;
    %imshow(picture);
    picture = bwareaopen(picture, 30);  % Remove small objects (< 30 pixels)
    %figure;
    %imshow(picture);

    minArea = 200;  % Minimum area for characters (adjust as needed)
    maxArea = 10000; % Maximum area to remove large objects like rectangles (adjust as needed)
    
    % Keep objects with area between minArea and maxArea
    %picture = bwareafilt(picture, [minArea maxArea]);

    if cc > 2000
        picture1 = bwareaopen(picture, 3500);  % Remove objects < 3500 pixels
    else
        picture1 = bwareaopen(picture, 3000);  % Remove objects < 3000 pixels
    end
    
    picture2 = picture - picture1;  % Isolate number plate
    picture2 = bwareaopen(picture2, 200);  % Keep only text in number plate
    
    [L, Ne] = bwlabel(picture2);  % Label connected components (characters)
    %disp(['Number of detected characters: ', num2str(Ne)]);
    
    % Get properties of each labeled region
    props = regionprops(L, 'Centroid', 'BoundingBox');
    
    % Initialize arrays to store character positions
    centroids = zeros(Ne, 2);  % [x, y] for each character
    for n = 1:Ne
        centroids(n, :) = props(n).Centroid;  % [x, y]
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
        
        % Debug: Print residuals
        %disp('Residuals from linear fit:');
        %disp(residuals);
        
        % Threshold to determine if it's a single tilted row
        residual_threshold = 20;  % Pixels; adjust based on plate size
        if max(residuals) < residual_threshold
            % Single row (possibly tilted): sort by x-coordinate (left to right)
            [~, sorted_idx] = sort(centroids(:, 1));
            %disp('Detected single row (possibly tilted). Sorting left to right.');
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
            %disp('Detected two rows. Sorting top to bottom, left to right.');
        end
    elseif Ne == 2
        % For 2 characters, check vertical separation
        y_diff = abs(centroids(1, 2) - centroids(2, 2));
        if y_diff > 30  % Adjust threshold based on plate size
            % Two rows
            [~, sorted_idx] = sort(centroids(:, 2));  % Sort by y (top to bottom)
            %disp('Detected two rows (2 characters). Sorting top to bottom.');
        else
            % Single row
            [~, sorted_idx] = sort(centroids(:, 1));  % Sort by x (left to right)
            %disp('Detected single row (2 characters). Sorting left to right.');
        end
    end
    
    firstCharBBox = [];
    final_output = [];
    for n = 1:Ne
        % Process characters in sorted order
        char_idx = sorted_idx(n);
        [r, c] = find(L == char_idx);
        n1 = picture(min(r):max(r), min(c):max(c));  % Crop nth object
        n1 = imresize(n1, [42, 24]);  % Resize to match database size
        %figure;
        %imshow(n1);
        % Store bounding box for the first character in sorted order
        if n == 1
            firstCharBBox = [min(c), min(r), max(c) - min(c) + 1, max(r) - min(r) + 1];
        end

        ans = ocr(n1).Text;
        disp(ans)
        
        x = [];
        totalLetters = size(imgfile, 2);
    
        for k = 1:totalLetters
            y = corr2(imgfile{1, k}, n1);
            x = [x y];
        end
        
        if max(x) > 0.35
            z = find(x == max(x));
            out = cell2mat(imgfile(2, z));
            final_output = [final_output out];
        end
    end
    
    np = final_output;
    %disp('Recognized characters:');
    disp(np);
end