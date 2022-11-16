function [best_val,best_x,best_y] = rotational_symmetry(img, deg, k)
    % img - input image
    % deg - degree of symmetry
    % k - number of subdivisions
    if ~exist('k')
        k = 1;
    end
    if size(img,3) > 1
        img = rgb2gray(img);
    end
    if ~islogical(img)
        img = img >= 128;
    end
    
    [Y,X] = find(img);
    xc = mean(X);
    yc = mean(Y);
    d = sqrt(max((X-xc).^2 + (Y-yc).^2));
    xmin = floor(xc - d);
    xmax = ceil(xc + d);
    ymin = floor(yc - d);
    ymax = ceil(yc + d);
    
    if xmin >= 1
        img = img(:, xmin:end);
    else
        img = [false(size(img,1), 1-xmin) img];
    end
    if size(img,2) >= xmax-xmin+1
        img = img(:, 1:xmax-xmin+1);
    else
        img = [img false(size(img,1), xmax-xmin+1-size(img,2))];
    end    
    
    if ymin >= 1
        img = img(ymin:end, :);
    else
        img = [false(1-ymin, size(img,2)); img];
    end
    if size(img,1) >= ymax-ymin+1
        img = img(1:ymax-ymin+1, :);
    else
        img = [img; false(ymax-ymin+1-size(img,1), size(img,2))];
    end  
    
    X = X - (xmin-1);
    Y = Y - (ymin-1);
    xc = xc - (xmin-1);
    yc = yc - (ymin-1);
    
    I = cell(1, deg);
    P = cell(1, deg);
    shift = cell(1, deg);
    I{1} = img;
    P{1} = sum(I{1}, 1) / sum(I{1}(:));
    shift{1} = [0 0];
    for i = 2:deg
        alpha = (i-1)/deg * 2*pi;
        %A1 = [1 0 -xc; 0 1 -yc; 0 0 1];
        %A2 = [cos(alpha) sin(alpha) 0; -sin(alpha) cos(alpha) 0; 0 0 1];
        %A3 = [1 0 +xc; 0 1 +yc; 0 0 1];
        %A = A3*A2*A1;
        %A * [x y t]' = [ cos(alpha)*(x-xc) + sin(alpha)*(y-yc) + xc;
        %                -sin(alpha)*(x-xc) + cos(alpha)*(y-yc) + yc;
        %                 0 0 1];
        A = [ cos(alpha) sin(alpha) xc*(1-cos(alpha)) - yc*sin(alpha); ...
             -sin(alpha) cos(alpha) yc*(1-cos(alpha)) + xc*sin(alpha); ...
              0 0 1];
        shift{i} = A(1:2,3)';
        I{i} = imwarp(I{1}, affine2d(A'), 'OutputView', imref2d(size(img)));
        P{i} = sum(I{i}, 1) / sum(I{i}(:));
    end
    
    tic;
    %[best_val, best_x, best_y] = jaccard_shifts(I, xc, yc, false);
    [best_val, best_x, best_y] = jaccard_projections(I, P, shift, xc, yc, true, k);
    fprintf('Center (%f, %f), value %f\n', best_x, best_y, best_val);
    toc;
    
    if deg == 2
        %A1 = [1 0 -best_x; 0 1 -best_y; 0 0 1];
        %A2 = [-1 0 0; 0 -1 0; 0 0 1];
        %A3 = [1 0 best_x; 0 1 best_y; 0 0 1];
        %A = A3*A2*A1;
        A = [-1 0 2*best_x; 0 -1 2*best_y; 0 0 1];
        J = imwarp(I{1}, affine2d(A'), 'OutputView', imref2d(size(I{1})));
        imshow(255*cat(3, ~I{1}, ~I{1} & ~J, ~J));
        hold on;
        scatter(best_x, best_y, 'MarkerEdgeColor', 'g');
    else
        d = zeros(deg,2);
        for i = 2:deg
            alpha = (i-1)/deg * 2*pi;
            A1 = [1 0 -best_x; 0 1 -best_y; 0 0 1];
            A2 = [cos(alpha) sin(alpha) 0; -sin(alpha) cos(alpha) 0; 0 0 1];
            A3 = [1 0 +best_x; 0 1 +best_y; 0 0 1];
            A = A3*A2*A1;
            d(i,:) = A(1:2, 3)' - shift{i};
        end
        left = floor(min(d(:,1)));
        right = ceil(max(d(:,1)));
        up = floor(min(d(:,2)));
        down = ceil(max(d(:,2)));
        d = round(d - [left up]);
        sz = [size(I{1},1) - left + right, size(I{1},2) - up + down];
        C = zeros([sz deg]);
        for i = 1:deg
            C(d(i,2)+(1:size(I{i},1)), d(i,1)+(1:size(I{i},2)), i) = I{i};
        end
        imshow(1 - mean(C,3));
        hold on;
        scatter(d(1,1) + best_x, d(1,2) + best_y, 'MarkerEdgeColor', 'g');
    end
end

function [best_val, best_x, best_y] = jaccard_shifts(I, xc, yc, use_stripes)
    
    deg = length(I);
    
    if use_stripes
        L = cell(1, deg);
        for i = 1:deg
            T = I{i} >= 0.5;
            [ys, xs] = find([false(1,size(T,2)); T] ~= [T; false(1,size(T,2))]);
            L{i} = [xs(1:2:end) ys(1:2:end) ys(2:2:end)];
        end
    end
    
    [Xt,Yt] = meshgrid(1:size(I{1},2), 1:size(I{1},1));
    H = zeros(size(Xt));
    for i = 1:deg
        alpha = (i-1)/deg * 2*pi;
        for j = i+1:deg
            beta = (j-1)/deg * 2*pi;
            if use_stripes
                dX = L{i}(:,1)' - L{j}(:,1);
                dX = repmat(dX, [1 1 4]);
                dY = cat(3, L{i}(:,2)' - L{j}(:,2), L{i}(:,2)' - L{j}(:,3), ...
                            L{i}(:,3)' - L{j}(:,2), L{i}(:,3)' - L{j}(:,3));
                dY = sort(dY, 3);
                D = min((L{i}(:,3) - L{i}(:,2))', L{j}(:,3) - L{j}(:,2));
                dV = cat(3, +D ./ (dY(:,:,2)-dY(:,:,1)), ...
                            -D ./ (dY(:,:,2)-dY(:,:,1)), ...
                            -D ./ (dY(:,:,4)-dY(:,:,3)), ...
                            +D ./ (dY(:,:,4)-dY(:,:,3)));
                xmin = min(dX(:));
                xmax = max(dX(:));
                ymin = min(dY(:));
                ymax = max(dY(:));
            else
                xmin = find(any(I{i},1), 1, 'first') - find(any(I{j},1), 1, 'last' );
                xmax = find(any(I{i},1), 1, 'last' ) - find(any(I{j},1), 1, 'first');
                ymin = find(any(I{i},2), 1, 'first') - find(any(I{j},2), 1, 'last' );
                ymax = find(any(I{i},2), 1, 'last' ) - find(any(I{j},2), 1, 'first');
            end
            
            [X,Y] = meshgrid(xmin:xmax, ymin:ymax);
            V = zeros(size(X));
            for x = xmin:xmax
                fprintf('Processing %d of %d\n', x-xmin+1, xmax-xmin+1);
                if ~use_stripes
                    for y = ymin:ymax
                        if x < 0
                            if y < 0
                                V(y-ymin+1, x-xmin+1) = ...
                                    sum(min(I{i}(1:end+y, 1:end+x), I{j}(-y+1:end, -x+1:end)), 'all');
                            else
                                V(y-ymin+1, x-xmin+1) = ...
                                    sum(min(I{i}(y+1:end, 1:end+x), I{j}(1:end-y, -x+1:end)), 'all');
                            end
                        else
                            if y < 0
                                V(y-ymin+1, x-xmin+1) = ...
                                    sum(min(I{i}(1:end+y, x+1:end), I{j}(-y+1:end, 1:end-x)), 'all');
                            else
                                V(y-ymin+1, x-xmin+1) = ...
                                    sum(min(I{i}(y+1:end, x+1:end), I{j}(1:end-y, 1:end-x)), 'all');
                            end
                        end
                    end
                else
                    mask = dX == x;
                    dy = dY(mask);
                    dv = dV(mask);
                    [~, order] = sort(dy);
                    dy = dy(order);
                    dv = dv(order);
                    dv = cumsum(dv);
                    dv = dv(1:end-1) .* (dy(2:end) - dy(1:end-1));
                    dv = [0; cumsum(dv)];

                    pos = find(dy ~= [dy(2:end); +Inf]);
                    dy = dy(pos);
                    dv = dv(pos);
                    yp = ceil(dy(1)):floor(dy(end));
                    V(yp-ymin+1, x-xmin+1) = interp1(dy, dv, yp);
                end
            end
            V(isnan(V)) = 0;
            V = V / sum(I{1},'all');
            % A = [cos(alpha) -sin(alpha) xc*(1-cos(alpha)) + yc*sin(alpha); ...
            %      sin(alpha)  cos(alpha) yc*(1-cos(alpha)) - xc*sin(alpha); ...
            %      0 0 1];
            dx = (Xt - xc)*(cos(alpha)-cos(beta)) + (Yt - yc)*(sin(beta) - sin(alpha));
            dy = (Yt - yc)*(cos(alpha)-cos(beta)) + (Xt - xc)*(sin(alpha) - sin(beta));
            add = interp2(X,Y,V,dx,dy);
            add(isnan(add)) = 0;
            %{
            T = zeros(size(I{1}));
            for x = 1:size(T,2)
                fprintf('Processing %d of %d\n', x, size(I{1},2));
                if true
                    for y = 1:size(T,1)
                        A1 = [1 0 -x; 0 1 -y; 0 0 1];
                        A2 = [cos(beta) -sin(beta) 0; sin(beta) cos(beta) 0; 0 0 1];
                        A3 = [1 0 +x; 0 1 +y; 0 0 1];
                        A = A3*A2*A1;
                        B = imwarp(I{1}, affine2d(A'), 'OutputView', imref2d(size(I{1})));
                        T(y,x) = sum(min(I{1},B), 'all'); 
                    end
                end
            end
            %}
            H = H + add;
        end
    end
    H = H / (deg*(deg-1)/2);
    [best_val,idx] = max(H(:));
    [best_y,best_x] = ind2sub(size(H), idx);
end

function [best_val, best_x, best_y] = jaccard_projections(I, P, shift, xc, yc, use_radon, num_stages)
    area = sum(I{1},'all');
    best_val = rotational_jaccard(I, area, xc, yc, shift);
    best_x = xc;
    best_y = yc;
    deg = length(I);
    
    if use_radon
        if mod(deg,2) == 1 && mod(num_stages,2) == 0
            num_stages = num_stages * 2;
        end
        ang_num = deg * num_stages;
        if mod(ang_num,2) == 0
            ang_num = ang_num/2;
        end      
        R = radon(I{1}, -(0:ang_num-1)/ang_num * 180);
        R = [R R(end:-1:1,:)];
        R = R / sum(I{1}, 'all');
        
        step = size(R,2) / deg;
        substep = step / num_stages;
        scheme = cell(fix(deg/2), 2);
        for i = 1:length(scheme)
            if mod(deg * num_stages, 2) == 1
                t = deg * num_stages;
            else
                t = deg * num_stages / 2;
            end
            scheme{i,1} = mod((0:t-1)'*substep + [0, i*step], size(R,2));
            scheme{i,2} = (1 + (2*i ~= deg)) * (deg/2);
        end
        J = (1:size(R,1)) - (1:size(R,1))';
    else
        J = (1:length(P{1})) - (1:length(P{1}))';
    end

    [Xt,Yt] = meshgrid(1:size(I{1},2), 1:size(I{1},1));
    H = zeros(size(Xt));
    
    if ~use_radon
        for i = 1:deg
            alpha = (i-1)/deg * 2*pi;
            for j = 1:deg-1
                beta = (j-1)/deg * 2*pi;
                G = min(P{i}, P{j}');
                S = accumarray(J(:) - min(J(:))+1, G(:));
                % around (xc,yc) by alpha: xc*(1-cos(alpha)) - yc*sin(alpha)
                % around (xt,yt) by alpha: xt*(1-cos(alpha)) - yt*sin(alpha)
                % shift from (xc,yc) to (xt,yt): (xc-xt)*cos(alpha) + (yc-yt)*sin(alpha) + (xt-xc)
                % shift from beta to alpha: (xt-xc)*(cos(beta)-cos(alpha)) + (yt-yc)*(sin(beta)-sin(alpha))
                dx = (Xt - xc)*(cos(beta)-cos(alpha)) + (Yt - yc)*(sin(beta) - sin(alpha));
                add = interp1(min(J(:)):max(J(:)), S, dx);
                add(isnan(add)) = 0;
                H = H + add;
            end
        end
    else
        x0 = floor((size(I{1}, 2) + 1) / 2);
        y0 = floor((size(I{1}, 1) + 1) / 2);
        for i = 1:size(scheme,1)
            add = ones(size(Xt));
            delta = (scheme{i,1}(1,2) - scheme{i,1}(1,1)) / size(R,2) * 2*pi;
            for j = 1:size(scheme{i,1})
                alpha = scheme{i,1}(j,1)/size(R,2) * 2*pi;
                beta  = scheme{i,1}(j,2)/size(R,2) * 2*pi;
                idx_alpha = scheme{i,1}(j,1) + 1;
                idx_beta  = scheme{i,1}(j,2) + 1;
                % around (x0,y0) by alpha+delta: x0*(1-cos(alpha+delta)) - y0*sin(alpha+delta)
                % around (xt,yt) by alpha, then by delta: (xt*(1-cos(alpha)) - yt*sin(alpha) - x0)*cos(delta) + (yt*(1-cos(alpha)) + xt*sin(alpha) - y0)*sin(delta)
                % around (x0,y0) by beta +delta: x0*(1-cos(beta +delta)) - y0*sin(beta +delta)
                % around (xt,yt) by beta , then by delta: (xt*(1-cos(beta )) - yt*sin(beta ) - x0)*cos(delta) + (yt*(1-cos(beta )) + xt*sin(beta ) - y0)*sin(delta)
                G = min(R(:, idx_alpha)', R(:, idx_beta));
                G = G / sum(R(:,1));
                S = accumarray(J(:) - min(J(:))+1, G(:));
                %{
                dx1 = -x0*cos(alpha+delta) + -y0*sin(alpha+delta);
                ax1 = (Xt*(1-cos(alpha)) - Yt*sin(alpha))*cos(delta) + (Yt*(1-cos(alpha)) + Xt*sin(alpha))*sin(delta);
                dx2 = -x0*cos(beta +delta) + -y0*sin(beta +delta);
                ax2 = (Xt*(1-cos(beta )) - Yt*sin(beta ))*cos(delta) + (Yt*(1-cos(beta )) + Xt*sin(beta ))*sin(delta);
                dx = (ax1 - dx1) - (ax2 - dx2);
                %}
                dx = (x0-Xt)*(cos(beta+delta) - cos(alpha+delta)) + ...
                     (y0-Yt)*(sin(beta+delta) - sin(alpha+delta));
                temp = interp1(J(end,1):J(1,end), S, dx);
                temp(isnan(temp)) = 0;
                add = min(add, temp);
            end
            %fprintf('(%d, %d, %f)\n', i, j, max(add,[],'all'));
            %imshow(add);
            H = H + scheme{i,2}*add;
        end
    end
    H = H / (deg*(deg-1)/2);
    
    idx = find(H > best_val);
    [upper, order] = sort(H(idx), 'descend');
    idx = idx(order);
    [Yt,Xt] = ind2sub(size(H), idx);
    
    %{
    dx = Xt*cos((0:k-1)/k * 2*pi) - Yt*sin((0:k-1)/k * 2*pi);
    dy = Yt*cos((0:k-1)/k * 2*pi) + Xt*sin((0:k-1)/k * 2*pi);
    dx = mean(abs(reshape(dx,[],1,6) - reshape(dx,1,[],6)), 3);
    dy = mean(abs(reshape(dy,[],1,6) - reshape(dy,1,[],6)), 3);
    d = max(sum(I{1}(:,2:end) ~= I{1}(:,1:end-1), 'all'), ...
            sum(I{1}(2:end,:) ~= I{1}(1:end-1,:), 'all')) / sum(I{1},'all');
    d = d * (dx + dy);
    %}
    
    i = 1;
    while i <= length(upper) && upper(i) > best_val
        fprintf('Checking stable point %d of %d\n', i, length(Xt));
        val = rotational_jaccard(I, area, Xt(i), Yt(i), shift);
        if val > best_val
            best_val = val;
            best_x = Xt(i);
            best_y = Yt(i);
        end
        i = i + 1;
    end

end

function [best_val, best_x, best_y] = jaccard_bruteforce(I, shift)
    best_val = -1;
    best_x = 0;
    best_y = 0;
    xs = 1:size(I{1}, 2);
    ys = 1:size(I{1}, 1);
    for x = xs
        for y = ys
            fprintf('(%d, %d) of (%d, %d)\n', find(xs == x), find(ys == y), length(xs), length(ys));
            val = rotational_jaccard(I, area, x, y, shift);
            if val > best_val
                best_val = val;
                best_x = x;
                best_y = y;
            end
        end
    end
end

function val = rotational_jaccard_old(I, area, x, y, shift)
    k = length(I);
    d = zeros(k,2);
    for i = 2:k
        alpha = (i-1)/k * 2*pi;
        A1 = [1 0 -x; 0 1 -y; 0 0 1];
        A2 = [cos(alpha) sin(alpha) 0; -sin(alpha) cos(alpha) 0; 0 0 1];
        A3 = [1 0 +x; 0 1 +y; 0 0 1];
        A = A3*A2*A1;
        d(i,:) = A(1:2, 3)' - shift{i};
    end

    left = floor(min(d(:,1)));
    right = ceil(max(d(:,1)));
    up = floor(min(d(:,2)));
    down = ceil(max(d(:,2)));
    d = round(d - [left up]);
    sz = [size(I{1},1) - left + right, size(I{1},2) - up + down];
    C = zeros([sz k]);
    for i = 1:k
        C(d(i,2)+(1:size(I{i},1)), d(i,1)+(1:size(I{i},2)), i) = I{i};
    end

    if true
        val = 0;
        for i = 1:k
            for j = i+1:k
                add = sum(min(C(:,:,i), C(:,:,j)), 'all') / area;
                val = val + add;
            end
        end
        val = val / (k*(k-1)/2);
    else
        C = round(sum(C,3));
        A = accumarray(C(:)+1, 1, [k+1 1]);
        val = sum(A' .* [0 0 arrayfun(@(x) nchoosek(x,2), 2:k)]);
        val = 2 * val / ((k-1) * sum(A' .* (0:k)));
    end
end

function val = rotational_jaccard(I, area, x, y, shift)
    k = length(I);
    area = sum(I{1}, 'all');
    val = 0;
    for i = 2:(fix(k/2)+1)
        alpha = (i-1)/k * 2*pi;
        A1 = [1 0 -x; 0 1 -y; 0 0 1];
        A2 = [cos(alpha) sin(alpha) 0; -sin(alpha) cos(alpha) 0; 0 0 1];
        A3 = [1 0 +x; 0 1 +y; 0 0 1];
        A = A3*A2*A1;
        dx = round(A(1,3) - shift{i}(1) + shift{1}(1));
        dy = round(A(2,3) - shift{i}(2) + shift{1}(2));
        
        if dx > 0
            x1 = dx;
            x2 = 0;
        else
            x1 = 0;
            x2 = -dx;
        end
        w = min(size(I{1},2)-x1, size(I{i},2)-x2);
        if dy > 0
            y1 = dy;
            y2 = 0;
        else
            y1 = 0;
            y2 = -dy;
        end
        h = min(size(I{1},1)-y1, size(I{i},1)-y2);
        add = sum(I{1}(y1+(1:h),x1+(1:w)) & I{i}(y2+(1:h),x2+(1:w)), 'all') / area;
        val = val + add * (1 + (2*i ~= k));
    end
    val = val/(k-1);
end