function best = mirror_symmetry(filename, nang, shear, invert)
    
    verbose = true;
    if ~exist('nang')
        nang = 180;
    end
    if ~exist('shear')
        shear = false;
    end
    if ~exist('invert')
        invert = false;
    end
    
    img = imread(filename);
    if size(img,3) > 1
        img = rgb2gray(img);
    end
    if ~islogical(img)
        bw = img >= 128;
    else
        bw = img;
    end
    if invert
        bw = ~bw;
    end 
    
    bw = bw(any(bw,2), any(bw,1));
    [Y,X] = find(bw);
    B = bwboundaries(bw, 'noholes');
    Yb = B{1}(:,1);
    Xb = B{1}(:,2);
    as = pi * (0:nang-1)/nang;
    R = radon(bw, -180*(0:nang-1)/nang);
    
    tic;
    global num_flips;
    num_flips = 0;
    [xc,yc,theta,phi] = axis_of_inertia(X,Y,shear);
    best = struct('val', -1, 'rad', 0, 'ang', 0, 'dev', 0);
    for i = 1:length(theta)
        d = xc * cos(theta(i)) + yc * sin(theta(i));
        bw1 = rotate_by_params(bw, d, theta(i), phi(i));
        val = sum(bw1 & bw, 'all') / sum(bw, 'all');
        if val > best.val
            best = struct('val', val, 'rad', d, 'ang', theta(i), 'dev', phi(i));
        end
    end
    
    collect_bounds = false;
    if collect_bounds
        bw0 = cell(length(as), 3);
        U = [];
        dx = zeros(length(as), 2);
        for i = 1:length(as)
            [ub, dx(i,:)] = possible_axis(bw, R, as(i), best);
            U = [U [ub; i*ones(1,size(ub,2))]];
        end
        [~,idx] = sort(U(1,:), 'descend');
        U = U(:,idx);
        
        i = 1;
        while i <= size(U,2) && U(1,i) > best.val
            fprintf('Processing axis %d of %d\n', i, size(U,2));
            pos = U(3,i);
            ang = as(pos);
            T = [cos(ang) sin(ang) 0; -sin(ang) cos(ang) 0; 0 0 1]';
            B = [Xb Yb]*T(1:2, 1:2);
            ymin = floor(min(B(:,2)));
            ymax = ceil(max(B(:,2)));
            if isempty(bw0{pos,1})
                bw0{pos,1} = imwarp(bw, affine2d(T), 'OutputView', ...
                    imref2d([ymax-ymin+1, dx(pos,2)], dx(pos,1)+0.5+[0 dx(pos,2)], [ymin-0.5 ymax+0.5]));
                bw0{pos,2} = sum(bw0{pos,1}, 'all');
                G = bw0{pos,1}' * bw0{pos,1};
                I = (0:size(G,1)-1) + (1:size(G,1))';
                bw0{pos,3} = accumarray(I(:), G(:)) / bw0{pos,2};
            end
            if ~shear
                v = bw0{pos,3}(U(2,i));
                if v > best.val
                    best = struct('val', v, 'rad', dx(pos,1) + (1 + U(2,i)) / 2, 'ang', ang, 'dev', 0);
                end
            end
            i = i + 1;
        end
    else
        used_angles = false(1, length(as));
        while any(~used_angles)
            [~,i] = min( min(abs(as - best.ang), abs(min(as, best.ang) + pi - max(as, best.ang))) + 1000*used_angles);
            if verbose
                fprintf('Processing angle %d of %d\n', sum(used_angles)+1, length(as));
            end
            best = intersections_by_angle(bw, Xb, Yb, R, as(i), best, shear);
            used_angles(i) = true;
        end
    end
    elapsed = toc;
    
    if verbose
        bw1 = rotate_by_params(bw, best.rad, best.ang, best.dev);
        val = sum(bw & bw1, 'all') / sum(bw, 'all');
        fprintf('Intersection rate is %f, elapsed time is %f sec, number of flips done is %d\n', val, elapsed, num_flips);
        h = figure;
        imshow(255*cat(3, ~bw, ~bw & ~bw1, ~bw1));
        hold on;
        x0 = best.rad*cos(best.ang);
        y0 = best.rad*sin(best.ang);
        line(x0 + size(bw,1)*sin(best.ang)*[-2 2], y0 - size(bw,1)*cos(best.ang)*[-2 2], 'color', 'g', 'LineWidth', 1.5);
        if shear
            balt = best.ang - pi/2 + best.dev;
            line(xc + size(bw,1)*sin(balt)*[-2 2], yc - size(bw,1)*cos(balt)*[-2 2], 'color', 'm', 'LineWidth', 1.5);
        end
    end
end

function bw1 = rotate_by_params(bw, rad, ang, dev)
    x0 = rad*cos(ang);
    y0 = rad*sin(ang);
    T = [cos(ang + pi + dev) sin(ang + pi + dev);
         cos(ang + pi/2) sin(ang + pi/2)];
    A1 = [1 0 -x0; 0 1 -y0; 0 0 1]';
    A2 = inv([T' [0; 0]; 0 0 1])';
    A3 = [-1 0 0; 0 1 0; 0 0 1]';
    A4 = [T' [0; 0]; 0 0 1]';
    A5 = [1 0 x0; 0 1 y0; 0 0 1]';
    A = A1*A2*A3*A4*A5;
    bw1 = imwarp(bw, affine2d(A), 'OutputView', imref2d(size(bw)));
end

function [ub, dx] = possible_axis(bw, R, ang, best)
    idx = round(ang * size(R,2) / pi) + 1;
    xc = floor(size(bw,2) / 2);
    yc = floor(size(bw,1) / 2);
    
    xproj = R(:, idx)';
    dx = -(size(R,1)+1)/2 + (xc*cos(ang) + yc*sin(ang));
    first = find(xproj, 1, 'first');
    last = find(xproj, 1, 'last');
    xproj = xproj(first:last);
    dx = dx + first - 1;
    area = sum(xproj);
    
    ub = [0 reshape(repmat(cumsum(xproj(1:end-1)),[2 1]), 1, [])];
    ub(1:2:end) = xproj + 2 * min(ub(1:2:end), area-xproj-ub(1:2:end));
    ub(2:2:end) = 2 * min(ub(2:2:end), area-ub(2:2:end));
    ub = ub / area;
    idx = find(ub > best.val);
    for i = idx
        if i <= length(xproj)
            ub(i) = sum(min(xproj(1:i), xproj(i:-1:1)));
        else
            first = i + 1 - length(xproj);
            ub(i) = sum(min(xproj(first:end), xproj(end:-1:first)));
        end
    end
    ub = ub(idx) / area;
    mask = ub > best.val;
    ub = [ub(mask); idx(mask)];
    dx = [dx length(xproj)];
end


function best = intersections_by_angle(bw, Xb, Yb, R, ang, best, shear)
    
    idx = round(ang * size(R,2) / pi) + 1;
    xc = floor(size(bw,2) / 2);
    yc = floor(size(bw,1) / 2);
    
    xproj = R(:, idx)';
    dx = -(size(R,1)+1)/2 + (xc*cos(ang) + yc*sin(ang));
    first = find(xproj, 1, 'first');
    last = find(xproj, 1, 'last');
    xproj = xproj(first:last);
    dx = dx + first - 1;
    area = sum(xproj);
    
    ub = [0 reshape(repmat(cumsum(xproj(1:end-1)),[2 1]), 1, [])];
    ub(1:2:end) = xproj + 2 * min(ub(1:2:end), area-xproj-ub(1:2:end));
    ub(2:2:end) = 2 * min(ub(2:2:end), area-ub(2:2:end));
    ub = ub / area;
    idx = find(ub > best.val);
    for i = idx
        if i <= length(xproj)
            ub(i) = sum(min(xproj(1:i), xproj(i:-1:1)));
        else
            first = i + 1 - length(xproj);
            ub(i) = sum(min(xproj(first:end), xproj(end:-1:first)));
        end
    end
    ub(idx) = ub(idx) / area;
    rate = sum(ub(idx) > best.val) / length(ub);
    
    if rate == 0
        return;
    end
    
    T = [cos(ang) sin(ang) 0; -sin(ang) cos(ang) 0; 0 0 1]';
    B = [Xb Yb]*T(1:2, 1:2);
    ymin = floor(min(B(:,2)));
    ymax = ceil(max(B(:,2)));
    bw0 = imwarp(bw, affine2d(T), 'OutputView', ...
          imref2d([ymax-ymin+1, length(xproj)], dx+0.5+[0 length(xproj)], [ymin-0.5 ymax+0.5]));
    area = sum(bw0, 'all');
    
    if ~shear
        G = bw0' * bw0;
        I = (0:size(G,1)-1) + (1:size(G,1))';
        S = accumarray(I(:), G(:));
        [v, idx] = max(S);
        v = v / area;
        if v > best.val
            best = struct('val', v, 'rad', dx + (1 + idx) / 2, 'ang', ang, 'dev', 0);
        end
    else
        ub = [ub 0];
        [~,order] = sort(ub(idx), 'descend');
        idx = idx(order);
        
        [Y,X] = find([false(1,size(bw0,2)); bw0] ~= [bw0; false(1,size(bw0,2))]);
        X = X(1:2:end);
        Y = reshape(Y,2,[])';
        [I,J] = meshgrid(1:length(X), 1:length(X));
        mask = (X(I) < X(J) & (ub(X(I) + X(J))) > best.val);
        I = I(mask);
        J = J(mask);
        S = X(I) + X(J);
        L = min(Y(J,2)-Y(J,1), Y(I,2)-Y(I,1));
        P = sort([Y(I,1) - Y(J,:), Y(I,2) - Y(J,:)], 2) ./ (X(J) - X(I));
        Q = [ L./(P(:,2)-P(:,1)) -L./(P(:,2)-P(:,1)), ...
             -L./(P(:,4)-P(:,3))  L./(P(:,4)-P(:,3))];
        
        i = 1;
        while i <= length(idx) && ub(idx(i)) > best.val
            mask = (S == idx(i)+1);
            T = P(mask,:);
            dY = Q(mask,:);
            [T,order] = sort(T(:));
            dY = cumsum(dY(order));
            Y = (T(2:end)-T(1:end-1)) .* dY(1:end-1);
            Y = cumsum(Y);
            [v, sub] = max(Y);
            if mod(idx(i),2)
                pos = fix(idx(i)/2);
                v = (2*v + xproj(pos+1)) / area;
            else
                v = 2*v / area;
            end
            if v > best.val
                best = struct('val', v, 'rad', dx + (1 + idx(i)) / 2, ...
                              'ang', ang, 'dev', -atan(T(sub+1)));
            end
            i = i+1;
        end
    end
end

function val = functional(q20, q11, q02, q30, q21, q12, q03)
    val = abs(q30) + abs(q03) + abs(q11).^1.5;
end

function [xc,yc,theta,phi] = axis_of_inertia(X,Y,shear)
    
    m00 = length(X);
    xc = mean(X);
    yc = mean(Y);
    m20 = sum(X.*X);
    m11 = sum(X.*Y);
    m02 = sum(Y.*Y);
    
    if ~shear
        h20 = m20/m00 - xc^2;
        h02 = m02/m00 - yc^2;
        h11 = m11/m00 - xc*yc;
        theta = atan2(2*h11, h20 - h02) / 2;
        theta = [theta theta + pi/2];
        phi = [0 0];
        return
    end
    
    m10 = sum(X);
    m01 = sum(Y);
    m30 = sum(X.^3);
    m21 = sum(X.^2 .* Y);
    m12 = sum(X .* Y.^2);
    m03 = sum(Y.^3);
    nang = 180;
    ndev = 90;
    best = struct('val', Inf, 'theta', 0, 'phi', 0);
    
    for i = 0:nang-1
        alpha = pi*i/nang;
        a11 = cos(alpha);
        a12 = sin(alpha);
        p00 = m00;
        p10 = m10*a11 + m01*a12;
        p20 = m20*a11^2 + 2*m11*a11*a12 + m02*a12^2;
        p30 = m30*a11^3 + 3*m21*a11^2*a12 + 3*m12*a11*a12^2 + m03*a12^3;
        
        q20 = p20 - p10^2/p00;
        q30 = p30 - 3*p20*p10/p00 + 2*p10^3/p00^2;
        
        for j = (-ndev+1):(ndev-1)
            beta = (pi/2)*j/ndev;
            k = tan(beta);
            a21 = -sin(alpha) - k*cos(alpha);
            a22 = +cos(alpha) - k*sin(alpha);
            
            p01 = m10*a21 + m01*a22;
            p11 = m20*a11*a21 + m11*(a11*a22 + a21*a12) + m02*a12*a22;
            p02 = m20*a21^2 + 2*m11*a21*a22 + m02*a22^2;
            p21 = m30*a11^2*a21 + m21*a11^2*a22 + 2*m21*a11*a12*a21 + 2*m12*a11*a12*a22 + m12*a12^2*a21 + m03*a12^2*a22;
            p12 = m30*a21^2*a11 + m21*a21^2*a12 + 2*m21*a21*a22*a11 + 2*m12*a21*a22*a12 + m12*a22^2*a11 + m03*a22^2*a12;
            p03 = m30*a21^3 + 3*m21*a21^2*a22 + 3*m12*a21*a22^2 + m03*a22^3;
            
            q11 = p11 - p10*p01/p00;
            q02 = p02 - p01^2/p00;
            q21 = p21 - p20*p01/p00 - 2*p11*p10/p00 + 2*p10^2*p01/p00^2;
            q12 = p12 - p02*p10/p00 - 2*p11*p01/p00 + 2*p01^2*p10/p00^2;
            q03 = p03 - 3*p02*p01/p00 + 2*p01^3/p00^2;
            
            val = functional(q20, q11, q02, q30, q21, q12, q03);
            if val < best.val
                best.val = val;
                best.theta = alpha;
                best.phi = beta;
            end
            
        end
    end
    theta = [best.theta best.theta-pi/2+best.phi];
    phi = [best.phi -best.phi];
    
end
