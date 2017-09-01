function tranmatrix = Untitled(data)
    distmatric = distanceMetric(48,64);
    tranmatrix = transferMatrix(distmatric, data, 48,64);
end


function distmatric = distanceMetric(h, w);
    a = [1:h];
    b = [1:w];
    idxa = repmat(a, [1,w]); 
    idxb = repelem(b, h);

    distmatric = zeros(w*h,w*h);
    factor_ = 0.15*w;
    for i=1:w*h
        for j=1:w*h
            if(i>=j)
                distX = (idxa(j)-idxa(i))^2;
                distY = (idxb(j)-idxb(i))^2;
                distmatric(i,j) = exp(-(distX+distY)/2/factor_/factor_);
                distmatric(j,i) = distmatric(i,j);
            end
        end

    end
end


function tranmatrix = transferMatrix(distmatric, data, h, w)
    tranmatrix = zeros(w*h,w*h);
    for i=1:w
        for j=1:h
            for p=1:w
                for q=1:h
                    if(i*j>=p*q)
                        tmp1 = data(i,j,:);
                        tmp2 = data(p,q,:);
                        tmp = tmp1-tmp2;
                        tranmatrix(i+j*w,p+q*w) = norm(tmp(:))+distmatric(i+j*w,p+q*w);
                        tranmatrix(p+q*w,i+j*w) = tranmatrix(p+q*w,i+j*w);
                    end
                end
            end
        end
    end

end