matrixdata = zeros(1200,1200);

for i = 1:40
    for j =1:30
        locx = (i-1)*30+1;
        locy = (j-1)*40+1;
        matrixdata(locx:locx+29,locy:locy+39) = res(:,:,(i-1)*30+j);
%         disp(['(' num2str(i) ', ' num2str(j) ')']);
    end
end