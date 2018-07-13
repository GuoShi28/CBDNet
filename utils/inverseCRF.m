%%% function: create inverse CRF lookup table for faster computation
%%%If this code is helpful to you, please Cite: https://arxiv.org/abs/1807.04686
load('201_CRF_data.mat')
invI = zeros(201,1024,'single');
invB = I;

for index = 1:201
    for bin = 1:1024
        temp = invB(index,bin);
        for bin_in = 1:1024
            tempB = B(index,bin_in);
             if tempB >= temp
                   xx = bin_in;
                   if bin_in > 1
                       comp1 = tempB - temp;
                       comp2 = temp - B(1,bin_in-1);
                       if comp2 < comp1
                           xx = bin_in-1;
                       end
                   end
                   invI(index,bin) = I(index,xx);
                   break;
               end
        end
    end
end
save('dorfCurvesInv.mat','invI','invB');