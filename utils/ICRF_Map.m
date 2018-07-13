%//////////////////////////////////////////////////////////////////////////
%                             ICRF_Map
%function: transfer irradiance L to image x according to CRF function
%If this code is helpful to you, please Cite: https://arxiv.org/abs/1807.04686
%//////////////////////////////////////////////////////////////////////////
%inputs:
%--------Img: input irradiance L, single or double image tyoe, [0,1]
%--------I,B: inverse CRF response lookup table
%outputs:
%--------Img: image x
%//////////////////////////////////////////////////////////////////////////
function [Img] = ICRF_Map(Img,I,B)
   [w,h,c] = size(Img);
   bin = size(I,2);
   Size = w*h*c;
   tiny_bin = 9.7656e-04;
   min_tiny_bin = 0.0039; 
   for i = 1:Size
       temp = Img(i);
       start_bin = 1;
       if temp > min_tiny_bin
           start_bin = floor(temp/tiny_bin - 1);
       end
       for b = start_bin: bin
           tempB = B(1,b);
           if tempB >= temp
               index = b;
               if index > 1
                   comp1 = tempB - temp;
                   comp2 = temp - B(1,index-1);
                   if comp2 < comp1
                       index = index-1;
                   end
               end
               Img(i) = I(1,index);
               break;
           end
       end
   end
end