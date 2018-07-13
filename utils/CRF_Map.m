%//////////////////////////////////////////////////////////////////////////
%                             CRF_Map
%function: transfer image x to irradiance L according to CRF function
%If this code is helpful to you, please Cite: https://arxiv.org/abs/1807.04686
%//////////////////////////////////////////////////////////////////////////
%inputs:
%--------Img: input image, single or double image tyoe, [0,1]
%--------I,B: CRF response lookup table
%outputs:
%--------Img: irradiance L
%//////////////////////////////////////////////////////////////////////////
function [Img] = CRF_Map(Img,I,B)
   [w,h,c] = size(Img);
   bin = size(I,2);
   tiny_bin = 9.7656e-04;
   min_tiny_bin = 0.0039;
   Size = w*h*c;
   for i = 1:Size
      temp = Img(i);
      if (temp < 0)
          temp = 0;
          Img(i) = 0;
      end
      if (temp > 1)
          temp = 1;
          Img(i) = 1;
      end
      start_bin = 1;
      if temp > min_tiny_bin
          start_bin = floor(temp/tiny_bin - 1);
      end
      for b = start_bin: bin
          tempB = I(1,b);
          if tempB >= temp
              index = b;
              if index > 1
                  comp1 = tempB - temp;
                  comp2 = temp - B(1,index-1);
                  if comp2 < comp1
                      index = index-1;
                  end
              end
              Img(i) = B(1,index);
              break;
          end
      end
   end      
end