function project_Murali_Sandhya(fn)
    addpath( '../dataset' );
    %extracts tumor from the MRI image
    project_cv(fn);
end

function project_cv(fn)
%read image
im=imread(fn);
figure,imagesc(im);
colormap(gray);
pause(2);

%displays the size of image
disp(size(im));

%checks if image is an rgb image
if(size(im,3)>2)
im=rgb2gray(im);
imagesc(im);
colormap(gray);
pause(2);
end

%converts image to double
im_double=im2double(im);
%removes noise by applying median filter
im_med=medfilt2(im);

imshow(im_double);

% smoothing filter matrix
filtr=[1 1 1 1 1 1 1 ;
       0 0 0 0 0 0 0];
   
   %texture filter is applied to determine the texture image
  im_text=rangefilt(im_double);
  %applying smoothing filter on the texture image
  im_text=imfilter(im_text,filtr);
  imshow(im_text);
  
  %takes coordinates of tumor region
  [row,col]=ginput();
tumor_region=[row,col];
%determine the texture values of the tumor region
val_tumor=impixel(im_text,tumor_region(:,1),tumor_region(:,2));

disp('skull');
imshow(im_text);
%take coordinates of the skull region
[rols,cols]=ginput();
skull_region=[rols,cols];

%determine the texture values of the skull region
val_skull=impixel(im_double,skull_region(:,1),skull_region(:,2));
disp(val_skull(:,1));

%target variable is a vector which divides into two classes :0 represents
%skull region and 1 represents tumor region
target_variable=[zeros(numel(val_skull(:,1)),1); ones(numel(val_tumor(:,1)),1)];

%making the dimensions of the target variable and tumor region same
val_tumor=[val_tumor(:,1);zeros(length(target_variable) - length(val_tumor),1)];

%compute cross correlation
correlation=xcorr2(target_variable,val_tumor);
disp(max(correlation(:)));


%Otsu thresholding

%imtophat computes morphological opening using the structuring element as
%specified and subtracts the result from the original image.
im_thresh=imtophat(im_med,strel('disk',40));
imshow(im_thresh);
pause(2);
%improves the contrast of the image
im_adjust=imadjust(im_thresh);
imshow(im_adjust);
pause(2);
%determines the threshold value to perform segmentation
level=graythresh(im_adjust);
%segments the image into two classes: 0 if less than level and 1 if greater
%or equal to level
BW=imbinarize(im_adjust,level);
imshow(BW);
%performs morphological erosion
strel_erode=strel('disk',3);
im_erode=imerode(BW,strel_erode);
imshow(im_erode);
title('Otsu thresholding');
pause(5);

%performs normalized cross correlation 
norm_corr=normxcorr2(im_erode,im);
disp(max(norm_corr(:)));
pause(5);

%applying local thresholding

%lower threshold
t0=50;
%upper threshold
th= t0+((max(im_med(:))+min(im_med(:)))./2);
seg_img=zeros(size(im_med,1),size(im_med,2));
for i= 1:1:size(im_med,1)
    for j=1:1:size(im_med,2)
        if im_med (i,j)>th
            %sets to 1 if greater than threshold
            seg_img(i,j)=1;
        else
            %sets to 0 if lesser than threshold
            seg_img(i,j)=0;
        end
    end
end

imshow(seg_img);
pause(2);

%performs morphological erosion with disk as the structuring element
strel_erode=strel('disk',3);
im_erode=imerode(seg_img,strel_erode);
imshow(im_erode);
pause(2);

%performs morphological dilation with disk as the structuring element
str_dilate=strel('disk',3);
im_dilate=imdilate(im_erode,str_dilate);
imshow(im_dilate);
pause(2);

%performs morphological erosion with disk as the structuring element
im_erode1=imerode(im_dilate,strel_erode);
imshow(im_erode1);
title('Local Thresholding');
pause(5);


%performs normalized cross correlation 
norm_corr=normxcorr2(im_erode1,im);
disp(max(norm_corr(:)));
pause(5);

%watershed segmentation

%imtophat computes morphological opening using the structuring element as
%specified and subtracts the result from the original image.
im_thresh=imtophat(im_med,strel('disk',40));
imshow(im_thresh);
pause(2);
%improves the contrast of the image
im_adjust=imadjust(im_thresh);
imshow(im_adjust);
pause(2);
%determines the threshold value for Otsu's segmentation
level=graythresh(im_adjust);
%segments the image using Otsu's threshold
BW=imbinarize(im_adjust,level);
imshow(BW);
pause(2);
%performs morphological erosion with disk as a structuring element
strel_erode=strel('disk',3);
%performs erosion with a structuring element of shape disk
im_erode=imerode(BW,strel_erode);
imshow(im_erode);
pause(2);
%take the compliment of the result
comp=~im_erode;
imshow(comp);
pause(2);
%compute distance between every pixel to every non-zero pixel
dist=-bwdist(comp);
dist(comp)=-Inf;
%apply watershed segmentation to get the labelled image
label=watershed(dist);
%convert the image to rgb
img_final=label2rgb(label,'gray','w');
imshow(img_final);
title('Watershed segmentation');
pause(5);

%perform %performs normalized cross correlation 
norm_corr=normxcorr2(rgb2gray(img_final),im);
disp(max(norm_corr(:)));
pause(5);

%k-means clustering

%converts image to linear shape
img_reshape=reshape(im_med,[],1);
%apply k-means with k value as 4
[imgVecQ,imgVecC]=kmeans(double(img_reshape),4); 
%arranging back into image
img_res=reshape(imgVecQ,size(im_med)); 
imagesc(img_res);
pause(2);
figure,
subplot(3,2,1),imshow(img_res==1,[]);
subplot(3,2,2),imshow(img_res==2,[]);
subplot(3,2,3),imshow(img_res==3,[]);
subplot(3,2,4),imshow(img_res==4,[]);
pause(2);

%perform normalized cross-correlation for each cluster
norm_corr=normxcorr2(img_res==1,im);
disp(max(norm_corr(:)));
pause(2);


norm_corr=normxcorr2(img_res==2,im);
disp(max(norm_corr(:)));
pause(2);


norm_corr=normxcorr2(img_res==3,im);
disp(max(norm_corr(:)));
pause(2);


norm_corr=normxcorr2(img_res==4,im);
disp(max(norm_corr(:)));
pause(2);


%MSER

%determines MSER features for a area range between 200 to 5000 and a
%threshold value of 12
[mserRegions, mserConnComp] = detectMSERFeatures(im_med, ...
    'RegionAreaRange',[200 5000],'ThresholdDelta',12);

figure,
imshow(im_med)
hold on
%plot the regions obtained for the features extracted
plot(mserRegions, 'showPixelList', true,'showEllipses',false)
title('MSER regions')
pause(2);

hold off

end
