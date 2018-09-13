img_org=imread('gray_img.png');
img1=imread('rst_std30.png');
img2=imread('rst_std100.png');
img3=imread('rst_std150.png');

sub1=double(img1)-double(img_org);
sub2=double(img2)-double(img_org);
sub3=double(img3)-double(img_org);

%차영상 평균과 표준편차
mean_1 = mean2(sub1);
std_1 = std2(sub1);

mean_2 = mean2(sub2);
std_2 = std2(sub2);

mean_3 = mean2(sub3);
std_3 = std2(sub3);

%차영상 시각화를 위한 normalization
max1 = max(im2col(sub1,size(sub1)));
min1 = min(im2col(sub1,size(sub1)));
range1 = max1 - min1;
ssub1 = uint8(255 * (sub1 - min1)/range1);

max2 = max(im2col(sub2,size(sub2)));
min2 = min(im2col(sub2,size(sub2)));
range2 = max2 - min2;
ssub2 = uint8(255 * (sub2 - min2)/range2);

max3 = max(im2col(sub3,size(sub3)));
min3 = min(im2col(sub3,size(sub3)));
range3 = max3 - min3;
ssub3 = uint8(255 * (sub3 - min3)/range3);

figure(1);
imshow(ssub1)

figure(2);
imshow(ssub2)

figure(3);
imshow(ssub3)

imwrite(ssub1,'rst_std30_sub.png')
imwrite(ssub2,'rst_std100_sub.png')
imwrite(ssub3,'rst_std150_sub.png')

%원본을 모를때 평탄한 로컬 영역의 노이즈를 보고 추정
rect_wall=[357 2 155 161];%중앙 벽 부분
rect_tile=[43 13 38 40];%왼쪽 체크무늬 부분

wall_img1=imcrop(img1,rect_wall);
tile_img1=imcrop(img1,rect_tile);
wall_mean1=mean2(wall_img1);
wall_std1=std2(wall_img1);
tile_mean1=mean2(tile_img1);
tile_std1=std2(tile_img1);

wall_img2=imcrop(img2,rect_wall);
tile_img2=imcrop(img2,rect_tile);
wall_mean2=mean2(wall_img2);
wall_std2=std2(wall_img2);
tile_mean2=mean2(tile_img2);
tile_std2=std2(tile_img2);

wall_img3=imcrop(img3,rect_wall);
tile_img3=imcrop(img3,rect_tile);
wall_mean3=mean2(wall_img3);
wall_std3=std2(wall_img3);
tile_mean3=mean2(tile_img3);
tile_std3=std2(tile_img3);

wall_size = 155*161/(555*695);
tile_size = 38*40/(555*695);

%시각화 파트(normalization)
wall_simg1=imcrop(img1,rect_wall);
tile_simg1=imcrop(img1,rect_tile);

wall_simg2=imcrop(img2,rect_wall);
tile_simg2=imcrop(img2,rect_tile);

wall_simg3=imcrop(img3,rect_wall);
tile_simg3=imcrop(img3,rect_tile);

imwrite(wall_simg1,'rst_std30_img_wall.png')
imwrite(tile_simg1,'rst_std30_img_tile.png')

imwrite(wall_simg2,'rst_std100_img_wall.png')
imwrite(tile_simg2,'rst_std100_img_tile.png')

imwrite(wall_simg3,'rst_std150_img_wall.png')
imwrite(tile_simg3,'rst_std150_img_tile.png')
