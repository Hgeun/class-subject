%img = imread('left1.png');
%img = imresize(img, 0.5);
%img = rgb2gray(img);
img = imread('gray_img.png');

figure(1);%처음 흑백 이미지
imshow(img)
%imwrite(img,'gray_img.png');

%Box-Muller transform을 이용한 gaussian noise 생성
rst = uint8(zeros(size(img)));
noise = zeros(size(img));
std = 150; % 표준 편차 

if floor(size(img,1)/2)==size(img,1)/2
    for i = 1 : floor(size(img,1)/2)
        for j = 1 : size(img,2)
            r = rand;
            p = rand;
            z1 = std * cos(2*pi*p) * sqrt(-2 * log(r));
            z2 = std * sin(2*pi*p) * sqrt(-2 * log(r));
            
            %매트랩에서 uint8형 변수일 때 연산 결과가 0보다 작으면 0
            %255보다 크면 255로 clipping 되므로 별도의 조건식 필요없음
            noise(2*i,j)=z1;
            noise(2*i+1,j)=z2;
            
            rst(2*i,j) = img(2*i,j) + z1;
            rst(2*i+1,j) = img(2*i+1,j) + z2;
        end
    end
else
    for i = 1 : floor(size(img,1)/2)
        for j = 1 : size(img,2)
            r = rand;
            p = rand;
            z1 = std * cos(2*pi*p) * sqrt(-2 * log(r));
            z2 = std * sin(2*pi*p) * sqrt(-2 * log(r));
            
            %매트랩에서 uint8형 변수일 때 연산 결과가 0보다 작으면 0
            %255보다 크면 255로 clipping 되므로 별도의 조건식 필요없음
            noise(2*i,j)=z1;
            noise(2*i+1,j)=z2;
            rst(2*i,j) = img(2*i,j) + z1;
            rst(2*i+1,j) = img(2*i+1,j) + z2;
        end
    end
    tmp = size(img,1);%마지막 행
    for j = 1: size(img,2)
        r = rand;
        z1 = std * cos(2*pi*p) * sqrt(-2 * log(r));
        
        %매트랩에서 uint8형 변수일 때 연산 결과가 0보다 작으면 0
        %255보다 크면 255로 clipping 되므로 별도의 조건식 필요없음
        noise(tmp,j) = z1;
        rst(tmp,j) = img(tmp,j) + z1;
    end
end

figure(2);%노이즈
imshow(noise)

figure(3);%결과이미지
imshow(rst)

filename = ['rst_std' num2str(std)  '.png' ];
imwrite(rst,filename);
