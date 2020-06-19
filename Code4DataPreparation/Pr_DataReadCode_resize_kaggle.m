clc
clear
close all

Path='./kaggle chest_xray/train/PNEUMONIA/';
PathOutput='./Data_Kaggle_512_PNEUMONIA/';
Dir=dir(Path);

ind=randperm(360);

for i=1:360
    
    if ind(i)==1 || ind(i)==2
       continue;  
    end
    img=imread([Path Dir(ind(i)).name]);
    img=imresize(img,[512 512]);    
    
    imwrite(img,[PathOutput Dir(ind(i)).name] );
    
    
end