function ims = mdnet_extract_regions(im, boxes, opts)
% MDNET_EXTRACT_REGIONS
% Extract the bounding box regions from an input image. 
%
% Hyeonseob Nam, 2015
% 

num_boxes = size(boxes, 1); % armadillo

crop_mode = opts.crop_mode; %oppts is a struct/class to be created in MDNet_init
crop_size = opts.input_size;
crop_padding = opts.crop_padding;

ims = zeros(crop_size, crop_size, 3, num_boxes, 'single'); %armadillo
% mean_rgb = mean(mean(single(im)));

% for i = 1:num_boxes %armadillo
%     bbox = boxes(i,:);
%     crop = im_crop_original(im, bbox, crop_mode, crop_size, crop_padding);
%     crop = single(crop);
%     ims(:,:,:,i) = crop;
% end

for i = 1:num_boxes %armadillo
    bbox = boxes(i,:);
    crop = im_crop_shell(im, bbox, crop_mode, crop_size, crop_padding);
    crop = single(crop);
    ims(:,:,:,i) = crop;
end