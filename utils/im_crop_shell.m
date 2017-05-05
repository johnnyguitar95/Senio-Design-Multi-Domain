function window = ...
    im_crop_shell(im, bbox, crop_mode, crop_size, padding) 
mean_rgb = {-1, -1, -1};
window = im_crop(im, bbox, crop_mode, crop_size, padding, mean_rgb);
