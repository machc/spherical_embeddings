function [R, error_angle] = err(im1, R1, im2, R2)
 R = spherical_correlation(im1, im2);
 error_R = R2' * R1 * R; % should be close to I
 error_angle = acos((trace(error_R) - 1)/2);
