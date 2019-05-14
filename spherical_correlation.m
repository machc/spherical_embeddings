% this depends on the SOFT library and its matlab interface (not included)
function [estimated_R, corr_grid, a, b, c] = spherical_correlation(imt, imrt)
  assert(size(imt, 1) == size(imt, 2));
  B = size(imt, 1) / 2;

  % accumulate correlation for each channel
  corr_grid = zeros(2*B, 2*B, 2*B);
  for ch=1:size(imt, 3)
    im1 = imt(:, :, ch);
    im2 = imrt(:, :, ch);
    cf_im = sft_mex(double(im1(:)));
    cf_imr = sft_mex(double(im2(:)));

    % correlation output = ISO3FT(SPECTRAL_MATRIX_PROD(cf_im, cf_imr));
    corr_grid_inc = correlate_many_mex(cf_im, cf_imr);
    corr_grid_inc = reshape(corr_grid_inc, 2 * [ B B B ]);
    corr_grid = corr_grid + corr_grid_inc;
  end

  [y, i] = max(corr_grid(:));
  [alpha_grid, beta_grid, gamma_grid] = so3_meshgrid(B);
  [a, b, c] = deal(alpha_grid(i), beta_grid(i), gamma_grid(i));
  estimated_R = rz(a) * ry(b) * rz(c);