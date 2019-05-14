function [err, Rs, R1s, R2s, id1s, id2s] = align(fmaps, rots, nrots, ...
                                                 sameinstance, seed, normalize, upsample, max_instances)
  if nargin >= 5
    rng(seed);
  end
  if nargin < 6
    normalize = false;
  end

  if nargin < 7
    upsample = 1;
  end

  if nargin < 8
      max_instances = inf;
  end

  assert(nrots > 1);
  ntotal = size(fmaps, 1);
  ninstances = ntotal / nrots;
  ninstances = min(ninstances, max_instances);
  err = [];
  Rs = {};
  R1s = {};
  R2s = {};
  id1s = [];
  id2s = [];
  for i=1:ninstances
    for j=1:(nrots-1)
      id1=i;
      % use this to fetch from other class
      if sameinstance
        id2=i+ninstances*j;
      else
        id2 = randi(ntotal);
      end
      % no transpose!
      im1 = squeeze(fmaps(id1, :, :, :));
      im2 = squeeze(fmaps(id2, :, :, :));

      if upsample > 1
        % TODO: would be more efficient to upsample by padding in the spectral domain
        im1 = imresize(im1, upsample);
        im2 = imresize(im2, upsample);
      end

      if normalize
        for k=1:size(im1, 3)
          im1(:, :, k) = im1(:, :, k)/norm(im1(:, :, k));
          im2(:, :, k) = im2(:, :, k)/norm(im2(:, :, k));
        end
      end

      R1 = squeeze(rots(id1, :, :));
      R2 = squeeze(rots(id2, :, :));
      [R, e] = spherical_correlation_err(im1, R1, im2, R2);
      err = [err; e];
      Rs = [Rs; R];
      R1s = [R1s; R1];
      R2s = [R2s; R2];
      id1s = [id1s; id1];
      id2s = [id2s; id2];      
    end
  end