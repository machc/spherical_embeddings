function run_alignment_experiment(filein, fileout, mode, nrots, ...
                                  sameinstance, upsample, max_instances, ...
                                  normalize, dirout, err_symm)
  % nrots: number of rotations per instance
  % sameinstance: same of different instances
  % upsample: feature map upsample factor
  if nargin < 8
    normalize = false;
  end
  if nargin < 9
    dirout = '~/results';
  end
  
  [~, name, ~] = fileparts(filein);
  datain = load(filein);
  if strcmp(mode, 'target')
    datain.fmap = datain.target;
  end
  [err, R, R1, R2, id1, id2] = align(datain.fmap, datain.rot, nrots, ...
                                     sameinstance, 0, normalize, ...
                                     upsample, max_instances);
  if sameinstance samediff = 'same'; else samediff = 'diff'; end

  if err_symm
    err = min(err, pi-err)
  end

  s = sprintf('|%s|%s|%s|%.4f|', name, mode, samediff, 180/pi*median(err));
  disp(s);
  file = fopen(fileout, 'a');
  fprintf(file, '%s\n', s);
  fclose(file);  

  outfile = sprintf('%s/%s_%s_%s_norm%s_up%dx', ...
                    dirout, name, mode, samediff, mat2str(normalize), upsample);
  save(outfile, 'err', 'R', 'R1', 'R2', 'id1', 'id2');