me = mfilename;                                            % what is my filename
mydir = which(me); mydir = mydir(1:end-2-numel(me));        % where am I located
addpath([mydir,'util'])
addpath([mydir, 'libGP'])
addpath([mydir, 'testHSM'])
