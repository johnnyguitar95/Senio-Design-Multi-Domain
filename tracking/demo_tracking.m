%% DEMO_TRACKING
%
% Running the MDNet tracker on a given sequence.
%
% Hyeonseob Nam, 2015
%

%clear;

%conf = genConfig('vot','ball1');
conf = genConfig('otb','Football');

switch(conf.dataset)
    case 'otb'
        net = fullfile('models','mdnet_vot-otb.mat'); %creates full file path
    case 'vot2014'
        net = fullfile('models','mdnet_otb-vot14.mat');
    case 'vot2015'
        net = fullfile('models','mdnet_otb-vot15.mat');
end

[time, result] = mdnet_run(conf.imgList, conf.gt(1,:), net);

plot(time);

for i=1:size(conf.gt(),1)
    % compares our results with datasets results
    comparative_results(i) = overlap_ratio(conf.gt(i,:), result(i,:));
end



