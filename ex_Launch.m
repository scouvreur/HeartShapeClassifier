clear all;
close all;
clc;

%% Plot data
load('data.mat');
unknownIDX = find(isnan(labels));

K = size(labels,1);
numCP = size(shapeED,3);

colorT = {[0,0,1],[1,0,0],[0,0,0]};
numGroups = length(colorT); %%% group 1 / group 2 / unknown

apex = 31; %numCP/2 + 0.5;

figure('units','normalized','position',[0 0 0.5 1]);
for spi=1:2 %%% ED / ES
    for sbi=1:numGroups
        if( sbi == 3 )
            idx = find(isnan(labels));
        else
            idx = find(labels==sbi);
        end        
        if(~isempty(idx))
            subplot(numGroups,2,(sbi-1)*2+spi); hold on;
            for ki=1:length(idx)
                k = idx(ki);
                if(spi==1)
                    tmp = squeeze( shapeED(k,:,:) );
                else
                    tmp = squeeze( shapeES(k,:,:) );
                end
                plot(tmp(1,:),tmp(2,:),'Color',colorT{sbi});
                plot(tmp(1,apex),tmp(2,apex),'k.');
            end
            axis equal; grid on;
            axis([-30,60,0,130]);
        end
    end
end

%% Compute data attributes

features = zeros(K,12);

shapeArea = zeros(K,2);
apexCurvature = zeros(K,2);
septalCurvature = zeros(K,2);
localLength = zeros(K,numCP-1,2);
for spi=1:2 %%% ED / ES
    for k=1:K
        if(spi==1)
            tmp = squeeze( shapeED(k,:,:) );
        else
            tmp = squeeze( shapeES(k,:,:) );
        end
        
        %%% length
        tmpD = tmp(:,2:numCP) - tmp(:,1:(numCP-1));
        tmpD = ( sum(tmpD.^2,1) ).^(1/2);
        localLength(k,:,spi) = tmpD;
        
        %%% features 1 & 2
        tmpD = sum(tmpD(:));
        features(k,spi) = tmpD;

        %%% features 3 & 4
        tmpA = polyarea(tmp(1,:),tmp(2,:));
        features(k,spi+2) = tmpA;
        
        c = zeros(numCP,1);
        %%% first derivative = tangent vector T
        tmpD = tmp(:,2:numCP) - tmp(:,1:(numCP-1));
        %%% normalize it
        tmpNorm = (sum(tmpD.^2,1)).^(1/2);
        tmpD = tmpD ./ repmat(tmpNorm,[2,1]);
        %%% second derivative = dT/ds
        tmpD2 = tmpD(:,2:(numCP-1)) - tmpD(:,1:(numCP-2));
        tmpNorm2 = (sum(tmpD2.^2,1)).^(1/2);
        c(2:(numCP-1)) = tmpNorm2;
        
        %%% features 7 & 8
        features(k,spi+6) = c(apex);
        %%% features 9 & 10
        features(k,spi+8) = mean(c(1:(apex-10)));   
        
    end
end
%%% feature 5
tmpS = squeeze( ( localLength(:,:,2) - localLength(:,:,1) ) ./ localLength(:,:,1) );
features(:,5) = mean( tmpS , 2 );
%%% feature 6
features(:,6) = (features(:,4) - features(:,3) ) ./ features(:,3);
%%% features 11 & 12
features(:,11) = sysDuration;
features(:,12) = HR;

%% Plot extracted features

legendT{1} = 'feature 1 = Length @ ED';
legendT{2} = 'feature 2 = Length @ ES';
legendT{3} = 'feature 3 = Area @ ED';
legendT{4} = 'feature 4 = Area @ ES';
legendT{5} = 'feature 5 = Global Longitudinal Strain (GLS)';
legendT{6} = 'feature 6 = Area Change';
legendT{7} = 'feature 7 = Apical Curvature @ ED';
legendT{8} = 'feature 8 = Apical Curvature @ ES';
legendT{9} = 'feature 9 = Septal Curvature @ ED';
legendT{10} = 'feature 10 = Septal Curvature @ ES';
legendT{11} = 'feature 11 = Systolic Duration';
legendT{12} = 'feature 12 = HR';

figure('units','normalized','position',[0.5 0 0.5 1]);
for fi=1:length(legendT)
    subplot(6,2,fi); hold on;
    for sbi=1:numGroups
        if( sbi == 3 )
            idx = find(isnan(labels));
        else
            idx = find(labels==sbi);
        end        
        tmp = features(:,fi);
        [nb,xb] = hist(tmp(idx),10);
        bh = bar(xb,nb);
        set(bh,'facecolor',colorT{sbi});
        xlabel([num2str(fi),'= ',legendT{fi}]);
        grid on;
        axisTop = 25;
        switch fi
            case 1
                axis([100,270,0,axisTop]);
            case 2
                axis([100,270,0,axisTop]);
            case 3
                axis([1000,6000,0,axisTop]);
            case 4
                axis([1000,6000,0,axisTop]);
            case 5
                axis([-0.25,0.05,0,axisTop]);
            case 6
                axis([-0.5,0,0,axisTop]);
            case 7
                axis([0,0.5,0,axisTop]);
            case 8
                axis([0,0.5,0,axisTop]);
            case 9
                axis([0,0.12,0,axisTop]);
            case 10
                axis([0,0.12,0,axisTop]);
            case 11
                axis([0.25,0.6,0,axisTop]);
            case 12
                axis([40,110,0,axisTop]);
        end        
    end
end
drawnow;

% %% Launch learning
% 
% featuresUsed = [1:12];
% y = NaN(K,1)-1;
% y(labels > 1) = 1;
% y(labels == 1) = 0;
% X = features(:,featuresUsed);
% %%% normalizing data
% X = ( X - repmat(mean(X,1),[K,1]) ) ./ repmat(std(X,0,1),[K,1]);
% 
% %%% Testing set = ???
% 
% %%% Training / validation set = ???
% 
% model = svmTrain(TODO,TODO,C,@(x1,x2)gaussianKernel(x1,x2,sigma)); 
% yPRED = svmPredict(model,...);
% 
% if( length(featuresUsed)==2 )
%     figure('units','normalized','position',[0 0.25 1 0.5]); hold on;
%     plot(TODO,TODO,'r.','MarkerSize',10);
%     plot(TODO,TODO,'b.','MarkerSize',10);
%     axis square; grid on;
%     axis manual;
%     drawnow;
%     visualizeBoundary(TODO,TODO,model);
% end
