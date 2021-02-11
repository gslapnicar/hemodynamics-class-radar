function [means,diffs,meanDiff,CR,linFit] = BlandAltman(var1, var2, flag, flag2)
 
    %%%Plots a Bland-Altman Plot
    %%%INPUTS:
    %%% var1 and var2 - vectors of the measurements
    %%%flag - how much you want to plot
        %%% 0 = no plot
        %%% 1 = just the data
        %%% 2 = data and the difference and CR lines
        %%% 3 = above and a linear fit
    %%%flag2 - use percentage scale on y axis
        %%% 0 = use regular y axis
        %%% 1 = use percentage y axis
    %%%
    %%%OUTPUTS:
    %%% means = the means of the data
    %%% diffs = the raw differences
    %%% meanDiff = the mean difference
    %%% CR = the 2SD confidence limits
    %%% linfit = the paramters for the linear fit
    
    
    if (nargin<1)
       %%%Use test data
       var1=[512,430,520,428,500,600,364,380,658,445,432,626,260,477,259,350,451];%,...
       var2=[525,415,508,444,500,625,460,390,642,432,420,605,227,467,268,370,443];
       flag = 3;
    end
    
    if nargin<3
        flag = 0;
        flag2 = 0;
    elseif nargin<4
        flag2 = 0;
    end
    
    means = mean([var1;var2]);
    if flag2
        diffs = ((var1-var2)./means).*100;
    else
        diffs = var1-var2;
    end
    
    meanDiff = mean(diffs);
    sdDiff = std(diffs);
    CR = [meanDiff + 1.96 * sdDiff, meanDiff - 1.96 * sdDiff]; %%95% confidence range
    
    linFit = polyfit(means,diffs,1); %%%work out the linear fit coefficients
    
    %%%plot results unless flag is 0
    if flag ~= 0
        figure
        plot(means,diffs,'.','MarkerSize',1)
        hold on
        xlabel('Mean of method A and B');
        if flag2
            ylabel('(method A - method B)/Average %');
        else
            ylabel('method A - method B');
        end
        if flag > 1
            plot(means, ones(1,length(means)).*CR(1),'k-'); %%%plot the upper CR
            plot(means, ones(1,length(means)).*CR(2),'k-'); %%%plot the lower CR
            plot(means,ones(1,length(means)).*meanDiff,'k'); %%%plot zero
        end
        if flag > 2
            plot(means, means.*linFit(1)+linFit(2),'Color',[0.8500 0.3250 0.0980]); %%%plot the linear fit
        end
        title('Bland-Altman Plot')
    end
    
end
