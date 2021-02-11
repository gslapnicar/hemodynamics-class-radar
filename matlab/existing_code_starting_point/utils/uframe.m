function framed = uframe(sig, windowsize, step)
%% function framed = uframe(sig, windowsize, step)
% frame the signal
% step optional

%%
nx = length(sig);
if (nargin < 3)
   step = windowsize;
end

nf = fix((nx-windowsize+step)/step);
framed=zeros(nf,windowsize);
indf= step*(0:(nf-1)).';
inds = (1:windowsize);
framed(:) = sig(indf(:,ones(1,windowsize))+inds(ones(nf,1),:));