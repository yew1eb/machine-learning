% 计算欧式距离
function dist = distEclud(vecA,vecB)
    dist  = sum(power((vecA-vecB),2));
end


function dist = softDist(vecA,vecB)
    dist = sum(abs(vecB-vecA));
end
