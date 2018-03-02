识别的图片处理
    


样本数据，有几个特点，所有的他的可能组合是0-9,a-z,A-Z，合计是62个，不包含大写的有36个
    切割方法由于存在粘连，识别起来比较费劲，所以，这个模型里面采用的直接识别，
    当然后续也可以去切割出来，即使粘连，可能还是可以单个识别的吧，但是后面再做尝试了。

    那么粘连在一起的时候，识别的就变成了一个多分类问题，而不是单分类问题了，
    也就是说，我几个数字一起识别，那么就有以下几个问题：
    1. 怎么识别多个字母？
        这里的解决办法，是靠经验判断大多数是几个字母，
        比如大多数是4位的，极少数是5个，那么就放弃5个的，直接认为是4分类。
        但是我认为如果超过5%的少数，就得按照少数来吧。
        那么当定义为5位的时候，那么4位的缺失就用个padding代替吧，比如"_",程序自行判断是个padding吧

    2. 对于多分类，label的vector如何构建？
        对于单分类，我们都是知道是构建一个one-hot的概率向量，然后和结果做交叉熵
        对于多分类，就构建一个one-hot概率向量组成的张量呗，然后继续交叉熵

    3. 交叉熵函数如何书写？      
        好问题，还不知道呢，一会儿研究