# 大数据精准营销中搜狗用户画像挖掘

> http://www.wid.org.cn/data/science/player/competition/detail/description/239

输入：用户查询词
输出：用户画像（年龄、性别、学历）


## 评价标准

本赛题采用分类准确率进行评价。

对参赛者提供的结果文件，全部与标准结果匹配计算准确率。其中，性别、年龄、学历分别计算准确率，最终以平均准确率作为评判依据。

> 具体如下：

1. 准确率计算：

![](http://latex.codecogs.com/gif.latex?P= \\frac{1}{N} \\sum_N^{i=0}I(y_i^* = y_i))


* 其中， 表示算法对第i个样本预测类别，表示第i个样本的真实类别。函数I为Indicator Function，当预测结果与真实结果完全相同时输出为1，否则为零。

2. 平均准确率计算：

![](http://latex.codecogs.com/gif.latex?P = \\frac{P_{gender} + P_{age} + P_{education}}{3})

3. 模型的判定结果不允许出现0，即只有明确的标签才为有效结果。

4. 预设指标：平均准确率50%以上为有效成绩。