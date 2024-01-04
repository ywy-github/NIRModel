本代码为前处理中选帧+减暗帧除LevLED部分，运行Test.py返回data
字段解释：
Laterality:病侧
Rows:图像的宽
Columns:图像的高
data.MainPressionTime：压力时间
data.MainPressure：压力值
data.ILState：选择的LED灯号
data.length：初始总帧数
data.totalframes：总帧数
data.Main_image：选帧并归一化后的Main_data
data.MainTime：选帧并归一化后的时间
data.LEDPeriod：3灯or5灯
data.Dark_image：暗帧Dark_data
data.ProcessingModeds:硬件参数
data.DarkDiff：暗帧差

