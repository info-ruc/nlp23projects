本次的实验报告是关于新闻文本的分类问题，数据集大小为74万篇，在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。代码采用了macbert-large-chinese模型，数据读取结果   ID   Count     Ratio      Weight  Weight Norm
科技   0  162245  0.194874    5.131522     0.008467
体育   1  130982  0.157401    6.353183     0.010482
时政   2   62867  0.075455   13.252862     0.021867
股票   3  153949  0.184670    5.415058     0.008935
娱乐   4   92228  0.110792    9.025897     0.014892
教育   5   41680  0.050159   19.936703     0.032895
家居   6   32363  0.038976   25.657085     0.042333
财经   7   36963  0.044372   22.536494     0.037184
房产   8   19922  0.023981   41.699695     0.068802
社会   9   50541  0.060820   16.442063     0.027129
游戏  10   24283  0.029152   34.303018     0.056598
彩票  11    7598  0.009077  110.171449     0.181778
星座  12    3515  0.004281  233.614095     0.385452
时尚  13   13335  0.015990   62.539146     0.103187
The length information of Train&&Dev is as follows:/n
count    832471.000000
mean         19.388112
std           4.097139
min           2.000000
25%          17.000000
50%          20.000000
75%          23.000000
max          48.000000
Name: text_a, dtype: float64
The length information of Test is as follows:/n
count    83599.000000
mean        19.815022
std          3.883845
min          3.000000
25%         17.000000
50%         20.000000
75%         23.000000
max         84.000000
Name: text_a, dtype: float64，
模型训练结果：global step 10, epoch: 1, batch: 10, loss: 2.80887, avgLoss: 2.79764, acc: 0.04922
global step 20, epoch: 1, batch: 20, loss: 2.81051, avgLoss: 2.79560, acc: 0.04844
global step 30, epoch: 1, batch: 30, loss: 2.72834, avgLoss: 2.78979, acc: 0.04779
global step 40, epoch: 1, batch: 40, loss: 2.67694, avgLoss: 2.77567, acc: 0.04736
global step 50, epoch: 1, batch: 50, loss: 2.62486, avgLoss: 2.75062, acc: 0.05063
global step 60, epoch: 1, batch: 60, loss: 2.40714, avgLoss: 2.71518, acc: 0.06003
global step 70, epoch: 1, batch: 70, loss: 2.39192, avgLoss: 2.67804, acc: 0.07511
global step 80, epoch: 1, batch: 80, loss: 2.28870, avgLoss: 2.63808, acc: 0.09463
global step 90, epoch: 1, batch: 90, loss: 2.03402, avgLoss: 2.58807, acc: 0.12144
global step 100, epoch: 1, batch: 100, loss: 2.02840, avgLoss: 2.53682, acc: 0.14992
global step 110, epoch: 1, batch: 110, loss: 1.87569, avgLoss: 2.48542, acc: 0.18107
global step 120, epoch: 1, batch: 120, loss: 1.76106, avgLoss: 2.43113, acc: 0.21530
global step 130, epoch: 1, batch: 130, loss: 1.53635, avgLoss: 2.36686, acc: 0.24988
global step 140, epoch: 1, batch: 140, loss: 1.47397, avgLoss: 2.30514, acc: 0.28103
global step 150, epoch: 1, batch: 150, loss: 1.32365, avgLoss: 2.23979, acc: 0.31013
global step 160, epoch: 1, batch: 160, loss: 1.14416, avgLoss: 2.17474, acc: 0.33774
global step 170, epoch: 1, batch: 170, loss: 0.97029, avgLoss: 2.10709, acc: 0.36441
global step 180, epoch: 1, batch: 180, loss: 0.85500, avgLoss: 2.04065, acc: 0.38885
global step 190, epoch: 1, batch: 190, loss: 0.81579, avgLoss: 1.97999, acc: 0.40987
global step 200, epoch: 1, batch: 200, loss: 0.80477, avgLoss: 1.92090, acc: 0.43092
global step 210, epoch: 1, batch: 210, loss: 0.67737, avgLoss: 1.86330, acc: 0.44933
global step 220, epoch: 1, batch: 220, loss: 0.63712, avgLoss: 1.80905, acc: 0.46726
global step 230, epoch: 1, batch: 230, loss: 0.63658, avgLoss: 1.75819, acc: 0.48392
global step 240, epoch: 1, batch: 240, loss: 0.56719, avgLoss: 1.70813, acc: 0.49946
global step 250, epoch: 1, batch: 250, loss: 0.67474, avgLoss: 1.66283, acc: 0.51361
global step 260, epoch: 1, batch: 260, loss: 0.55390, avgLoss: 1.61891, acc: 0.52707
global step 270, epoch: 1, batch: 270, loss: 0.46013, avgLoss: 1.57744, acc: 0.53947
global step 280, epoch: 1, batch: 280, loss: 0.44831, avgLoss: 1.53829, acc: 0.55137
global step 290, epoch: 1, batch: 290, loss: 0.42532, avgLoss: 1.50052, acc: 0.56273
global step 300, epoch: 1, batch: 300, loss: 0.40972, avgLoss: 1.46471, acc: 0.57346
eval loss: 0.38166, accu: 0.88790
The best model is found in epoch: 1, batch: 300
global step 310, epoch: 1, batch: 310, loss: 0.50498, avgLoss: 1.43246, acc: 0.86797
global step 320, epoch: 1, batch: 320, loss: 0.46750, avgLoss: 1.40040, acc: 0.87617
global step 330, epoch: 1, batch: 330, loss: 0.31396, avgLoss: 1.36967, acc: 0.87799
global step 340, epoch: 1, batch: 340, loss: 0.27074, avgLoss: 1.34135, acc: 0.87900
global step 350, epoch: 1, batch: 350, loss: 0.31216, avgLoss: 1.31326, acc: 0.88516
global step 360, epoch: 1, batch: 360, loss: 0.34148, avgLoss: 1.28729, acc: 0.88522
global step 370, epoch: 1, batch: 370, loss: 0.21923, avgLoss: 1.26295, acc: 0.88633
global step 380, epoch: 1, batch: 380, loss: 0.30752, avgLoss: 1.23911, acc: 0.88716
global step 390, epoch: 1, batch: 390, loss: 0.36066, avgLoss: 1.21651, acc: 0.88880
global step 400, epoch: 1, batch: 400, loss: 0.27352, avgLoss: 1.19502, acc: 0.89020
global step 410, epoch: 1, batch: 410, loss: 0.33598, avgLoss: 1.17442, acc: 0.88871
global step 420, epoch: 1, batch: 420, loss: 0.46979, avgLoss: 1.15539, acc: 0.88864
global step 430, epoch: 1, batch: 430, loss: 0.39568, avgLoss: 1.13622, acc: 0.88909
global step 440, epoch: 1, batch: 440, loss: 0.38146, avgLoss: 1.11820, acc: 0.88943
global step 450, epoch: 1, batch: 450, loss: 0.26302, avgLoss: 1.10059, acc: 0.89044
global step 460, epoch: 1, batch: 460, loss: 0.29684, avgLoss: 1.08349, acc: 0.89143
global step 470, epoch: 1, batch: 470, loss: 0.36123, avgLoss: 1.06718, acc: 0.89198
global step 480, epoch: 1, batch: 480, loss: 0.43308, avgLoss: 1.05180, acc: 0.89234
global step 490, epoch: 1, batch: 490, loss: 0.26844, avgLoss: 1.03712, acc: 0.89282
global step 500, epoch: 1, batch: 500, loss: 0.29666, avgLoss: 1.02351, acc: 0.89352
global step 510, epoch: 1, batch: 510, loss: 0.37351, avgLoss: 1.00962, acc: 0.89347
global step 520, epoch: 1, batch: 520, loss: 0.35245, avgLoss: 0.99562, acc: 0.89370
global step 530, epoch: 1, batch: 530, loss: 0.39360, avgLoss: 0.98262, acc: 0.89446
global step 540, epoch: 1, batch: 540, loss: 0.32699, avgLoss: 0.97040, acc: 0.89495
global step 550, epoch: 1, batch: 550, loss: 0.24113, avgLoss: 0.95897, acc: 0.89505
global step 560, epoch: 1, batch: 560, loss: 0.38939, avgLoss: 0.94733, acc: 0.89542
global step 570, epoch: 1, batch: 570, loss: 0.34163, avgLoss: 0.93702, acc: 0.89515
global step 580, epoch: 1, batch: 580, loss: 0.28249, avgLoss: 0.92585, acc: 0.89562
global step 590, epoch: 1, batch: 590, loss: 0.21886, avgLoss: 0.91426, acc: 0.89604
global step 600, epoch: 1, batch: 600, loss: 0.23334, avgLoss: 0.90440, acc: 0.89645
eval loss: 0.26711, accu: 0.90184
The best model is found in epoch: 1, batch: 600
global step 610, epoch: 1, batch: 610, loss: 0.28467, avgLoss: 0.89408, acc: 0.90781
global step 620, epoch: 1, batch: 620, loss: 0.27774, avgLoss: 0.88418, acc: 0.91016
global step 630, epoch: 1, batch: 630, loss: 0.26561, avgLoss: 0.87450, acc: 0.91081