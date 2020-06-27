import numpy as np
import matplotlib.pyplot as plt

data = np.load("./showdata.npy")


def plotday(data, label):
    plt.figure(num=3, figsize=(8, 5))
    plt.ylabel('Traffic Flow', fontsize=16)
    plt.plot(data, lw=2, c='blue', ms=4)
    plt.tick_params(labelsize=14)
    plt.ylim(0, 320)
    plt.xlim(0, 24 * 7 + 10)
    peak_hour = [18, 19 + 24 * 1, 18 + 24 * 2, 18 + 24 * 3, 19 + 24 * 4, 17 + 24 * 5, 20 + 24 * 6]
    point = [data[18], data[19 + 24 * 1], data[18 + 24 * 2],
             data[18 + 24 * 3], data[19 + 24 * 4], data[17 + 24 * 5], data[20 + 24 * 6]]
    #     peak_hour = 18
    #     point = data[peak_hour::24]
    #     index = list(range(peak_hour,24*7,24))
    plt.scatter(peak_hour, point, marker='o', c='', edgecolors='r', s=100)

    txt = ["6:00pm", "7:00pm", "6:00pm", "6:00pm", "7:00pm", "5:00pm", "8:00pm", ]
    for i in range(len(txt)):
        #         plt.annotate(txt[i], xy = (peak_hour[i], point[i]), xytext = (peak_hour[i]-10, point[i]+10))
        plt.text(peak_hour[i] - 8, point[i] + 10, txt[i], weight="bold",fontsize=13)
    plt.xticks(list(range(13, 24 * 7, 24)), label)
    plt.savefig("./day.png", dpi = 1000)
    plt.show()


def plotweek(data, label):
    plt.figure(num=3, figsize=(8, 5))
    plt.ylabel('Traffic Flow', fontsize=16)
    plt.plot(data, lw=2, c='darkgreen', ms=4)
    plt.tick_params(labelsize=14)
    plt.ylim(0, 320)
    plt.xlim(0, 24 * 5 + 5)
    peak_hour = [16, 16 + 24 * 1, 19 + 24 * 2, 18 + 24 * 3, 18 + 24 * 4]
    point = [data[16], data[16 + 24 * 1], data[19 + 24 * 2], data[18 + 24 * 3], data[18 + 24 * 4]]
    #     peak_hour = 18
    #     point = data[peak_hour::24]
    #     index = list(range(peak_hour,24*5 - 1,24))
    plt.scatter(peak_hour, point, marker='o', c='', edgecolors='r', s=100)

    txt = ["4:00pm", "4:00pm", "7:00pm", "6:00pm", "6:00pm"]
    for i in range(len(txt)):
        #         plt.annotate(txt[i], xy = (peak_hour[i], point[i]), xytext = (peak_hour[i]-10, point[i]+10))
        plt.text(peak_hour[i] - 7, point[i] + 10, txt[i], weight="bold",fontsize=13)
    plt.xticks(list(range(12, 24 * 5 - 1, 24)), label)
    plt.savefig("./week.png",dpi = 1000)
    plt.show()

def week():
    # list(range(peak_hour,24*7,24))
    label = ["11/01", "11/08", " 11/15", "11/22", "11/29"]
    newdata = []
    for i in range(5):
        newdata.extend(data[-6 + (1 + 7 * i) * 24: -6 + (2 + 7 * i) * 24])

    print(len(newdata))
    plotweek(newdata, label)


# list(range(peak_hour,24*7,24))
def day():
    label = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plotday(data[-6 + (7 + 4) * 24: -6 + 24 * (2 * 7 + 4)], label)

def plot_rmse_mae(dataset,MC,DMC,Y_S,Y_E,savefig):
    plt.figure(figsize=(6, 5))
    X = ['MSTGCN', 'ASTGCN', 'DM-RGCN']
    # 把条形图向右移动
    bar_width = 0.32
    bar1 = list(range(len(X)))
    bar2 = [i + bar_width for i in bar1]

    ylabel = 'RMSE' if min(MC) > 22 else 'MAE'

    plt.tick_params(labelsize=13)
    plt.xlabel(dataset, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(bar2, X, fontsize=14)

    #,color='pink',color='teal'
    plt.ylim(Y_S, Y_E)  # 22,26 , 14,18 , 29,33 ,19,23
    plt.bar(X, MC, width=bar_width, label='Multi-component')
    plt.bar(bar2, DMC, width=bar_width, label='Dynamic Multi-component')
    # txt = ["6:00pm", "7:00pm", "6:00pm", "6:00pm", "7:00pm", "5:00pm", "8:00pm", ]
    for i in range(len(MC)):
        plt.text(bar1[i] - 0.14, MC[i] + 0.1, MC[i], weight="bold", fontsize=13)
        plt.text(bar2[i] - 0.11, DMC[i] + 0.1, DMC[i], weight="bold", fontsize=13)

    plt.legend(fontsize=12)
    plt.savefig(savefig + '.png', dpi=1000)
    plt.show()

def p8_rmse():
    PE8_RMSE_MC = [25.45, 25.04, 23.40]
    PE8_RMSE_DMC = [24.31, 24.50, 23.02]
    plot_rmse_mae('PEMSD8',PE8_RMSE_MC,PE8_RMSE_DMC,22,26,'p8_rmse')

def p8_mae():
    PE8_MAE_MC = [17.08, 16.94, 15.65]
    PE8_MAE_DMC = [16.00, 16.20, 15.18]
    plot_rmse_mae('PEMSD8',PE8_MAE_MC, PE8_MAE_DMC, 14, 18,'p8_mae')

def p4_rmse():
    PE4_RMSE_MC = [32.56, 32.53, 30.21]
    PE4_RMSE_DMC = [31.53, 31.60, 30.05]
    plot_rmse_mae('PEMSD4',PE4_RMSE_MC,PE4_RMSE_DMC,29,34,'p4_rmse')

def p4_mae():
    PE4_MAE_MC = [21.87, 22.02, 20.11]
    PE4_MAE_DMC = [20.92, 21.28, 19.93]
    plot_rmse_mae('PEMSD4',PE4_MAE_MC,PE4_MAE_DMC,19,23,'p4_mae')

if __name__ == '__main__':
    day()
    week()
    p8_rmse()
    p8_mae()
    p4_rmse()
    p4_mae()

