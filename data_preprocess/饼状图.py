from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Data for the first pie chart (前对后错)
    labels1 = ['良性', '恶性']
    sizes1 = [11, 5]
    total1 = sum(sizes1)

    # Calculate percentages
    percentages1 = [size / total1 * 100 for size in sizes1]

    # Data for the second pie chart (前错后对)
    labels2 = ['恶性', '良性']
    sizes2 = [19, 12]
    total2 = sum(sizes2)

    # Calculate percentages
    percentages2 = [size / total2 * 100 for size in sizes2]

    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # Create the first pie chart
    plt.subplot(1, 2, 1)
    plt.pie(sizes1, labels=[f'{label}' for label, size in zip(labels1, sizes1)],
            autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, total1 * p / 100),
            startangle=90, colors=['skyblue', 'lightcoral'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('b对d错')

    # Create the second pie chart
    plt.subplot(1, 2, 2)
    plt.pie(sizes2, labels=[f'{label}' for label, size in zip(labels2, sizes2)],
            autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, total2 * p / 100),
            startangle=90, colors=['lightcoral', 'skyblue'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('b错d对')

    # Show the plots
    plt.show()
