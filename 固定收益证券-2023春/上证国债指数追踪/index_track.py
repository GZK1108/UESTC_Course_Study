
import datetime
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten


# 编写一个函数，用于对国债收盘价.xlsx中债券的久期进行分层
def duration_classify():
    # 导入文件国债收盘价.xlsx
    df = pd.read_excel('国债收盘价.xlsx', sheet_name=0, dtype=object)
    # 删除duration为0的行
    df = df[df['duration'] != 0].reset_index(drop=True)
    # 将类别相同的行合并
    df = df.groupby('类别').agg({'duration': 'mean', 'close': 'mean', '纳入日期': 'min'}).reset_index()
    # 将df按duration的大小进行从小到大排序
    df = df.sort_values(by='duration').reset_index(drop=True)
    # 删除df的close和纳入日期两列
    df = df.drop(['close', '纳入日期'], axis=1)
    # 将df的duration列按照久期大小进行分层，并增加一列名为层级的列，久期0-5为1层，5-10为2层，10-15为3层，15-20为4层，20-25为5层，25-30为6层，30以上为7层
    df['层级'] = pd.cut(df['duration'], bins=[0, 5, 10, 15, 20, 25, 30, 100], labels=[1, 2, 3, 4, 5, 6, 7])
    # 导入文件国债收盘价.xlsx
    dg = pd.read_excel('国债收盘价.xlsx', sheet_name=0, dtype=object)
    # 将df中的层级列合并到dg中类别相同的行
    dg = pd.merge(dg, df[['类别', '层级']], on='类别', how='left')
    # 输出dg到文件上证国债代码_以久期分层.xlsx
    dg.to_excel('上证国债代码_以久期分层.xlsx', index=False)


# 用于判断跟踪效果
def track_effect(results):
    date, y_pre, y_true = results['date'], results['y_pre'], results['y_true']
    # 计算跟踪偏离度
    td = y_pre - y_true
    # 计算跟踪误差
    td_mean = np.mean(td)
    te = np.sqrt(np.sum((td - td_mean) ** 2) / (len(td) - 1))
    return td, te


# 画图
def draw_picture(results):
    # 画图
    date, y_pre, y_true, td = results['date'], results['y_pre'], results['y_true'], results['TD']
    td_mean = np.mean(td)
    # 设置窗口大小
    plt.figure(figsize=(25, 8))
    # 画图，让date为x轴，y为y轴，date倾斜显示
    plt.xlabel('Date', fontsize=15, color='r')
    plt.ylabel('r', fontsize=15, color='r')
    plt.xticks(rotation=90)  # 旋转90度
    plt.plot(date, y_pre, label='simulate_r')
    plt.plot(date, y_true, label='index_r')
    plt.title('Index Tracking', fontsize=20, color='r')
    plt.legend()
    plt.figure(figsize=(25, 8))
    plt.xlabel('Date', fontsize=15, color='r')
    plt.ylabel('TD', fontsize=15, color='r')
    plt.xticks(rotation=90)  # 旋转90度
    plt.plot(date, td, label='TD')
    # 画均值，并显示数值
    plt.axhline(y=td_mean,c="r", ls="--")
    print('日均跟踪偏离度为：', td_mean)
    plt.legend()
    plt.title('Index Tracking deviation', fontsize=20, color='r')
    plt.show()


# 追踪结果计算
def track_result(bond_index_return, w):
    """
    :param bond_index_return: [mx(n+1)],m为时间长度，n为债券数量，最后一列为指数
    :param w: 权重
    :return: 时间，模拟指数，真实指数
    """
    date = bond_index_return['date'].copy()
    # 只保留date的月和日
    date = date.str.slice(start=5)
    # 提取x和y，x为债券价格，y为指数价格
    x = bond_index_return.iloc[:, 1:-1].values
    y = bond_index_return.iloc[:, -1].values
    # 将w转换为列向量
    w = np.array(w).reshape(-1, 1)
    simulate_r = np.dot(x, w)
    return date, simulate_r, y


# 神经网络求解参数，为了保证可解释性只构造了一层，效果和线性求解相差不大，因此未采用
def nn_track(bond_index_return):
    x = bond_index_return.iloc[:, 1:-1]
    y = bond_index_return.iloc[:, -1]
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(1, kernel_initializer='normal'))  # 作为输出层，因为是回归，所以不用激活函数
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    # 训练
    model.fit(x, y, epochs=100, batch_size=64, verbose=2)
    return model


# 指数追踪，求解参数
def index_track(bond_index_return):
    """
    :param bond_index_return: [mx(n+1)],m为时间长度，n为债券数量，最后一列为指数
    :return: 权值解
    """
    bond_index_return = bond_index_return.drop('date', axis=1)
    n = bond_index_return.shape[1] - 1  # n为债券数量
    x = [0.0] * n  # x为债券权重,一列向量
    P = 2 * matrix(bond_index_return.cov().iloc[:-1, :-1].values)  # P为债券之组合协方差矩阵
    q = - 0.5 * matrix(bond_index_return.cov().iloc[:-1, -1].values)  # q为债券与指数协方差，一行向量

    G = matrix(-np.eye(len(x)))  # G为不等式约束矩阵，为单位矩阵
    h = matrix(np.zeros(len(x)))  # h为不等式约束条件，为0向量,GX<=h表示权重大于0

    A = matrix([1.0] * n, (1, n))  # A为等式约束矩阵，为1向量
    b = matrix(1.0)  # b为等式约束条件，为1，AX=b表示权重之和为1

    sol = solvers.qp(P, q, G, h, A, b)  # 求解
    return sol


# 生成用于确定参数的数据，使用收益率
def train_data_create(df, start_time, end_time):
    """
    :param df: 原始数据
    :param start_time: 开始时间
    :param end_time: 结束时间
    :return: 处理后的数据
    """
    data = df[['date', '收益率', '类别', '纳入日期']].copy()
    data.rename(columns={'收益率': '债券'}, inplace=True)

    # 剔除纳入日期在start_time之后的数据
    data = data[data['纳入日期'] <= start_time].reset_index(drop=True)
    # 生成一个时间序列，按日生成
    date = pd.date_range(start_time, end_time, freq='D')
    # 计算date长度
    date_len = len(date)
    # date转为字符串
    date = date.strftime('%Y-%m-%d')
    # 将时间序列转换为dataframe
    dsave = pd.DataFrame(date, columns=['date'])
    # 取数
    for symbol in data['类别'].unique():
        mini_data = data[data['类别'] == symbol][['date', '债券']].reset_index(drop=True)
        # 选取时间区间在start_time到end_time之间的数据
        mini_data = mini_data[mini_data['date'].isin(date)].reset_index(drop=True)
        # 下面用于剔除数据过少的债券
        if mini_data.shape[0] < 0:
            continue
        # 重命名
        mini_data.rename(columns={'债券': symbol}, inplace=True)  # 将久期分层信息放入债券名称后面，便于后期筛选
        # mini_data与dsave合并
        dsave = pd.merge(dsave, mini_data, on='date', how='left')
        # 利用指数为基准，剔除空值
        if symbol == '000012':
            dsave = dsave.dropna().reset_index(drop=True)
    # 将指数放最后，注意最后一个0为层级信息
    dsave['指数'] = dsave.pop('000012')
    # ===========================按久期筛选（作废）==================================
    # duration_df = pd.read_excel('上证国债代码_以久期分层.xlsx', sheet_name=0, dtype=object)[['类别', '层级']]
    # mini_symbols = dsave.columns[1:].tolist()
    # # 选择duration_df中在mini_symbols中的债券
    # duration_df = duration_df[duration_df['类别'].isin(mini_symbols)].reset_index(drop=True)
    # # 按层级分组，每组提取30%的债券,4
    # new_bonds = duration_df.groupby('层级').apply(lambda x: x.sample(frac=0.3, random_state=7)).reset_index(drop=True)
    # # 选择dsave中在new_bonds中的债券
    # dsave = dsave[['date'] + new_bonds['类别'].tolist() + ['指数']]
    # ======================================================================
    # 填充
    dsave = dsave.fillna(0)
    return dsave


# 生成用于跟踪指数的数据，采用收益率
def track_data_create(df, train_symbols, start_time, end_time):
    # 插入'000012'到trainsymbols
    train_symbols.insert(0, '000012')
    data = df[['date', '收益率', '类别']].copy()
    # 生成一个时间序列
    date = pd.date_range(start_time, end_time, freq='D')
    # date转为字符串
    date = date.strftime('%Y-%m-%d')
    # 将时间序列转换为dataframe
    dsave = pd.DataFrame(date, columns=['date'])
    # 提取类别为trainsymbols的数据
    data = data[data['类别'].isin(train_symbols)].reset_index(drop=True)
    for symbol in data['类别'].unique():
        mini_data = data[data['类别'] == symbol][['date', '收益率']].reset_index(drop=True)
        # 选取时间区间在2023-01-01到2023-04-30之间的数据
        mini_data = mini_data[mini_data['date'].isin(date)].reset_index(drop=True)
        # 重命名
        mini_data.rename(columns={'收益率': symbol}, inplace=True)
        # mini_data与dsave合并
        dsave = pd.merge(dsave, mini_data, on='date', how='left')
        # 利用指数为基准，剔除空值
        if symbol == '000012':
            dsave = dsave.dropna().reset_index(drop=True)
    # 将指数放最后
    dsave['指数'] = dsave.pop('000012')
    # 前值填充
    dsave = dsave.fillna(0)
    return dsave


if __name__ == '__main__':
    df = pd.read_excel('数据/国债收益率.xlsx', sheet_name=0, dtype=object)
    df = df.dropna().reset_index(drop=True)
    # ============================这一部分用于按久期选取债券============================
    duration_df = pd.read_excel('数据/上证国债代码_以久期分层.xlsx', sheet_name=0, dtype=object)[['类别', '层级']]
    # 按层级分组，每组取30%的类别
    # 获取当前时间
    now_time = datetime.datetime.now()
    # 提取秒
    seed = now_time.second
    print(seed)  # 43可以49,21
    duration_df = duration_df.groupby('层级').apply(lambda x: x.sample(frac=0.3, random_state=21)).reset_index(drop=True)
    mini_symbols = duration_df['类别'].tolist()
    mini_symbols = mini_symbols + ['000012']
    # 保存duration_df为csv，编码为GBK
    # duration_df.to_csv('duration_df.csv', encoding='GBK')
    # 取出df中类别在MINI_SYMBOLS中的数据，注释表示不抽样
    df = df[df['类别'].isin(mini_symbols)].reset_index(drop=True)
    # ==============================================================================
    # 参数设置
    start_time = '2022-11-30'  # 训练数据开始时间，可变
    end_time = '2023-04-30'  # 追踪结束时间，不可变
    # 按月生成数据，作为训练测试参考
    datelist = pd.date_range(start_time, end_time, freq='M')
    datelist = datelist.strftime('%Y-%m-%d').tolist()

    # 新建存储
    final_date = []
    final_y_pre = []
    final_y_true = []

    for i in range(4):
        train_data = train_data_create(df, start_time=datelist[i], end_time=datelist[i + 1])
        symbols = train_data.columns[1:-1].tolist()  # 债券代码，保证测试数据与求参的数据债券一致
        test_data = track_data_create(df, symbols, start_time=datelist[i + 1], end_time=datelist[i + 2])
        # 求解
        sol = index_track(train_data)
        w = list(sol['x'])
        # NN网络
        # model = nn_track(train_data)
        # 模型权重
        # w_nn = model.get_weights()[0]
        # ============权重保存部分==================
        # weight = pd.DataFrame(w, index=symbols[1:], columns=['权重'])
        # weight.to_csv(f'权重{i}.csv', encoding='GBK')
        # ========================================
        mini_date, y_pre, y_true = track_result(test_data, w)
        final_date.extend(mini_date)
        # 提取y_pre的值
        y_pre = [i[0] for i in y_pre]
        final_y_pre.extend(y_pre)
        final_y_true.extend(y_true)

    # 生成结果
    result = pd.DataFrame({'date': final_date, 'y_pre': final_y_pre, 'y_true': final_y_true})
    TD, TE = track_effect(result)
    print('跟踪误差为：', TE)
    # print('跟踪偏差为：', TD)
    result['TD'] = TD
    # 画图
    draw_picture(result)
