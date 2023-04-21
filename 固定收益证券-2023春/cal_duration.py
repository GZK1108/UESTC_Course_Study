from sympy import *
"""
包含计算价格、收益率、四种久期、凸性值、有效凸性值以及使用凸性值预测价格
@author: Gong Zhike
"""

# 计算久期及凸性
# 计算价格
def calprice(n, y, M, i, steps):
    """
    :param n: 债券期限（年）
    :param y: 到期收益率
    :param M: 票面价值
    :param i:息票率
    :param steps:每年计息次数
    :return:价格
    """
    k = y / steps  # 每一期收益率
    N = n * steps  # 期数
    C = M * i / steps  # 计算每期息票
    return C * (1 - (1 + k) ** (-N)) / k + M / (1 + k) ** N


# 计算到期收益率
def calyeild(p, n, M, i, steps):
    N = n * steps  # 期数
    C = M * i / steps  # 计算每期息票
    x = symbols('x')
    answer = solve(C * (1 - (1 + x) ** (-N)) / x + M / (1 + x) ** N - p, x)  # 解方程，可能有多实数根
    res = list(filter(lambda x: 0 < x < 1, answer))[0]  # 取第一个在（0,1）的根（如果存在多个0-1根，则自行判断几个正确）
    y = round(steps * res, 4)
    return y


# 计算久期
def duration(p, n, y, M, i, steps, isyear):
    """
    :param p: 债券价格
    :param n: 债券期限（年）
    :param y: 到期收益率
    :param M: 票面价值
    :param i:息票率
    :param steps:每年计息次数
    :param isyear:是否转换成年,选择True或者False
    :return:在每年计息steps次的情况下的麦考利久期或年久期
    """
    C = M * i / steps  # 计算每期息票
    N = n * steps  # 计算记期次数
    s = 0
    for step in range(N):
        k = step + 1
        s = s + C * k / (1 + y / steps) ** k
    s = s + N * M / (1 + y / steps) ** N
    d = s / p
    if isyear:
        d = d / steps
    return d


# 计算修正久期
def modified_duration(p, n, y, M, i, steps, isyear):
    d = duration(p, n, y, M, i, steps, isyear)
    dm = d / (1 + y / steps)
    return dm


# 美元久期
def dallor_duration(p, n, y, M, i, steps):
    # 默认换算成年，将True改成False取消年
    dm = modified_duration(p, n, y, M, i, steps, True)
    da = dm * p
    return da


# 计算有效久期
def effective_duration(p, n, y, M, i, steps, deltay):
    """
    :param p: 价格
    :param n: 期限（年）
    :param y: 到期收益率
    :param M: 票面价值
    :param i: 息票率
    :param steps: 每年计息次数
    :param deltay: 收益率变化大小
    :return: 有效久期
    """
    # 计算收益率上升
    y1 = y + deltay
    p1 = calprice(n, y1, M, i, steps)
    # 计算收益率下降
    y2 = y - deltay
    p2 = calprice(n, y2, M, i, steps)
    ed = (p2 - p1) / (2 * p * deltay)  # 计算有效久期
    return ed


# 计算凸性值，isyear=True表示换算成年凸性值
def convexity_measure(p, n, y, M, i, steps, isyear):
    # 返回值为在计息次数为steps下的凸性，换算成年凸性需要除以steps^2
    C = M * i / steps  # 计算每期息票
    N = n * steps  # 计算记期次数
    r = y / steps  # 每一期收益率
    s = 0
    for step in range(N):
        k = step + 1
        s = s + k * (k + 1) * C / (1 + r) ** k
    d2p_dy2 = (1 / (1 + r) ** 2) * (s + N * (N + 1) * M / (1 + r) ** N)

    conv = d2p_dy2 / p
    if isyear:
        conv = conv / steps ** 2  # 执行表示年凸性，不执行每年计息steps次凸性
    return conv


# 计算有效凸性，和不带1/2的凸性值对应
def effective_convexity(p, n, y, M, i, steps, deltay):
    """
    :param p: 价格
    :param n: 期限（年）
    :param y: 到期收益率
    :param M: 票面价值
    :param i: 息票率
    :param steps: 每年计息次数
    :param deltay: 收益率变化大小
    :return: 有效久期
    """
    # 计算收益率上升
    y1 = y + deltay
    p1 = calprice(n, y1, M, i, steps)
    # print(p1)
    # 计算收益率下降
    y2 = y - deltay
    p2 = calprice(n, y2, M, i, steps)
    # print(p2)
    ec = (p2 + p1 - 2 * p) / (p * deltay ** 2)  # 计算有效凸性
    return ec


# 使用凸性值+久期预测价格
def pre_price_conv(p, n, y, M, i, steps, deltay):
    new_p = calprice(n, y + deltay, M, i, steps)
    dm = modified_duration(new_p, n, y + deltay, M, i, steps, True)
    conv = convexity_measure(new_p, n, y + deltay, M, i, steps, True)
    pred_p = (-dm * deltay + 1 / 2 * conv * deltay ** 2) * p + p
    return pred_p


if __name__ == "__main__":
    # 看函数说明
    # y = calyeild(972.73,1,1000,0.07,1)
    # d = duration(972.73, 1, 0.09999692, 1000, 0.07, 1, True)
    # dm = modified_duration(100, 5, 0.09, 100, 0.09, 2, True)
    p = calprice(3, 0.035, 100, 0.06, 1)
    # c = convexity_measure(104.055,5,0.08,100,0.09,2,True)
    # ed = effective_duration(104.055,5,0.08,100,0.09,2,0.002)
    # ec = effective_convexity(104.055,5,0.08,100,0.09,2,0.001)
    # pre_p = pre_price_conv(104.055, 5, 0.08, 100, 0.09, 2, 0.01)
    print(p)
