# Задача 1. Даны значения величины заработной платы заемщиков банка (zp) и значения их поведенческого кредитного скоринга (ks):
# zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110],
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832].
# Используя математ.операции, посчитать коэффициенты линейной регрессии, приняв за Х заработную плату (т.е. zp - признак), а за у значения скорингового балла (т.е. ks - целевая переменная).
# Произвести расчет как с использованием intercept, так и без.
# Посчитать коэффициент линейной регрессии при заработной плате (zp), используя градиентный спуск (без intercept).

# расчет с использованием intercept
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

# y = a + bx
n1 = len(zp)
n2 = len(ks)
print("Длина массивов:", n1, n2)

b = (n1*np.sum(zp*ks) - np.sum(zp)*np.sum(ks)) / \
    (n1*np.sum(zp**2) - np.sum(zp)**2)
print("Угол наклона прямой =", b)
a = np.mean(ks) - b*np.mean(zp)
print("Интерсепт=", a)
r = np.corrcoef(zp, ks)
r_2 = r[1, 0]**2
ks_pred = a + b*zp
print("Коэффициенты линейной регрессии:", ks_pred)
plt.scatter(zp, ks)
plt.plot(zp, ks_pred)

# или вторым способом

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

# y = a + bx
n1 = len(zp)
n2 = len(ks)

zp_shape = zp.reshape(len(zp), 1)
ks_shape = ks.reshape(len(ks), 1)
model = LinearRegression()
regres = model.fit(zp_shape, ks_shape)
r_sq = model.score(zp_shape, ks_shape)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

# или третьим способом

zp = np.vstack([np.ones((1, len(zp))), zp])
print("Коэффициенты:", np.dot(np.dot(np.linalg.inv(np.dot(zp, zp.T)), zp), ks.T))

# расчет без интерсепта - 1ый способ

# x=sm.add_constant(zp)
# model = sm.OLS(ks, zp)
# result = model.fit()
# print(result.summary())

# второй способ без интерсепта
zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])
zp_r = zp.reshape(10, 1)
ks_r = ks.reshape(10, 1)
print(zp_r)
print(ks_r)
ZP = zp_r.T.dot(zp_r)
ZP_inv = np.linalg.inv(ZP)
b = ZP_inv.dot(zp_r.T).dot(ks_r)
print(b)

# Посчитать коэффициент линейной регрессии при заработной плате (zp), используя градиентный спуск (без intercept).
n = len(zp)


def mse_(B1, y=ks, x=zp, n=10):
    return np.sum((B1*x - y) ** 2) / n


alpha = 1e-6
B1 = 0.1
for i in range(3000):
    B1 = alpha * (2/n) * np.sum(2 * (B1 * zp - ks) * zp)
    if i % 500 == 0:
        print('Итерация = {i}, B1 = {B1}, mse ={mse}'.format(
            i=i, B1=B1, mse=mse_(B1)))

print(mse_(-0.31905616))
