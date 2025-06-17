import math
import matplotlib.pyplot as plt
  


def func(t):
    if t < 0:
        return 0
    else:
        return t


t_min = 0
t_max = 100
t_vals = [i for i in range(t_min, t_max)]
# print(t_vals)

# delays = np.linspace(t_min, t_max, num=5)
# delays = [0, 25, 50, 75, 100]
delays = [t_min, 50, t_max]
coeff = [1, 1]

post = [0] * len(t_vals)
for t in t_vals:
    for i in range(0, len(delays) - 1):
        post[t] += coeff[i]*func(t-delays[i])
plt.plot(t_vals, post, color='r')

# pre = post[delays[0]:delays[1]] + [0 for i in range(delays[1], t_max)]
# for i in range(1, len(delays) - 1):
#     for t in range(delays[i], delays[i+1]):
#         pre[t] = post[t]
#         for delay in delays[1:i+1]:
#             pre[t] -= pre[t-delay]
pre = [0 for i in range(t_min, t_max)]
for i in range(0, len(delays) - 1):
    for t in range(delays[i], delays[i+1]):
        pre[t] = post[t]/coeff[0]
        for j in range(1, i+1):
            pre[t] = pre[t] - coeff[j]*pre[t-delays[j]]/coeff[0]

plt.plot(t_vals, pre, color='b')
plt.savefig("output.png")
