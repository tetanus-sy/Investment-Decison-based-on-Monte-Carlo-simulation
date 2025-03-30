import numpy as np
import matplotlib.pyplot as plt

def simulate_npv(num_simulations=10000, seed=None):
    """蒙特卡洛模拟NPV"""
    if seed is not None:
        np.random.seed(seed)
    
    # 时间参数
    t = np.arange(1, 41)  # 时间轴 t=1到40年
    
    # 预计算Z_t的参数（向量化）
    mu_Z = 30 * (1.05)**t + 14 * (1.06)**t
    sigma_Z = np.sqrt(
        (1.73*(1.05)**t)**2 + 
        (2.5*(1.06)**t)**2 + 
        1.73*2.5*(1.05**t)*(1.06**t) ) # 注意协方差项
    
    # 折现因子
    discount_factors = (1.08)**t
    
    # 生成Z_t的随机样本 (num_simulations × 40)
    Z = np.random.normal(loc=mu_Z, scale=sigma_Z, size=(num_simulations, 40))
    
    # 计算现金流现值
    # 前两年现金流
    cashflow_part1 = (Z[:, 0] - 8.042)/discount_factors[0] + (Z[:, 1] - 8.042)/discount_factors[1]
    
    # 第三年至第四十年现金流
    cashflow_part2 = np.sum((Z[:, 2:] - 2.2) / discount_factors[2:], axis=1)
    
    # 终值
    terminal = 2 / discount_factors[-1]
    
    # 初始成本C0
    C0 = np.random.normal(46.5, 5, num_simulations)
    
    # 总NPV
    npv = cashflow_part1 + cashflow_part2 + terminal - C0
    
    return npv

def plot_results(npv_samples):
    """可视化NPV分布"""
    # 计算统计值
    mean_npv = np.mean(npv_samples)
    percentile_5 = np.percentile(npv_samples, 5)
    percentile_95 = np.percentile(npv_samples, 95)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(npv_samples, bins=50, density=True, alpha=0.7, color='skyblue')
    plt.axvline(mean_npv, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {mean_npv:.2f}')
    plt.axvline(percentile_5, color='gray', linestyle='dashed', linewidth=1, 
                label='5% Percentile')
    plt.axvline(percentile_95, color='gray', linestyle='dashed', linewidth=1, 
                label='95% Percentile')
    plt.title(f'NPV Distribution (Monte Carlo: {len(npv_samples)} Simulations)')
    plt.xlabel('NPV')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

# 运行模拟
if __name__ == "__main__":
    # 运行模拟
    npv_samples = simulate_npv(num_simulations=100000, seed=42)
    
    # 统计结果
    print(f"【统计结果】")
    print(f"均值 NPV: {np.mean(npv_samples):.2f}")
    print(f"标准差: {np.std(npv_samples):.2f}")
    print(f"最小值: {np.min(npv_samples):.2f}, 最大值: {np.max(npv_samples):.2f}")
    print(f"5%分位数: {np.percentile(npv_samples, 5):.2f}, 95%分位数: {np.percentile(npv_samples, 95):.2f}")
    print(f"负NPV概率: {np.mean(npv_samples < 0):.2%}")
    
    # 可视化
    plot_results(npv_samples)