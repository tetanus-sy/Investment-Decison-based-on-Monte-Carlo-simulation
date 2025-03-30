import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, triang

def monte_carlo_npv(num_simulations=10000, growth_dist='normal', seed=None):
    """
    蒙特卡洛模拟NPV（考虑增长率的概率分布）
    
    参数:
        num_simulations (int): 模拟次数
        growth_dist (str): 增长率分布类型 ('normal', 'uniform', 'triangular')
        seed (int): 随机种子（确保结果可重复）
        
    返回:
        dict: 包含统计指标和模拟结果的字典
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成服从正态分布的增长率,调整标准差使分布更对称
    growth_rates = np.random.normal(loc=0.04, scale=0.005, size=num_simulations)
    # 限制增长率的范围,避免极端值
    growth_rates = np.clip(growth_rates, 0.02, 0.06)
    
    # 预计算时间轴和折现因子
    t = np.arange(1, 41)
    discount_factors = (1.08) ** t
    
    # 计算每个增长率的NPV（向量化加速）
    npv_results = []
    for g in growth_rates:
        growth_factors = (1 + g) ** t
        cashflows = np.where(
            t <= 2,
            (33.6 * growth_factors - 2) / discount_factors,
            33.6 * growth_factors / discount_factors
        )
        npv = cashflows.sum()
        npv_results.append(npv)
    
    # 统计指标
    stats = {
        "mean": np.mean(npv_results),
        "std": np.std(npv_results),
        "min": np.min(npv_results),
        "max": np.max(npv_results),
        "5%_percentile": np.percentile(npv_results, 5),
        "95%_percentile": np.percentile(npv_results, 95),
        "prob_negative": np.mean(np.array(npv_results) < 0),
        "simulations": npv_results
    }
    return stats

def plot_results(stats):
    """可视化NPV分布"""
    plt.figure(figsize=(10, 6))
    # 增加bins数量使分布更平滑
    plt.hist(stats["simulations"], bins=100, density=True, alpha=0.7, color='skyblue')
    plt.axvline(stats["mean"], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {stats["mean"]:.2f}')
    plt.axvline(stats["5%_percentile"], color='gray', linestyle='dashed', linewidth=1, label='5% Percentile')
    plt.axvline(stats["95%_percentile"], color='gray', linestyle='dashed', linewidth=1, label='95% Percentile')
    plt.title(f'NPV Distribution (Monte Carlo: {len(stats["simulations"])} Simulations)')
    plt.xlabel('NPV')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 运行模拟
    simulations = 100000  # 模拟次数（建议至少1万次）
    results = monte_carlo_npv(num_simulations=simulations, seed=42)
    
    # 输出统计结果
    print(f"【统计结果】")
    print(f"均值 NPV: {results['mean']:.2f}")
    print(f"标准差: {results['std']:.2f}")
    print(f"最小值: {results['min']:.2f}, 最大值: {results['max']:.2f}")
    print(f"5%分位数: {results['5%_percentile']:.2f}, 95%分位数: {results['95%_percentile']:.2f}")
    print(f"负NPV概率: {results['prob_negative']:.2%}")
    
    # 可视化
    plot_results(results)