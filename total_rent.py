import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_npv(num_simulations=10000, seed=None):
    """
    蒙特卡洛模拟计算NPV（考虑利润的正态分布）
    
    参数:
        x (float): 终值相关参数
        num_simulations (int): 模拟次数
        seed (int): 随机种子（确保结果可重复）
        
    返回:
        dict: 包含统计指标和模拟结果的字典
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 基础参数
    discount_rate = 0.08
    growth_rate = 0.05
    t = np.arange(1, 41)  # 时间轴 t=1到40年
    
    # 预计算趋势和标准差
    trend = 30 * (1 + growth_rate) ** t
    std_dev = 1.73 * (1 + growth_rate) ** t
    
    # 预计算折现因子
    discount_factors = (1 + discount_rate) ** t
    
    # 初始化结果存储
    npv_results = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        # 生成随机利润（向量化操作）
        p_y = np.random.normal(loc=trend, scale=std_dev)
        
        # 计算各期现金流（分段处理）
        cashflows = np.where(
            t <= 2,
            (p_y - 8.042) / discount_factors,
            (p_y - 2.2) / discount_factors
        )
        
        # 终值计算
        terminal = 2 / (1 + discount_rate)**40 - 16.5
        
        # 总NPV = 初始值60 + 各期现金流之和 + 终值
        npv_results[i] = 60 + cashflows.sum() + terminal
    
    # 统计指标
    stats = {
        "mean": np.mean(npv_results),
        "std": np.std(npv_results),
        "min": np.min(npv_results),
        "max": np.max(npv_results),
        "5%_percentile": np.percentile(npv_results, 5),
        "95%_percentile": np.percentile(npv_results, 95),
        "prob_negative": np.mean(npv_results < 0),
        "simulations": npv_results
    }
    return stats

def plot_results(stats):
    """可视化NPV分布"""
    plt.figure(figsize=(10, 6))
    plt.hist(stats["simulations"], bins=50, density=True, alpha=0.7, color='skyblue')
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
    # 参数设置         
    simulations = 100000  # 模拟次数（建议至少1万次）
    
    # 运行模拟
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