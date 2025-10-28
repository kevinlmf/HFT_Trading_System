"""
Visualize Throughput Benchmark Results
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 获取脚本目录并构建结果路径
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(os.path.dirname(script_dir), 'results', 'benchmarks')
csv_path = os.path.join(results_dir, 'benchmark_results.csv')

# 读取结果
df = pd.read_csv(csv_path)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('HFT System Throughput Benchmark Results', fontsize=16, fontweight='bold')

# 1. 吞吐量对比
ax1 = axes[0, 0]
throughput = df['Throughput (ops/s)'] / 1000  # Convert to K ops/sec
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
bars = ax1.barh(df['Test Name'], throughput, color=colors)
ax1.set_xlabel('Throughput (K ops/sec)', fontweight='bold')
ax1.set_title('Throughput Comparison', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, throughput)):
    if val > 1000:
        label = f'{val/1000:.1f}M'
    else:
        label = f'{val:.0f}K'
    ax1.text(val, bar.get_y() + bar.get_height()/2, label,
             ha='left', va='center', fontweight='bold', fontsize=9)

# 2. 延迟分布
ax2 = axes[0, 1]
latency_data = df[['Latency P50 (μs)', 'Latency P95 (μs)', 'Latency P99 (μs)']].values
x = np.arange(len(df))
width = 0.25

bars1 = ax2.bar(x - width, latency_data[:, 0], width, label='P50', color='#3498db', alpha=0.8)
bars2 = ax2.bar(x, latency_data[:, 1], width, label='P95', color='#f39c12', alpha=0.8)
bars3 = ax2.bar(x + width, latency_data[:, 2], width, label='P99', color='#e74c3c', alpha=0.8)

ax2.set_ylabel('Latency (μs)', fontweight='bold')
ax2.set_title('Latency Distribution', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([name[:15] + '...' if len(name) > 15 else name
                      for name in df['Test Name']], rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_yscale('log')

# 3. 数据处理量
ax3 = axes[1, 0]
data_points = df['Data Points'] / 1_000_000  # Convert to millions
bars = ax3.barh(df['Test Name'], data_points, color=colors)
ax3.set_xlabel('Data Points (Millions)', fontweight='bold')
ax3.set_title('Total Data Points Processed', fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars, data_points):
    label = f'{val:.2f}M' if val < 1 else f'{val:.1f}M'
    ax3.text(val, bar.get_y() + bar.get_height()/2, label,
             ha='left', va='center', fontweight='bold', fontsize=9)

# 4. 性能总结表
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

# 创建总结统计
summary_data = []
for idx, row in df.iterrows():
    throughput_str = f"{row['Throughput (ops/s)']/1000:.0f}K" if row['Throughput (ops/s)'] < 1e6 else f"{row['Throughput (ops/s)']/1e6:.1f}M"
    summary_data.append([
        row['Test Name'][:20],
        throughput_str + ' ops/s',
        f"{row['Latency P50 (μs)']:.2f} μs",
        f"{row['Elapsed Time (s)']:.2f}s"
    ])

table = ax4.table(cellText=summary_data,
                  colLabels=['Test', 'Throughput', 'P50 Latency', 'Time'],
                  cellLoc='left',
                  loc='center',
                  colWidths=[0.35, 0.25, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# 设置表头样式
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置行颜色
for i in range(1, len(summary_data) + 1):
    for j in range(4):
        table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

ax4.set_title('Performance Summary', fontweight='bold', pad=20)

plt.tight_layout()
output_path = os.path.join(results_dir, 'throughput_benchmark.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"📊 Visualization saved to {output_path}")
plt.close()

# 打印关键性能指标
print("\n" + "="*70)
print("📈 KEY PERFORMANCE INDICATORS")
print("="*70)

print("\n🚀 THROUGHPUT (Higher is better):")
for _, row in df.iterrows():
    throughput = row['Throughput (ops/s)']
    if throughput >= 1e6:
        print(f"   {row['Test Name']:30s} {throughput/1e6:>8.2f} M ops/sec")
    else:
        print(f"   {row['Test Name']:30s} {throughput/1e3:>8.2f} K ops/sec")

print("\n⚡ LATENCY P99 (Lower is better):")
for _, row in df.iterrows():
    latency = row['Latency P99 (μs)']
    if latency >= 1000:
        print(f"   {row['Test Name']:30s} {latency/1000:>8.2f} ms")
    else:
        print(f"   {row['Test Name']:30s} {latency:>8.2f} μs")

print("\n💪 SYSTEM CAPABILITIES:")
total_ops = df['Data Points'].sum()
total_time = df['Elapsed Time (s)'].sum()
avg_throughput = total_ops / total_time

print(f"   Total Operations:     {total_ops:>15,}")
print(f"   Total Time:           {total_time:>15.2f} seconds")
print(f"   Average Throughput:   {avg_throughput:>15,.0f} ops/sec")

# 最快和最慢的组件
fastest = df.loc[df['Throughput (ops/s)'].idxmax()]
slowest = df.loc[df['Throughput (ops/s)'].idxmin()]

print(f"\n   🏆 Fastest Component:  {fastest['Test Name']}")
print(f"      Throughput:        {fastest['Throughput (ops/s)']/1e6:.2f} M ops/sec")

print(f"\n   🐌 Slowest Component:  {slowest['Test Name']}")
print(f"      Throughput:        {slowest['Throughput (ops/s)']/1e3:.2f} K ops/sec")

print("\n" + "="*70)
