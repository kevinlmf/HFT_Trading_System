"""
结果生成和导出模块
生成完整的交易分析报告（JSON、CSV、HTML、图表）
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端


class ResultGenerator:
    """
    结果生成器
    
    功能：
    - 生成JSON报告
    - 生成CSV数据
    - 生成HTML报告
    - 生成可视化图表
    - 导出交易建议
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_all_reports(self, result: Dict, prefix: str = "trading_analysis") -> Dict[str, str]:
        """
        生成所有报告
        
        Args:
            result: 完整流程结果
            prefix: 文件前缀
        
        Returns:
            生成的文件路径字典
        """
        files = {}
        
        # 1. JSON报告
        json_file = self.generate_json_report(result, prefix)
        files['json'] = json_file
        
        # 2. CSV数据
        csv_files = self.generate_csv_reports(result, prefix)
        files.update(csv_files)
        
        # 3. HTML报告
        html_file = self.generate_html_report(result, prefix)
        files['html'] = html_file
        
        # 4. 可视化图表
        chart_files = self.generate_charts(result, prefix)
        files.update(chart_files)
        
        # 5. 交易建议
        recommendation_file = self.generate_recommendation(result, prefix)
        files['recommendation'] = recommendation_file
        
        return files
    
    def generate_json_report(self, result: Dict, prefix: str) -> str:
        """生成JSON报告"""
        filename = self.output_dir / f"{prefix}_{self.timestamp}.json"
        
        # 准备JSON数据
        profile = result.get('pipeline_result', {}).get('profile', {})
        json_data = {
            'timestamp': self.timestamp,
            'data_quality': {
                'quality': getattr(profile, 'quality', {}).value if hasattr(getattr(profile, 'quality', None), 'value') else str(getattr(profile, 'quality', 'unknown')),
                'n_records': getattr(profile, 'n_records', 0),
                'memory_size_mb': getattr(profile, 'memory_size_mb', 0)
            },
            'execution': {
                'method': result.get('pipeline_result', {}).get('execution_result', {}).get('execution_info', {}).get('implementation', 'unknown'),
                'execution_time_ms': result.get('pipeline_result', {}).get('execution_result', {}).get('execution_info', {}).get('execution_time', 0) * 1000,
                'throughput': result.get('pipeline_result', {}).get('execution_result', {}).get('execution_info', {}).get('throughput', 0)
            },
            'strategy_comparison': self._extract_strategy_comparison(result),
            'position_management': self._extract_position_management(result),
            'final_recommendation': result.get('final_recommendation', {})
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ JSON报告已保存: {filename}")
        return str(filename)
    
    def generate_csv_reports(self, result: Dict, prefix: str) -> Dict[str, str]:
        """生成CSV报告"""
        files = {}
        
        # 1. 策略对比CSV
        comparison_results = result.get('comparison_results', [])
        if comparison_results:
            comparison_data = []
            for r in comparison_results:
                bt = r.backtest_result
                if hasattr(bt, 'equity_curve') and len(bt.equity_curve) > 0:
                    total_return = (bt.equity_curve.iloc[-1]/bt.equity_curve.iloc[0]-1)*100
                else:
                    total_return = 0
                
                comparison_data.append({
                    'Strategy': r.strategy_name,
                    'Score': r.overall_score,
                    'Sharpe_Ratio': r.sharpe_ratio,
                    'Total_Return': total_return,
                    'Max_Drawdown': r.risk_metrics.max_drawdown*100 if r.risk_metrics and hasattr(r.risk_metrics, 'max_drawdown') else 0,
                    'VaR_95': r.risk_metrics.var_95*100 if r.risk_metrics and hasattr(r.risk_metrics, 'var_95') else 0,
                    'CVaR_95': r.risk_metrics.cvar_95*100 if r.risk_metrics and hasattr(r.risk_metrics, 'cvar_95') else 0,
                    'Num_Trades': len(bt.trades) if hasattr(bt, 'trades') else 0,
                    'Risk_Checks_Passed': r.passed_risk_checks
                })
            
            df = pd.DataFrame(comparison_data)
            filename = self.output_dir / f"{prefix}_strategy_comparison_{self.timestamp}.csv"
            df.to_csv(filename, index=False)
            files['strategy_comparison'] = str(filename)
            print(f"✓ 策略对比CSV已保存: {filename}")
        
        # 2. 仓位配置CSV
        position_mgmt = result.get('position_management', {})
        if position_mgmt.get('optimal_weights'):
            position_data = []
            for symbol, weight in position_mgmt['optimal_weights'].items():
                position_data.append({
                    'Symbol': symbol,
                    'Weight': weight * 100,
                    'Position_Size': position_mgmt.get('position_sizes', {}).get(symbol, 0)
                })
            
            df = pd.DataFrame(position_data)
            filename = self.output_dir / f"{prefix}_positions_{self.timestamp}.csv"
            df.to_csv(filename, index=False)
            files['positions'] = str(filename)
            print(f"✓ 仓位配置CSV已保存: {filename}")
        
        return files
    
    def generate_html_report(self, result: Dict, prefix: str) -> str:
        """生成HTML报告"""
        filename = self.output_dir / f"{prefix}_report_{self.timestamp}.html"
        
        html_content = self._generate_html_content(result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML报告已保存: {filename}")
        return str(filename)
    
    def generate_charts(self, result: Dict, prefix: str) -> Dict[str, str]:
        """生成可视化图表"""
        files = {}
        
        # 1. 策略对比图表
        comparison_results = result.get('comparison_results', [])
        if comparison_results:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Strategy Comparison', fontsize=16, fontweight='bold')
            
            # 提取数据
            strategies = [r.strategy_name for r in comparison_results]
            scores = [r.overall_score for r in comparison_results]
            sharpes = [r.sharpe_ratio for r in comparison_results]
            returns = []
            drawdowns = []
            for r in comparison_results:
                if hasattr(r.backtest_result, 'equity_curve') and len(r.backtest_result.equity_curve) > 0:
                    ret = (r.backtest_result.equity_curve.iloc[-1]/r.backtest_result.equity_curve.iloc[0]-1)*100
                else:
                    ret = 0
                returns.append(ret)
                
                if r.risk_metrics and hasattr(r.risk_metrics, 'max_drawdown'):
                    drawdowns.append(r.risk_metrics.max_drawdown*100)
                else:
                    drawdowns.append(0)
            
            # 综合评分
            axes[0, 0].barh(strategies, scores)
            axes[0, 0].set_xlabel('Overall Score')
            axes[0, 0].set_title('Overall Score Comparison')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Sharpe Ratio
            axes[0, 1].barh(strategies, sharpes)
            axes[0, 1].set_xlabel('Sharpe Ratio')
            axes[0, 1].set_title('Sharpe Ratio Comparison')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 收益率
            axes[1, 0].barh(strategies, returns)
            axes[1, 0].set_xlabel('Total Return (%)')
            axes[1, 0].set_title('Return Comparison')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 最大回撤
            axes[1, 1].barh(strategies, drawdowns)
            axes[1, 1].set_xlabel('Max Drawdown (%)')
            axes[1, 1].set_title('Risk Comparison')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            chart_file = self.output_dir / f"{prefix}_strategy_comparison_{self.timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            files['strategy_chart'] = str(chart_file)
            print(f"✓ 策略对比图表已保存: {chart_file}")
        
        # 2. 仓位配置饼图
        position_mgmt = result.get('position_management', {})
        if position_mgmt.get('optimal_weights'):
            weights = position_mgmt['optimal_weights']
            if weights:
                fig, ax = plt.subplots(figsize=(10, 8))
                symbols = list(weights.keys())
                values = [weights[s] * 100 for s in symbols]
                
                ax.pie(values, labels=symbols, autopct='%1.1f%%', startangle=90)
                ax.set_title('Optimal Position Allocation', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                chart_file = self.output_dir / f"{prefix}_positions_{self.timestamp}.png"
                plt.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close()
                files['position_chart'] = str(chart_file)
                print(f"✓ 仓位配置图表已保存: {chart_file}")
        
        # 3. 权益曲线对比
        if comparison_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            for r in comparison_results[:5]:  # 只显示前5个
                if hasattr(r.backtest_result, 'equity_curve') and len(r.backtest_result.equity_curve) > 0:
                    equity = r.backtest_result.equity_curve / r.backtest_result.equity_curve.iloc[0]
                    ax.plot(range(len(equity)), equity.values, label=r.strategy_name, linewidth=2)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Normalized Equity')
            ax.set_title('Equity Curve Comparison', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            chart_file = self.output_dir / f"{prefix}_equity_curves_{self.timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            files['equity_chart'] = str(chart_file)
            print(f"✓ 权益曲线图表已保存: {chart_file}")
        
        return files
    
    def generate_recommendation(self, result: Dict, prefix: str) -> str:
        """生成交易建议文件"""
        filename = self.output_dir / f"{prefix}_recommendation_{self.timestamp}.txt"
        
        recommendation = result.get('final_recommendation', {})
        position_mgmt = result.get('position_management', {})
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TRADING RECOMMENDATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 推荐策略
            if recommendation.get('strategy'):
                f.write(f"Recommended Strategy: {recommendation['strategy']}\n")
                f.write(f"Risk Checks: {'PASSED' if recommendation.get('passed_all_checks') else 'FAILED'}\n\n")
            
            # 仓位配置
            f.write("Position Allocation:\n")
            f.write("-"*80 + "\n")
            optimal_weights = position_mgmt.get('optimal_weights', {})
            position_sizes = position_mgmt.get('position_sizes', {})
            
            for symbol, weight in sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True):
                size = position_sizes.get(symbol, 0)
                f.write(f"{symbol:10s}  Weight: {weight*100:6.2f}%  Size: {size:10.2f} shares\n")
            
            # 风险指标
            risk_metrics = position_mgmt.get('risk_metrics', {})
            if risk_metrics:
                f.write("\nPortfolio Risk Metrics:\n")
                f.write("-"*80 + "\n")
                f.write(f"Volatility:     {risk_metrics.get('volatility', 0)*100:6.2f}%\n")
                f.write(f"VaR (95%):       {abs(risk_metrics.get('var_95', 0))*100:6.2f}%\n")
                f.write(f"CVaR (95%):      {abs(risk_metrics.get('cvar_95', 0))*100:6.2f}%\n")
                f.write(f"Max Drawdown:    {abs(risk_metrics.get('max_drawdown', 0))*100:6.2f}%\n")
                if risk_metrics.get('sharpe_ratio'):
                    f.write(f"Sharpe Ratio:    {risk_metrics.get('sharpe_ratio', 0):6.3f}\n")
            
            # 风险违规
            violations = position_mgmt.get('violations', [])
            if violations:
                f.write("\n⚠ Risk Violations:\n")
                f.write("-"*80 + "\n")
                for violation in violations:
                    f.write(f"  - {violation}\n")
            else:
                f.write("\n✓ All risk checks passed\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ 交易建议已保存: {filename}")
        return str(filename)
    
    def _extract_strategy_comparison(self, result: Dict) -> List[Dict]:
        """提取策略对比数据"""
        comparison_results = result.get('comparison_results', [])
        data = []
        for r in comparison_results:
            bt = r.backtest_result
            if hasattr(bt, 'equity_curve') and len(bt.equity_curve) > 0:
                total_return = (bt.equity_curve.iloc[-1]/bt.equity_curve.iloc[0]-1)*100
            else:
                total_return = 0
            
            data.append({
                'strategy_name': r.strategy_name,
                'overall_score': r.overall_score,
                'sharpe_ratio': r.sharpe_ratio,
                'total_return': total_return,
                'max_drawdown': r.risk_metrics.max_drawdown*100 if r.risk_metrics and hasattr(r.risk_metrics, 'max_drawdown') else 0,
                'var_95': r.risk_metrics.var_95*100 if r.risk_metrics and hasattr(r.risk_metrics, 'var_95') else 0,
                'cvar_95': r.risk_metrics.cvar_95*100 if r.risk_metrics and hasattr(r.risk_metrics, 'cvar_95') else 0,
                'passed_risk_checks': r.passed_risk_checks
            })
        return data
    
    def _extract_position_management(self, result: Dict) -> Dict:
        """提取仓位管理数据"""
        position_mgmt = result.get('position_management', {})
        return {
            'optimal_weights': position_mgmt.get('optimal_weights', {}),
            'position_sizes': position_mgmt.get('position_sizes', {}),
            'risk_metrics': position_mgmt.get('risk_metrics', {}),
            'violations': position_mgmt.get('violations', [])
        }
    
    def _generate_html_content(self, result: Dict) -> str:
        """生成HTML内容"""
        recommendation = result.get('final_recommendation', {})
        position_mgmt = result.get('position_management', {})
        comparison_results = result.get('comparison_results', [])
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Trading Analysis Report - {self.timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px; background: #ecf0f1; border-radius: 5px; }}
        .metric-label {{ font-weight: bold; color: #7f8c8d; }}
        .metric-value {{ font-size: 24px; color: #2c3e50; }}
        .pass {{ color: #27ae60; font-weight: bold; }}
        .fail {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Trading Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Recommended Strategy</h2>
        <div class="metric">
            <div class="metric-label">Strategy</div>
            <div class="metric-value">{recommendation.get('strategy', 'N/A')}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Risk Checks</div>
            <div class="metric-value {'pass' if recommendation.get('passed_all_checks') else 'fail'}">
                {'PASSED' if recommendation.get('passed_all_checks') else 'FAILED'}
            </div>
        </div>
        
        <h2>Strategy Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Strategy</th>
                    <th>Score</th>
                    <th>Sharpe</th>
                    <th>Return</th>
                    <th>Max DD</th>
                    <th>Risk</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for i, r in enumerate(comparison_results, 1):
            bt = r.backtest_result
            if hasattr(bt, 'equity_curve') and len(bt.equity_curve) > 0:
                total_return = (bt.equity_curve.iloc[-1]/bt.equity_curve.iloc[0]-1)*100
            else:
                total_return = 0
            max_dd = r.risk_metrics.max_drawdown*100 if r.risk_metrics and hasattr(r.risk_metrics, 'max_drawdown') else 0
            risk_status = 'PASS' if r.passed_risk_checks else 'FAIL'
            
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{r.strategy_name}</td>
                    <td>{r.overall_score:.2f}</td>
                    <td>{r.sharpe_ratio:.3f}</td>
                    <td>{total_return:.2f}%</td>
                    <td>{max_dd:.2f}%</td>
                    <td class="{'pass' if r.passed_risk_checks else 'fail'}">{risk_status}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <h2>Position Allocation</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Weight</th>
                    <th>Position Size</th>
                </tr>
            </thead>
            <tbody>
"""
        
        optimal_weights = position_mgmt.get('optimal_weights', {})
        position_sizes = position_mgmt.get('position_sizes', {})
        for symbol, weight in sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True):
            size = position_sizes.get(symbol, 0)
            html += f"""
                <tr>
                    <td>{symbol}</td>
                    <td>{weight*100:.2f}%</td>
                    <td>{size:.2f}</td>
                </tr>
"""
        
        risk_metrics = position_mgmt.get('risk_metrics', {})
        if risk_metrics:
            html += """
            </tbody>
        </table>
        
        <h2>Portfolio Risk Metrics</h2>
        <div class="metric">
            <div class="metric-label">Volatility</div>
            <div class="metric-value">""" + f"{risk_metrics.get('volatility', 0)*100:.2f}%" + """</div>
        </div>
        <div class="metric">
            <div class="metric-label">VaR (95%)</div>
            <div class="metric-value">""" + f"{abs(risk_metrics.get('var_95', 0))*100:.2f}%" + """</div>
        </div>
        <div class="metric">
            <div class="metric-label">CVaR (95%)</div>
            <div class="metric-value">""" + f"{abs(risk_metrics.get('cvar_95', 0))*100:.2f}%" + """</div>
        </div>
        <div class="metric">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value">""" + f"{abs(risk_metrics.get('max_drawdown', 0))*100:.2f}%" + """</div>
        </div>
"""
        
        html += """
        </div>
    </div>
</body>
</html>
"""
        return html

