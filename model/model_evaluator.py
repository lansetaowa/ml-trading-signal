import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import pandas as pd

from scipy.stats import spearmanr

class RegressionEvaluator:

    @staticmethod
    def overall_ic(pred_df):
        """ Calculate and print overall IC
        Input:
        - pred_df: should contain columns "actuals" and "predicted"
        """
        lr_r, lr_p = spearmanr(pred_df.actuals, pred_df.predicted)
        print(f'Information Coefficient (overall): {lr_r:.3%} (p-value: {lr_p:.4%})')

    @staticmethod
    def plot_preds_scatter(pred_df, symbol=None):
        """ Prediction vs Actual Scatter Plot
        Input:
        - pred_df: should contain columns "actuals" and "predicted"
        """
        df = pred_df.copy()
        if symbol is not None:
            df = df.loc[df.index.get_level_values(0) == symbol]
        j = sns.jointplot(x='predicted', y='actuals',
                          robust=True, ci=None,
                          line_kws={'lw': 1, 'color': 'k'},
                          scatter_kws={'s': 1},
                          data=df,
                          kind='reg')
        j.ax_joint.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        j.ax_joint.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
        j.ax_joint.set_xlabel('Predicted')
        j.ax_joint.set_ylabel('Actuals')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_ic_distribution(score_df):
        """ plot IC distribution
        Input:
        - score_df: should contain columns "ic" and "rmse"
        """
        df = score_df.copy()
        ax = sns.distplot(score_df.ic)
        mean, median = df.ic.mean(), df.ic.median()
        ax.axvline(0, lw=1, ls='--', c='k')
        ax.text(x=.05, y=.9,
                s=f'Mean: {mean:8.2f}\nMedian: {median:5.2f}',
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel('Information Coefficient')
        sns.despine()
        plt.tight_layout()

    @staticmethod
    def plot_rolling_ic(score_df):
        """ Rolling IC trend
        Input:
        - score_df: should contain columns "ic" and "rmse"
        """
        df = score_df.copy()

        fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(14, 8))
        rolling_result = df.sort_index().rolling(7).mean().dropna()
        mean_ic = df.ic.mean()
        rolling_result.ic.plot(ax=axes[0],
                               title=f'Information Coefficient (Mean: {mean_ic:.2f})',
                               lw=1)
        axes[0].axhline(0, lw=.5, ls='-', color='k')
        axes[0].axhline(mean_ic, lw=1, ls='--', color='k')

        mean_rmse = df.rmse.mean()
        rolling_result.rmse.plot(ax=axes[1],
                                 title=f'Root Mean Squared Error (Mean: {mean_rmse:.2%})',
                                 lw=1,
                                 ylim=(0, df.rmse.max()))
        axes[1].axhline(df.rmse.mean(), lw=1, ls='--', color='k')
        sns.despine()
        plt.tight_layout()

    @staticmethod
    def plot_cumulative_returns_by_quantile(pred_df, n_bins=4):
        """
        按照 predicted 值的分位数，对 actuals 计算累计收益，并绘制不同 quantile 的收益曲线

        参数:
        - pred_df: pd.DataFrame，包含 'actuals' 和 'predicted' 两列，index 为时间
        - n_bins: int，分位数的数量（默认为 4）

        输出:
        - 绘图（matplotlib 折线图）
        """

        df = pred_df.copy()
        df = df.sort_index().dropna()
        df['predicted_quantile'] = pd.qcut(df['predicted'], q=n_bins, labels=False)

        cumulative_dict = {}
        for group_id, group_data in df.groupby('predicted_quantile'):
            group_data = group_data.sort_index()
            cum_return = (group_data['actuals'] + 1).cumprod()
            cumulative_dict[f'Quantile {group_id + 1}'] = cum_return

        cumulative_returns = pd.concat(cumulative_dict, axis=1)

        plt.figure(figsize=(12, 6))
        cumulative_returns.plot()
        plt.title('Cumulative Return by Predicted Quantile')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend(title='Quantile')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compute_return_stats_by_quantile(pred_df_all, target_cols, n_quantiles=4):
        """
        按 predicted 分组，计算多个 target 的 mean/median/std 并返回统计结果

        参数:
        - pred_df_all: DataFrame，包含 'predicted' 和若干 'target_xxh' 列
        - target_cols: List[str]，目标列名列表
        - n_quantiles: int，分组数量

        返回:
        - group_stats: 分组后的统计结果 DataFrame
        """
        df = pred_df_all.copy()
        df['quantile'] = pd.qcut(df['predicted'], q=n_quantiles, labels=False) + 1
        group_stats = df.groupby('quantile')[target_cols].agg(['mean', 'median', 'std'])
        return group_stats

    @staticmethod
    def plot_return_stats_by_quantile(group_stats):
        """
        绘制 mean 和 median 的柱状图，来自 compute_return_stats_by_quantile 的 group_stats 输出

        参数:
        - group_stats: 多重列索引的 DataFrame，包含 mean/median/std 信息
        """
        mean_data = group_stats[[col for col in group_stats.columns if col[1] == 'mean']]
        mean_data.columns = [col[0] for col in mean_data.columns]

        median_data = group_stats[[col for col in group_stats.columns if col[1] == 'median']]
        median_data.columns = [col[0] for col in median_data.columns]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        mean_data.plot(kind='bar', ax=axes[0])
        axes[0].set_title("Mean Returns by Prediction Quantile")
        axes[0].set_xlabel("Prediction Quantile")
        axes[0].set_ylabel("Mean Return")
        axes[0].grid(True)

        median_data.plot(kind='bar', ax=axes[1])
        axes[1].set_title("Median Returns by Prediction Quantile")
        axes[1].set_xlabel("Prediction Quantile")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compute_long_short_spread(pred_df_all, target_cols, n_quantiles=4):
        """
        计算每个目标列在最高分组和最低分组之间的 long-short spread

        参数:
        - pred_df_all: 包含 'predicted' 和 target_xh 列的 DataFrame
        - target_cols: List[str]，目标收益列
        - n_quantiles: int，预测值分组数量

        返回:
        - spread_df: 每列的 long-short spread DataFrame（包含 long, short, spread）
        """
        df = pred_df_all.copy()
        df['quantile'] = pd.qcut(df['predicted'], q=n_quantiles, labels=False) + 1
        long = df[df['quantile'] == n_quantiles]
        short = df[df['quantile'] == 1]

        spread_data = []
        for col in target_cols:
            long_mean = long[col].mean()
            short_mean = short[col].mean()
            spread = long_mean - short_mean
            spread_data.append({
                'target': col,
                'long_mean': long_mean,
                'short_mean': short_mean,
                'spread': spread
            })

        spread_df = pd.DataFrame(spread_data)
        return spread_df

    @staticmethod
    def plot_long_short_spread(spread_df):
        plt.figure(figsize=(8, 5))
        sns.barplot(data=spread_df, x='target', y='spread')
        plt.title("Long-Short Return Spread by Horizon")
        plt.ylabel("Spread (Long - Short)")
        plt.xlabel("Target Horizon")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
