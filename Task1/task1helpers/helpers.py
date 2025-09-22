import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

def npv(cashflows, r):
    """Compute NPV given annual cashflows where index 0 is year 0."""
    return sum(cf / ((1 + r) ** t) for t, cf in enumerate(cashflows))



def generate_boxplot(df,
                     title="Sensitivity Analysis: Average NPV by Parameter Deciles",
                     hue=None,
                     reference_points=None,
                     plot_configs=None):

    # Default plot configurations if none provided
    if plot_configs is None:
        plot_configs = [
            {'col': 'mean_vc', 'title': 'Variable Cost Sensitivity', 'xlabel': 'Variable Cost Decile'},
            {'col': 'mean_vol', 'title': 'Volume Sensitivity', 'xlabel': 'Volume Decile'},
            {'col': 'mean_salvage', 'title': 'Salvage Value Sensitivity', 'xlabel': 'Salvage Value Decile'},
            {'col': 'mean_fx', 'title': 'FX Rate Sensitivity', 'xlabel': 'FX Rate Decile'}
        ]

    df_work = df.copy()

    # Create decile columns
    for config in plot_configs:
        col_name = f"{config['col'].replace('mean_', '').title()} Decile"
        if hue is not None:
            df_work[col_name] = df_work.groupby(hue)[config['col']].transform(
                lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
            )
        else:
            df_work[col_name] = pd.qcut(df_work[config['col']], 10, labels=False, duplicates='drop')

    df_work['npv_millions'] = df_work['npv'] / 1_000_000

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharey=True)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Generate colors for reference lines if provided
    if reference_points is not None:
        reference_colors = plt.cm.Set1(np.linspace(0, 1, len(reference_points)))

    all_handles, all_labels = None, None

    for i, config in enumerate(plot_configs):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        decile_col = f"{config['col'].replace('mean_', '').title()} Decile"

        if hue is not None:
            sns.boxplot(x=decile_col, y='npv_millions', data=df_work,
                        ax=ax, hue=hue)
        else:
            colors = sns.color_palette("viridis", n_colors=10)
            sns.boxplot(x=decile_col, y='npv_millions', data=df_work,
                        ax=ax, hue=decile_col, palette=colors, legend=False)

        ax.set_title(config['title'], fontweight='bold', pad=20)
        ax.set_xlabel(config['xlabel'])
        if col == 0:
            ax.set_ylabel('NPV ($ Millions)')

        # Add reference lines if provided
        if reference_points is not None:
            for j, (ref_name, ref_value) in enumerate(reference_points.items()):
                ax.axhline(y=ref_value / 1_000_000, color=reference_colors[j],
                           linestyle='--', alpha=0.7, linewidth=2, label=ref_name)

        # Grab legend info from the first axis only
        if all_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            all_handles, all_labels = handles, labels

        ax.legend_.remove() if ax.get_legend() else None
        ax.grid(True, alpha=0.3)

    # Shared legend at bottom
    fig.legend(all_handles, all_labels,
               loc='lower center', ncol=4, fontsize=10, frameon=False)

    for ax in axes.flat:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}M'))
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()