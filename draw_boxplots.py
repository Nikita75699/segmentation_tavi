import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

def main():

    metrics_path = "wandb_export_2025-02-12T14_24_03.776+07_00.csv"
    save_dir = "save_graph"
    os.makedirs(save_dir, exist_ok=True)

    # Read DataFrame with metrics
    df = pd.read_csv(metrics_path)

    # Проверяем, существуют ли нужные колонки перед преобразованием
    required_columns = {'Loss', 'Dice', 'Model', 'Class'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Ошибка: В файле отсутствуют колонки {missing_columns}")
        return

    df['Loss'] = df['Loss'].str.replace(',', '.').astype(float)
    df['Dice'] = df['Dice'].str.replace(',', '.').astype(float)

    # Define y-limits for each class
    y_limits = {
        'aorta': (0.6, 1),
    }

    # Define the order of x-axis categories
    model_order = [
        'U-Net++',
        'LinkNet',
        'FPN',
        'PSPNet',
        'DeepLabV3+',
        'MA-Net',
    ]

    # Проверяем, есть ли 'Model' в данных
    if 'Model' not in df.columns:
        print("Ошибка: 'Model' не найден в CSV-файле.")
        return

    # Plotting
    sns.set(style='whitegrid')

    for metric in ['Loss', 'Dice']:  # Оставляем только нужные графики
        print(f"Обрабатываем: {metric}")
        plt.figure(figsize=(12, 12))

        # Создаем палитру под количество моделей
        palette = sns.color_palette('muted', n_colors=len(model_order))

        ax = sns.boxplot(
            x='Model',
            y=metric,
            data=df,
            palette=palette,
            hue='Model',  # Чтобы избежать ошибки в Seaborn 0.14+
            legend=False,
            showfliers=False,  # Убираем выбросы
            order=model_order,
            linewidth=2.0,
        )

        plt.ylabel(metric, fontsize=36)
        plt.xticks(rotation=90, fontsize=30)
        plt.yticks(fontsize=30)
        ax.set_xlabel('')

        # Применяем y-границы
        if metric in y_limits:
            ax.set_ylim(y_limits[metric])
            ax.yaxis.set_major_locator(MultipleLocator(0.1))

        sns.despine()
        plt.tight_layout()

        # Сохраняем график
        save_path = os.path.join(save_dir, f'{metric}_boxplot.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()