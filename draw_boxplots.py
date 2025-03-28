import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

def main():

    metrics_path = "metrics/output.csv"
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

    # df['Loss'] = df['Loss'].str.replace(',', '.').astype(float)
    # df['Dice'] = df['Dice'].str.replace(',', '.').astype(float)

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

        print(f"\nСтатистика по {metric}:")
        for model in model_order:
            model_data = df[df['Model'] == model][metric].dropna()
            if len(model_data) > 0:
                q1 = model_data.quantile(0.25)
                median = model_data.median()
                q3 = model_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                print(f"\n{model}:")
                print(f"  Медиана: {median:.3f}")
                print(f"  25-й перцентиль: {q1:.3f}")
                print(f"  75-й перцентиль: {q3:.3f}")
                print(f"  IQR: {iqr:.3f}")
                print(f"  Нижняя граница (без выбросов): {lower_bound:.3f}")
                print(f"  Верхняя граница (без выбросов): {upper_bound:.3f}")
                # print(f"  Минимум: {model_data.min():.3f}")
                # print(f"  Максимум: {model_data.max():.3f}")
            else:
                print(f"\n{model}: Нет данных")

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
        save_path = os.path.join(save_dir, f'{metric}_boxplot.jpg')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()