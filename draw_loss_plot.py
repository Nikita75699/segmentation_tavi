import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():

    # Define absolute paths
    metrics_path = "wandb_export_2025-02-12T14_24_03.776+07_00.csv"
    save_dir = "save_graph"
    os.makedirs(save_dir, exist_ok=True)

    # Read DataFrame with metrics
    df = pd.read_csv(metrics_path)

    df['Loss'] = df['Loss'].str.replace(',', '.').astype(float)
    df['Dice'] = df['Dice'].str.replace(',', '.').astype(float)

    gb = df.groupby('model')

    for model_name, df_model in gb:
        # Filter for the 'Mean' class
        df_filt = df_model[df_model['Class'] == 'aorta']
        # Plot
        sns.set(style='whitegrid')
        plt.figure(figsize=(12, 10))

        # Customize color palette
        palette = sns.color_palette('bright', 2)

        # Draw line plots with confidence intervals
        sns.lineplot(
            data=df_filt[df_filt['Split'] == 'test'],
            x='Epoch',
            y='Loss',
            color=palette[0],
            linewidth=3.0,
            label='Loss (Test)',
            err_style='band',
            errorbar=('ci', 95),
        )
        sns.lineplot(
            data=df_filt[df_filt['Split'] == 'test'],
            x='Epoch',
            y='Dice',
            color=palette[1],
            linewidth=3.0,
            label='DSC (Test)',
            err_style='band',
            errorbar=('ci', 95),
        )

        plt.xlabel('Epoch', fontsize=36)
        plt.ylabel('Metric Value', fontsize=36)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=26, loc='center right')
        plt.grid(True)

        # Set coordinate axis limits
        plt.ylim(0, 1)
        plt.xlim(0, 30)
        plt.tight_layout(pad=0.9)

        # Save plot
        save_path = os.path.join(save_dir, f'{model_name}_loss.png')
        plt.savefig(save_path, dpi=600)
        plt.show()


if __name__ == '__main__':
    main()