import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
from src.visualization.visualize import plot_roc

method_dict = {
    'true': {
        'name': 'GT',
        'style': (0, (3, 1, 1, 1, 1, 1)),
    },
    'rand': {
        'name': 'RND',
        'style': 'dotted',
    },
    'len': {
        'name': 'LEN',
        'style': 'dashdot',
    },
    'CPPOD': {
        'name': 'CPPOD',
        'style': 'solid',
    },
}
methods = ['rand', 'CPPOD', 'len', 'true']
# datasets = ['pois', 'gam']
datasets = ['pois']
outliers = ['commiss', 'omiss']
curves = [
    {
        'func': plot_roc,
        'name': 'AUROC',
        'x': 'FPR',
        'y': 'TPR',
        'filename': 'roc',
    },
]
plot = True
matplotlib.rcParams.update({'font.size': 6})
pc = ['0.1']


def get_metrics():
    results = []
    figures = []
    for dataset in datasets:
        for outlier in outliers:
            folder = f'results/{dataset}/{outlier}'
            for p in pc:
                for curve in curves:
                    fig = plt.figure(figsize=(10, 6))
                    for method in methods:
                        df = pd.read_csv(f'{folder}/{method}_{p}.csv')
                        auc = curve['func'](df['label'], df[f'score_{outlier}'], method_dict[method], plot)
    
                        results.append(OrderedDict({
                            'dataset': f'{dataset}_{p}',
                            'outlier': outlier,
                            'method': method_dict[method]['name'],
                            'metric': curve['name'],
                            'value': auc,
                        }))
                    plt.xlabel(curve['x'])
                    plt.ylabel(curve['y'])
                    plt.legend()
                    filename = curve['filename']
                    figures.append((fig, f'results/fig/{filename}_{dataset}_{p}_{outlier}.pdf', outlier))
    return figures, results


if __name__ == '__main__':
    figures, results = get_metrics()
    pd.DataFrame(results).to_csv('metrics/metrics.csv', index=False)
    for fig, path, _ in figures:
        fig.savefig(path, bbox_inches='tight')
