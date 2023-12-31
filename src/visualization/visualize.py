import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics


def plot_events(seq, title='Event Visualization'):
    """Plots a visual representation of events in a sequence.

    Args:
        seq (dict): A dictionary containing the sequence of events.
        title (str, optional): The title of the plot. Defaults to 'Event Visualization'.
    """
    vt_event = seq['time_target']
    lambda_x = seq['lambda_x']
    t_x = seq['t_x']
    vt_omiss = seq.get('time_omiss')

    scale = 0.25 * np.max(lambda_x)

    plt.figure(figsize=(10, 6))

    plt.plot(t_x, lambda_x, label='lambda_x')
    plt.xlabel('Time (t_x)')
    plt.ylabel('Intensity (lambda_x)')

    if vt_omiss is None:
        vlabel = seq.get('label_test')
        if vlabel is None:
            plt.stem(vt_event, scale * np.ones_like(vt_event), linefmt='k-', markerfmt='ko', label='vt_event')
        else:
            plt.stem(vt_event[vlabel == 0], scale * np.ones_like(vt_event[vlabel == 0]), linefmt='k-', markerfmt='ko', label='vt_event')
            if any(vlabel):
                plt.stem(vt_event[vlabel == 1], scale * np.ones_like(vt_event[vlabel == 1]), linefmt='r-', markerfmt='ro', label='vt_commiss')
    else:
        if len(vt_event) > 0:
            plt.stem(vt_event, scale * np.ones_like(vt_event), linefmt='k-', markerfmt='ko', label='vt_event')
        if len(vt_omiss) > 0:
            plt.stem(vt_omiss, scale * np.ones_like(vt_omiss), linefmt='r-', markerfmt='ro', label='vt_omiss')

    plt.title(title)
    plt.legend()


def plot_data(seq):
    """Plots target events, context changes, and intensity of the Poisson Process over time.

    Args:
        seq (dict): A dictionary containing the sequence of events.

    Returns:
        matplotlib.figure.Figure: A Matplotlib figure object.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    axs[0].scatter(seq['time_target'], seq['mark_target'], color='red', alpha=0.6)
    axs[0].set_title('Target Events Over Time')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Mark')

    axs[1].step(seq['time_context'], seq['mark_context'], where='post')
    axs[1].set_title('Context Changes Over Time')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Context State')

    axs[2].plot(seq['t_x'], seq['lambda_x'], color='blue')
    axs[2].set_title('Intensity of the Poisson Process Over Time')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Intensity')

    plt.tight_layout()
    return fig


def plot_method(method, x, y, metric=None):
    """Plots the method curve on a graph.

    Args:
        method (dict): A dictionary containing information about the method.
        x (array-like): The x-coordinates of the points on the curve.
        y (array-like): The y-coordinates of the points on the curve.
        metric (float, optional): An optional metric value to display. Defaults to None.
    """
    name = method['name']
    if metric is None:
        label = name
    else:
        label = f'{name} ({metric:.3f})'
    plt.plot(x, y, label=label, linestyle=method['style'])


def plot_roc(label, score, method, plot=True):
    """Plots the Receiver Operating Characteristic (ROC) curve and returns the Area Under Curve (AUC).

    Args:
        label (array-like): True binary labels.
        score (array-like): Target scores.
        method (dict): A dictionary containing information about the method.
        plot (bool, optional): Whether to plot the ROC curve. Defaults to True.

    Returns:
        float: AUC score.
    """
    fpr, tpr, _ = metrics.roc_curve(label, score)
    auc = metrics.roc_auc_score(label, score)
    if plot:
        plot_method(method, fpr, tpr)
    return auc
