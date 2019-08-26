"""functions for plotting
Author: Mohammadamin Barekatain
Affiliation: TUM & OSX

Parts of this script has been copied from https://github.com/araffin/rl-baselines-zoo
"""
import matplotlib
matplotlib.use('Agg')
import argparse
import seaborn
import numpy as np
import utils.plot_utils as plt_util
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from collections import defaultdict
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
from stable_baselines.results_plotter import ts2xy
from stable_baselines.results_plotter import load_results


def plot_results(allresults, split_fn=plt_util.default_split_fn, group_fn=plt_util.defalt_group_fn, xaxis='episodes',
    average_group=False, shaded_std=True, shaded_err=True, figsize=None, legend=True, legend_outside=False, resample=0,
    smooth_step=1.0, title=None, xbase=None, ybase=None, font_scale=1.5, style='white', color_dict=None):
    """
    Plot multiple Results objects
    split_fn: function Result -> hashable, function that converts results objects into keys to split curves into
        sub-panels by. That is, the results r for which split_fn(r) is different will be put on different sub-panels.
        The sub-panels are stacked vertically in the figure.
    group_fn: function Result -> hashable, function that converts results objects into keys to group curves by.
        That is, the results r for which group_fn(r) is the same will be put into the same group.
        Curves in the same group have the same color (if average_group is False), or averaged over
        (if average_group is True). The default value is the same as default value for split_fn
    xaxis: str, name for x axis of the plot. can be one of 'timesteps', 'episodes' and 'walltime_hrs'.
    average_group: bool, if True, will average the curves in the same group and plot the mean. Enables resampling
        (if resample = 0, will use 512 steps)
    shaded_std: bool, if True (default), the shaded region corresponding to standard deviation of the group of curves
        will be shown (only applicable if average_group = True)
    shaded_err: bool, if True (default), the shaded region corresponding to error in mean estimate of the group of curves
        (that is, standard deviation divided by square root of number of curves) will be shown
        (only applicable if average_group = True)
    legend: bool, draw legends
    figsize: tuple or None, size of the resulting figure (including sub-panels). By default, width is 11.7 and height
        is 8.27 times number of sub-panels.
    legend_outside: bool, if True, will place the legend outside of the sub-panels.
    resample: int, if not zero, size of the uniform grid in x direction to resample onto.
        Resampling is performed via symmetric EMA smoothing (see the docstring for symmetric_ema). Default is zero
        (no resampling). Note that if average_group is True, resampling is necessary; in that case, default value is 512.
    smooth_step: float, when resampling (i.e. when resample > 0 or average_group is True), use this EMA decay parameter
        (in units of the new grid step). See docstrings for decay_steps in symmetric_ema or one_sided_ema functions.
    title: str, optional title for the plots
    xbase: int, regular intervals to put ticks on x axis
    ybase: int, regular intervals to put ticks on y axis
    font_scale: float
    style: seaborn style
    color_dict: dictionary specifying the color for each curve.
    """
    seaborn.set(style=style, font_scale=font_scale)

    if split_fn is None: split_fn = lambda _ : ''
    if group_fn is None: group_fn = lambda _ : ''

    # splitkey2results
    sk2r = defaultdict(list)
    for result in allresults:
        splitkey = split_fn(result)
        sk2r[splitkey].append(result)

    assert len(sk2r) > 0
    assert isinstance(resample, int), "0: don't resample. <integer>: that many samples"
    nrows = len(sk2r)
    ncols = 1
    figsize = figsize or (11.7 * 1.5, 8.27 * nrows)
    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize)

    groups = list(set(group_fn(result) for result in allresults))

    default_samples = 512
    if average_group:
        resample = resample or default_samples

    for (isplit, sk) in enumerate(sorted(sk2r.keys())):
        g2l = {}
        g2c = defaultdict(int)
        sresults = sk2r[sk]
        gresults = defaultdict(list)
        ax = axarr[isplit][0]
        for result in sresults:
            group = group_fn(result)
            g2c[group] += 1
            x, y = ts2xy(result.monitor, xaxis)
            if x is None: x = np.arange(len(y))
            x, y = map(np.asarray, (x, y))
            if average_group:
                gresults[group].append((x,y))
            else:
                if resample:
                    x, y, counts = plt_util.symmetric_ema(x, y, x[0], x[-1], resample, decay_steps=smooth_step)
                color = plt_util.COLORS[groups.index(group) % len(plt_util.COLORS)] \
                    if color_dict is None else color_dict[group]
                l, = ax.plot(x, y, color=color)
                g2l[group] = l
        if average_group:
            for group in sorted(groups):
                xys = gresults[group]
                if not any(xys):
                    continue
                color = plt_util.COLORS[groups.index(group) % len(plt_util.COLORS)] \
                    if color_dict is None else color_dict[group]
                origxs = [xy[0] for xy in xys]
                minxlen = min(map(len, origxs))
                def allequal(qs):
                    return all((q==qs[0]).all() for q in qs[1:])
                if resample:
                    low  = max(x[0] for x in origxs)
                    high = min(x[-1] for x in origxs)
                    usex = np.linspace(low, high, resample)
                    ys = []
                    for (x, y) in xys:
                        ys.append(plt_util.symmetric_ema(x, y, low, high, resample, decay_steps=smooth_step)[1])
                else:
                    assert allequal([x[:minxlen] for x in origxs]),\
                        'If you want to average unevenly sampled data, set resample=<number of samples you want>'
                    usex = origxs[0]
                    ys = [xy[1][:minxlen] for xy in xys]
                ymean = np.mean(ys, axis=0)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                l, = axarr[isplit][0].plot(usex, ymean, color=color)
                g2l[group] = l
                if shaded_err:
                    ax.fill_between(usex, ymean - ystderr, ymean + ystderr, color=color, alpha=.4)
                if shaded_std:
                    ax.fill_between(usex, ymean - ystd, ymean + ystd, color=color, alpha=.2)


        # https://matplotlib.org/users/legend_guide.html
        plt.tight_layout()
        if legend and any(g2l.keys()):
            # ax.legend(
            #     g2l.values(),
            #     ['%s (%i)'%(g, g2c[g]) for g in g2l] if average_group else g2l.keys(),
            #     loc=4 if legend_outside else None,
            #     bbox_to_anchor=(1, 1) if legend_outside else None)
            ax.legend(
                g2l.values(), g2l.keys(), loc=4, bbox_to_anchor=(1, 1) if legend_outside else None)

        ax.set_xlabel(xaxis.capitalize())
        ax.set_ylabel('Episodic Reward')
        if title is None:
            ax.set_title(sk)
        else:
            ax.set_title(title)
        if xbase:
            loc = plticker.MultipleLocator(base=xbase)
            ax.xaxis.set_major_locator(loc)
            #ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        if ybase:
            loc = plticker.MultipleLocator(base=ybase)
            ax.yaxis.set_major_locator(loc)
            #ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    return f, axarr


if __name__ == '__main__':
    # Init seaborn
    seaborn.set()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--log-dirs', help='Log folder(s)', nargs='+', required=True, type=str)
    parser.add_argument('--title', help='Plot title', default='Learning Curve', type=str)
    parser.add_argument('--smooth', action='store_true', default=False,
                        help='Smooth Learning Curve')
    args = parser.parse_args()

    results = []
    algos = []

    for folder in args.log_dirs:
        timesteps = load_results(folder)
        results.append(timesteps)
        if folder.endswith('/'):
            folder = folder[:-1]
        algos.append(folder.split('/')[-1])

    min_timesteps = np.inf

    # 'walltime_hrs', 'episodes'
    for plot_type in ['timesteps']:
        xy_list = []
        for result in results:
            x, y = ts2xy(result, plot_type)
            if args.smooth:
                x, y = plt_util.smooth((x, y), window=50)
            n_timesteps = x[-1]
            if n_timesteps < min_timesteps:
                min_timesteps = n_timesteps
            xy_list.append((x, y))

        fig = plt.figure(args.title)
        for i, (x, y) in enumerate(xy_list):
            print(algos[i])
            plt.plot(x[:min_timesteps], y[:min_timesteps], label=algos[i], linewidth=2)
        plt.title(args.title)
        plt.legend()
        if plot_type == 'timesteps':
            if min_timesteps > 1e6:
                formatter = FuncFormatter(plt_util.millions)
                plt.xlabel('Number of Timesteps')
                fig.axes[0].xaxis.set_major_formatter(formatter)

    plt.show()
