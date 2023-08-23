# TODO reorganize
# TODO test with MadGraph
from __future__ import annotations

import json
import operator
import time
import warnings
from functools import cache, partial, reduce
from itertools import chain
from pathlib import Path
from typing import Callable, Literal

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pandas as pd
import scipy.optimize as opt
from heptools.physics import Coupling
from optimize_config import *
from rich.console import Console
from rich.terminal_theme import MONOKAI as THEME

# constants
console = Console(record = True)
n_events = 1_000_000
relunc_label = R'$\delta_\sigma$'
relunc_sqrtn_label = R'$\sqrt{N}\delta_\sigma$'
devi_label = R'$\sigma/\sigma_0$'
limits = {
    'VHH exp': {'CV': (-3.1, 3.1), 'C2V': (-7.2, 8.9), 'C3': (-30.1, 28.9)},
    'VHH obs': {'CV': (-3.7, 3.8), 'C2V': (-12.2, 13.5), 'C3': (-37.7, 37.2)},
    'nature': {'C2V': (0.67, 1.38), 'C3': (-1.24, 6.49)}
}
opt_step = 0.5
opt_unc_bound = 3

# config
opt_tasks = ['optimal', 'finetune:optimal']
plot_tasks = ['basis:optimal', 'overall:optimal', 'finetune:optimal', 'dimension:optimal']

# auto config
process = '__'.join(cfg.__name__ for cfg in Optimization.__unified__)
bounds = Optimization.bounds
kappas = Optimization.kappa
reserved = Optimization.reserved_basis
VHH_samples = Optimization.sample
VHH_model = Optimization.model
VHH_xs = Optimization.default_basis
n_kappas = len(kappas)
opt_funcs = Optimization.objective_functions
opt_algos = Optimization.algorithm_global
opt_dims = Optimization.dimensions
finetune_algos = Optimization.algorithm_local
rounding = Optimization.rounding

# plot
plot_step = 0.05
plot_range = (0.3, 2.0)
bin_width = 0.02

warnings.filterwarnings("ignore")
CMS = dict(mplhep.style.CMS)
CMS['mathtext.fontset'] = 'stix'
del CMS['mathtext.default']
plt.style.use(CMS)
base = '_'.join(sum(([k, str(l), str(u)] for k, (l, u) in bounds.items()), []))
base = Path(f'./TEMP/{process}__{base}/')
plot_path = base.joinpath('figs')
basis_path = base.joinpath('basis')
plot_path.mkdir(parents = True, exist_ok = True)
basis_path.mkdir(parents = True, exist_ok = True)

# check
assert not Optimization.undefined
console.print(Optimization.report())
console.save_html(base.joinpath('log.html'), theme = THEME)

text2tex = {
    'lme': R'$e^x$',
    'mean': R'$x$',
}
def latex(name):
    return text2tex.get(name, name)

# coupling

@cache
def couplings_benchmark(step: float):
    return Coupling(kappas).meshgrid(C2V = np.arange(*bounds['C2V'], step), C3 = np.arange(*bounds['C3'], step))

@cache
def couplings_grid(step: float):
    return np.meshgrid(np.arange(*bounds['C3'], step), np.arange(*bounds['C2V'], step))

# distribution

def mask(step: float):
    ... # TODO

def benchmark(basis, step: float, variation = None):
    m = mask(step)
    couplings = Coupling(kappas).append(basis)
    xs = VHH_samples.xs(couplings)
    if variation is not None:
        xs += np.asarray(variation) * VHH_samples.xs_unc(couplings)
    xs = xs.reshape(-1, 1)
    model = VHH_model(np.concatenate([basis, xs], axis = 1))
    r = model.xs_unc(couplings_benchmark(step)) / model.xs(couplings_benchmark(step))
    if m is not None:
        r[~m] = np.nan
    return r

def deviation(basis, step: float):
    model = VHH_model(np.concatenate([basis, VHH_samples.xs(Coupling(kappas).append(basis)).reshape(-1, 1)], axis = 1))
    return model.xs(couplings_benchmark(step)) / VHH_samples.xs(couplings_benchmark(step))

def find_point(c3: float, c2v: float):
    return int((c3-bounds['C3'][0])/opt_step), int((c2v-bounds['C2V'][0])/opt_step)

# optimization

def optimize_basis(dimension: list[int], objective_functions: dict[str, Callable], optimize_algorithms: dict[str, Callable]):
    for algo_name, algo in optimize_algorithms.items():
        for n in dimension:
            for func_name, func in objective_functions.items():
                print('start', algo_name, func_name, n)
                def ev_func(x):
                    return func(benchmark(np.concatenate((reserved, x.reshape(-1,n_kappas))), opt_step))
                wall_time = time.time()
                result = algo(ev_func,
                              bounds = opt.Bounds(
                    lb = np.array([bounds[k][0] for k in kappas] * (n - len(reserved))), 
                    ub = np.array([bounds[k][1] for k in kappas] * (n - len(reserved)))))
                wall_time = time.time() - wall_time
                print(wall_time, result.nfev)
                yield {
                    'function' : func_name,
                    'algorithm': algo_name,
                    'basis'    : np.concatenate((reserved, result.x.reshape(-1,n_kappas))).tolist(),
                    'time'     : wall_time,
                    'nfevs'    : result.nfev
                }

def estimate_uncertainty(bases, objective_functions: dict[str, Callable], optimize_algorithms: dict[str, Callable]):
    for basis in bases:
        basis = np.asarray(basis)
        for algo_name, algo in optimize_algorithms.items():
            for func_name, func in objective_functions.items():
                print('estimate uncertainty', algo_name, func_name)
                def ev_func(x):
                    return -func(benchmark(basis, opt_step, x))
                wall_time = time.time()
                result = algo(ev_func, bounds = opt.Bounds(
                    lb = -opt_unc_bound * np.ones(len(basis)),
                    ub =  opt_unc_bound * np.ones(len(basis))))
                wall_time = time.time() - wall_time
                print(wall_time, result.nfev)
                yield {
                    'function'   : func_name,
                    'algorithm'  : algo_name,
                    'basis'      : basis.tolist(),
                    'uncertainty': result.x.tolist(),
                    'time'       : wall_time,
                    'nfevs'      : result.nfev
                }

def finetune_optimum(bases: pd.DataFrame, minimize: list[str]):
    for algo in minimize:
        print(algo)
        for _, basis in bases.iterrows():
            print('finetune')
            print(basis)
            def ev_func(x):
                return opt_funcs[basis.function](benchmark(np.concatenate((reserved, x.reshape(-1,n_kappas))), opt_step))
            start = np.asarray(basis.basis)[len(reserved):].reshape(-1)
            wall_time = time.time()
            result = opt.minimize(ev_func, start, method = algo)
            wall_time = time.time() - wall_time
            print(wall_time, result.nfev)
            yield {
                'function'   : basis.function,
                'algorithm'  : algo,
                'basis'      : np.concatenate((reserved, result.x.reshape(-1,n_kappas))).tolist(),
                'time'       : wall_time,
                'nfevs'      : result.nfev,
                'global_algorithm' : basis.algorithm,
                'global_basis'     : basis.basis,
            }


# io
def save_basis(bases: list, filename: str):
    json.dump(bases, open(basis_path.joinpath(f'{filename}.json'), 'w'), indent = 4)

def load_basis(*files: str, df = True):
    try:
        bases = reduce(operator.add, (json.load(open(basis_path.joinpath(f'{file}.json'), 'r')) for file in files))
        if df:
            return pd.DataFrame(bases)
        return bases
    except:
        return None

def select_basis(basis, dimension:int = None, variation: float = None, algorithm: str = None, **select: str):
    for k, v in select.items():
        basis = basis[basis[k] == v]
    if algorithm is not None:
        basis = basis[basis['algorithm'].str.startswith(f'{algorithm}')]
    if dimension is not None:
        basis = basis[basis['basis'].apply(lambda x: len(x)) == dimension]
    if variation is not None and 'uncertainty' in basis.columns:
        basis.uncertainty = basis.uncertainty.apply(lambda x: np.asarray(x) * variation)
    return basis

# plot

def save_fig(prefix: str, title: str):
    plt.savefig(plot_path.joinpath('_'.join((prefix, title.replace(' ', '_')))))

def get_color(cmap: str, values, reverse = False, alpha = 1):
    cmap = plt.get_cmap(cmap)
    order = np.argsort(values)
    order = np.argsort(order[::-1] if reverse else order)
    colors = cmap(order / len(values))
    colors[:, -1] = alpha
    return dict(zip(values, colors))

def get_range(*values, rounding = 0.5):
    values = np.asarray(values)
    return np.max([0, np.floor(np.min(values)/rounding) * rounding]), np.min([50, np.ceil(np.nanmax(values)/rounding) * rounding])

def abbr(string: str):
    if string is None:
        return None
    string = string.replace('_', ' ')
    return ''.join(s for s in string.title() if s.isupper())

def basis_label(func: str = None, algo: str = None, dimension: int = None, tex = True):
    if dimension is not None:
        dimension = str(dimension)
    if tex:
        func = latex(func)
        algo = latex(abbr(algo))
        dimension = latex(dimension)
    else:
        algo = abbr(algo)
    return ' '.join(filter(None, [func, algo, dimension]))

def get_label(basis_df: pd.DataFrame, index):
    return basis_label(basis_df.function[index], basis_df.algorithm[index], len(basis_df.basis[index]))

def get_arg(args: tuple, index: int):
    if len(args) > index and args[index] is not ...:
        return args[index]

def plot_uncertainty_1d(filename: str, legend: str, hists: list[tuple], normed = True, x_range: tuple[float, float] = ..., r_range = 0.1, y_log = False, tags = None, fill = False, legend_colunms = 1, alpha = None):
    '''
    hists: (label, hist, color, vline, func)
    '''
    fig = plt.figure(dpi = 192)
    gs = fig.add_gridspec(2, 1,  height_ratios=(4, 1), bottom = 0.05, left = 0.08, top = 0.92, right = 0.98)
    main = fig.add_subplot(gs[0, 0])
    ratio = fig.add_subplot(gs[1, 0], sharex=main)
    ref = None
    if x_range is ...:
        x_range = get_range(*(hist[1] for hist in hists))
    for args in hists:
        kwargs = {}
        label = args[0]
        hist = args[1]
        color = get_arg(args, 2)
        vline = get_arg(args, 3)
        funcs = get_arg(args, 4)
        if color is not None:
            kwargs['color'] = tuple(color)
        if funcs is not None:
            if isinstance(funcs, str):
                funcs = [funcs]
            if isinstance(funcs, list):
                label += ''.join(f'({basis_label(k)}={opt_funcs[k](hist):.4g})' for k in funcs)
            elif isinstance(funcs, dict):
                label += ''.join(f'({basis_label(k)}={v(hist):.4g})' for k, v in funcs.items())
        if alpha is not None:
            kwargs['alpha'] = alpha
        histtype = 'fill' if fill or get_arg(args, 5) else 'step'
        hist = hist[~np.isnan(hist)]
        h = np.histogram(hist, bins = int((x_range[1] - x_range[0])/bin_width), range = x_range, normed = normed)
        mplhep.histplot(h, label = label, ax = main, histtype = histtype, **kwargs)
        if ref is None:
            ref = h 
            ratio.set_title(f'reference: {legend} = {label}', fontsize = 15)
        r = (np.nan_to_num(h[0]/ref[0], nan = 1), ref[1])
        mplhep.histplot(r, ax = ratio, **kwargs)
        if vline is not None:
            main.axvline(vline, **kwargs)
    mplhep.cms.text('internal', ax = main, fontsize = 15)
    if isinstance(r_range, tuple):
        ratio.set_ylim(*r_range)
    else:
        ratio.set_ylim(1 - r_range, 1 + r_range)
    if y_log:
        main.set_yscale('log')
    main.set_xlabel(relunc_sqrtn_label, fontsize = 22)
    main.xaxis.set_label_coords(1, -0.05)
    if legend != '':
        main.legend(title = legend, ncol = legend_colunms, fontsize = 15)
    title = [relunc_label, 'distribution']
    if tags is None:
        tags = []
    if normed:
        tags += ['normalized']
    tags = [f'[{i}]' for i in tags]
    title = tags + title
    fig.suptitle(' '.join(title), fontsize = 20)
    save_fig(legend.replace(' ', '_'), filename)


def plot_grid_size(filename: str, size: list[float], basis):
    reverse = False
    if size[0] == min(size):
        reverse = True
    colors = get_color('viridis', size, reverse)
    plot_uncertainty_1d(filename, 'grid size',
                 [(f'{i:.2g}' + ('(optimize)' if i == opt_step else ''), benchmark(basis, i), colors[i], ..., opt_funcs) for i in size],
                 normed = True, x_range = plot_range, r_range = 0.1
                 )

def plot_stability(filename: str, rounding: list[int], basis):
    colors = get_color('viridis', np.arange(len(rounding)))
    rounding = [(f'{r}', np.round(basis, r))  if r is not None else ('original', basis) for r in rounding]
    plot_uncertainty_1d(filename, 'stability',
                 [(l, benchmark(b, opt_step), colors[i], ..., opt_funcs) for i, (l, b) in enumerate(rounding)],
                 normed = True, x_range = plot_range, r_range = 0.1
                 )

def plot_seed(algo: str, func: str, dimension:int, basis_df: pd.DataFrame):
    basis_df = basis_df[basis_df['algorithm'].str.startswith(f'{algo}_')]
    basis_df = select_basis(basis_df, function = func, dimension = dimension)
    plot_uncertainty_1d(basis_label(func, algo, dimension, False), 'seed',
                [(basis.algorithm.removeprefix(f"{algo}_"), benchmark(basis.basis, plot_step), ..., ..., basis.function) for _, basis in basis_df.iterrows()],
                normed = True, x_range = plot_range, r_range = 1, tags = basis_label(func, algo, dimension).split(' ')
                )

def plot_variation(filename: str, basis, func = None, **variation):
    cmap = plt.get_cmap('viridis')
    if func is None:
        func = []
    plot_uncertainty_1d(filename, 'variation',
                [('central', benchmark(basis, plot_step), cmap(1.0), ..., func)] + [(k, benchmark(basis, plot_step, v), cmap(1 - (i + 1)/len(variation)), ..., func) for i, (k, v) in enumerate(variation.items())],
                normed = True, x_range = plot_range, r_range = 0.5
                )

def plot_finetune(basis_df):
    global_minima = []
    finetuned_bases = []
    for _, basis in basis_df.iterrows():
        matched = False
        for i, global_minimum in enumerate(global_minima):
            if (len(basis.global_basis) == len(global_minimum.global_basis)) and np.allclose(basis.global_basis, global_minimum.global_basis):
                matched = True
                finetuned_bases[i].append(basis)
                break
        if not matched:
            global_minima.append(basis)
            finetuned_bases.append([basis])
    for i, bases in enumerate(finetuned_bases):
        global_minimum = global_minima[i]
        plot_uncertainty_1d(f'basis_{i}_{basis_label(global_minimum.function, global_minimum.global_algorithm, len(global_minimum.global_basis), False)}', 'finetune',
                    [(f'global({global_minimum.global_algorithm})', benchmark(global_minimum.global_basis, plot_step), ..., ..., basis.function, True)] + 
                    [(basis.algorithm, benchmark(basis.basis, plot_step), ..., ..., basis.function) for basis in bases],
                    x_range = plot_range, r_range = 0.1)

def plot_uncertainty_2d(title: str, grids: list[list[dict[str, str] | tuple]], basis_df: pd.DataFrame = None, content: Literal['benchmark', 'deviation'] = 'benchmark', z_range: tuple[float, float] = ...):
    height = len(grids)
    width  = max(len(row) for row in grids)
    z_max = 0
    z_min = np.inf
    fig = plt.figure(figsize = (width * 7, height * 6), dpi = 192)
    for i, row in enumerate(grids):
        for j, cell in enumerate(row):
            if isinstance(cell, dict):
                if 'step' in cell:
                    step = cell['step']
                    cell.pop('step')
                    label = f'{step}_'
                else:
                    step = plot_step
                    label = ''
                basis = select_basis(basis_df, **cell)
                idx = list(basis.index)[0]
                if content == 'benchmark':
                    variation = basis.uncertainty[idx] if 'uncertainty' in basis.columns else None
                    unc = benchmark(basis.basis[idx], step, variation)
                elif content == 'deviation':
                    unc = deviation(basis.basis[idx], step)
                label += get_label(basis, idx)
            elif isinstance(cell, tuple):
                var = get_arg(cell, 2)
                step = get_arg(cell, 3)
                if step is None:
                    step = plot_step
                if content == 'benchmark':
                    unc = benchmark(cell[0], step, var)
                elif content == 'deviation':
                    unc = deviation(cell[0], step)
                label = cell[1]
            else:
                raise TypeError('cell must be dict or tuple')
            grids[i][j] = (unc, label, step)
            z_max = max(z_max, np.nanmax(unc))
            z_min = min(z_min, np.nanmin(unc))
    if z_range is ...:
        z_min, z_max = get_range(z_min, z_max)
    else:
        z_min, z_max = z_range
    for i, row in enumerate(grids):
        for j, (unc, label, step) in enumerate(row):
            ax = fig.add_subplot(height, width, i * width + j + 1)
            c3, c2v = couplings_grid(step)
            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color = 'white')
            cb = ax.pcolormesh(c3, c2v, unc.reshape(c3.shape), shading = 'nearest', cmap = cmap, vmin = z_min, vmax = z_max)
            ax.set_title(label, fontsize = 20)
            ax.set_xlabel(R'$\kappa_\lambda$')
            ax.xaxis.set_label_coords(1.00, -0.05)
            ax.set_ylabel(R'$\kappa_{2V}$')
            ax.yaxis.set_label_coords(-0.08, 1.00)
            for k, v in limits.items():
                rect = patches.Rectangle((v['C3'][0], v['C2V'][0]), v['C3'][1] - v['C3'][0], v['C2V'][1] - v['C2V'][0], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.text(np.mean(v['C3']), v['C2V'][1], k, fontsize = 15, color = 'r', horizontalalignment='center')
            ax.set_xlim(*bounds['C3'])
            ax.set_ylim(*bounds['C2V'])
    fig.subplots_adjust(left = 1/(width*7), right = 1-2/(width*7))
    cbar = fig.colorbar(cb, cax = fig.add_axes([1-1.5/(width*7), 0.15, 0.3/(width*7), 0.7]))
    if content == 'benchmark':
        cbar.set_label(relunc_sqrtn_label, fontsize = 20)
    elif content == 'deviation':
        cbar.set_label(devi_label, fontsize = 20)
    fig.suptitle(title, fontsize = 20)
    save_fig(content, title.replace(' ', '_'))

# tasks

class PlotTasks:
    _optima = {}
    @classmethod
    def optima(cls, filename: str):
        return cls._optima.setdefault(filename, load_basis(filename))

    @classmethod
    def overall(cls, filename: str):
        print('comparing overall performance')
        all_basis = [('original', benchmark(VHH_xs[:, :n_kappas], plot_step), ..., ..., list(opt_funcs.keys()))]
        for func in opt_funcs.keys():
            for algo in opt_algos.keys():
                for dimension in opt_dims:
                    basis = select_basis(cls.optima(filename), function = func, algorithm = algo, dimension = dimension)
                    idx = list(basis.index)[0]
                    all_basis.append((basis_label(func, algo, dimension), benchmark(basis.basis[idx], plot_step), ..., ..., func))
        plot_uncertainty_1d(f'overall', 'optimization', all_basis, fill = True, x_range = (0, 5), legend_colunms = 2, alpha = 0.5)
        plot_uncertainty_1d(f'overall_log', 'optimization', all_basis, fill = True, y_log = True, legend_colunms = 2, alpha = 0.5)
        plot_uncertainty_2d(f'overall',[[(VHH_xs[:, :n_kappas], 'original'), {'function': 'lme', 'algorithm': 'dual_annealing', 'dimension': opt_dims[-1]}]], cls.optima(filename))
        plot_uncertainty_2d(f'VHH',[[(VHH_xs[:, :n_kappas], 'original', ..., 0.01)]])

    @classmethod
    def objective_function(cls, filename: str):
        print('comparing objective functions')
        for algo in opt_algos.keys():
            for dimension in opt_dims:
                bases = select_basis(cls.optima(filename), algorithm = algo, dimension = dimension)
                plot_uncertainty_1d(basis_label(None, algo, dimension, False), 'objective function', [(basis_label(func = basis.function), benchmark(basis.basis, plot_step), ..., 1/np.sqrt(dimension), basis.function) for _, basis in bases.iterrows()], tags = basis_label(algo = algo, dimension = dimension).split(' '))
            plot_uncertainty_2d(algo,[
                                [{'function': func, 'dimension': dimension, 'algorithm': algo} for dimension in opt_dims]for func in opt_funcs.keys()
                            ], cls.optima(filename))
    
    @classmethod
    def algorithm(cls, filename: str):
        print('comparing algorithms')
        for func in opt_funcs.keys():
            for dimension in opt_dims:
                bases = select_basis(cls.optima(filename), function = func, dimension = dimension)
                plot_uncertainty_1d(basis_label(func, None, dimension, False), 'algorithm', [(basis_label(algo = basis.algorithm), benchmark(basis.basis, plot_step), ..., 1/np.sqrt(dimension), func) for _, basis in bases.iterrows()], tags = [func, f'{dimension}'])
            plot_uncertainty_2d(func,[
                                [{'function': func, 'dimension': dimension, 'algorithm': algo} for dimension in opt_dims]for algo in opt_algos.keys()
                            ], cls.optima(filename))

    @classmethod
    def dimension(cls, filename: str):
        print('comparing dimensions')
        for func in opt_funcs.keys():
            for algo in opt_algos.keys():
                basis = select_basis(cls.optima(filename), function = func, algorithm = algo)
                indices = list(basis.index)
                dimension = {idx: len(basis.basis[idx]) for idx in indices}
                colors = get_color('viridis', list(dimension.values()), alpha = 0.5)
                plot_uncertainty_1d(basis_label(func, algo, tex = False), 'n basis', [(basis_label(dimension = dimension[idx]), benchmark(basis.basis[idx], plot_step), colors[dimension[idx]], 1/np.sqrt(dimension[idx]), func) for idx in indices], tags = basis_label(func, algo).split(' '), fill = True, x_range = plot_range)

    @classmethod
    def finetune(cls, filename: str):
        print('comparing finetuning performance')
        plot_finetune(cls.optima(f'{filename}_finetune'))

    @classmethod
    def seed(cls):
        print('comparing seed stability')
        for algo in opt_algos.keys():
            for func in opt_funcs.keys():
                plot_seed(algo, func, opt_dims[-1], cls.optima('seed'))

    @classmethod
    def basis(cls, filename: str):
        for func in opt_funcs.keys():
            for algo in opt_algos.keys():
                for dimension in opt_dims:
                    basis = select_basis(cls.optima(filename), function = func, algorithm = algo, dimension = dimension)
                    idx = list(basis.index)[0]
                    # grid size
                    plot_grid_size(basis_label(func, algo, dimension, False), [0.05, 0.1, 0.2, 0.3, 0.4, 0.5], basis.basis[idx])
                    plot_grid_size(basis_label(func, algo, dimension, False) + '_large', [0.05, 0.6, 0.8, 1.0], basis.basis[idx])
                    plot_stability(basis_label(func, algo, dimension, False), [None, 5, 4, 3, 2, 1], basis.basis[idx])
                    # xs variation
                    if 'uncertainty' in basis.columns:
                        variation = np.asarray(basis.uncertainty[idx])
                        if np.all(np.abs(variation) == opt_unc_bound):
                            plot_variation(basis_label(func, algo, dimension, False), basis.basis[idx], **{
                                R'$1\sigma$': variation * 1/3,
                                R'$2\sigma$': variation * 2/3,
                                R'$3\sigma$': variation * 3/3}, func = func)
                        else:
                            plot_variation(basis_label(func, algo, dimension, False), basis.basis[idx], **{
                                Rf'worst($\pm 3\sigma$)': variation}, func = func)

class OptimizeTasks:
    @staticmethod
    def optimal():
        bases = []
        for result in optimize_basis(opt_dims, opt_funcs, opt_algos):
            bases.append(result)
            save_basis(bases, 'optimal')

    @staticmethod
    def seed():
        seeds = []
        seeded_algos = {f'{algo}_{seed}': partial(opt.__dict__[algo], seed = seed) for seed in range(5) for algo in opt_algos.keys()}
        for result in optimize_basis([opt_dims[-1]], opt_funcs, seeded_algos):
            seeds.append(result)
            save_basis(seeds, 'seed')

    @staticmethod
    def finetune(filename: str):
        basis = load_basis(filename)
        finetunes = []
        for result in finetune_optimum(basis, finetune_algos):
            finetunes.append(result)
            save_basis(finetunes, filename + '_finetune')

    @staticmethod
    def uncertainty(filename: str):
        bases = load_basis(filename, df = False)
        uncertainties = []
        optimizes = []
        for basis in bases:
            algo = basis['algorithm']
            func = basis['function']
            basis = basis['basis']
            optimizes.append(estimate_uncertainty([basis], {func: opt_funcs[func]}, {algo: opt_algos[algo]}))
        for result in chain(*optimizes):
            uncertainties.append(result)
            save_basis(uncertainties, 'uncertainty')

def run(TaskCollection, tasks: list[str]):
    for task in tasks:
        args = task.split(':')
        getattr(TaskCollection, args[0])(*args[1:])

def best(*filenames: str, stability: float = None):
    best_path = basis_path.joinpath('best')
    best_path.mkdir(parents = True, exist_ok = True)
    all_bases = []
    best_bases = {}
    for filename in filenames:
        bases = load_basis(filename)
        for _, basis in bases.iterrows():
            b = basis.basis
            b_r = np.round(b, rounding)
            benchmark_b = opt_funcs[basis.function](benchmark(b  , plot_step))
            benchmark_r = opt_funcs[basis.function](benchmark(b_r, plot_step))
            best_basis = best_bases.setdefault(basis.function, {'overall': None})
            best_basis.setdefault(len(b), None)
            all_bases.append(
                {
                    'basis'     : b,
                    'rounded'   : b_r,
                    'dimension' : len(b),
                    'function'  : basis.function,
                    'benchmark' : benchmark_b,
                    'stability' : (benchmark_r - benchmark_b)/benchmark_b,
                }
            )
    all_bases = pd.DataFrame(all_bases)
    print(best_bases)
    print(all_bases)
    # TODO per dimension, per func, write to tex, json

best('optimal', 'optimal_finetune', stability = 1e-4)
# run(OptimizeTasks, opt_tasks)
# run(PlotTasks, plot_tasks)
