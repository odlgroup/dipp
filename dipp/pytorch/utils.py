# Copyright 2017,2018 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for pyTorch."""

__all__ = ('show_image_matrix',)


import numpy as np
from torch import Tensor


def show_image_matrix(image_batches, titles=None, indices=None, **kwargs):
    """Visualize a 2D set of images arranged in a grid.

    This function shows a 2D grid of images, where the i-th column
    shows images from the i-th batch. The typical use case is to compare
    results of different approaches with the same data, or to compare
    against a ground truth.

    Parameters
    ----------
    image_batches : sequence of `Tensor` or `Variable`
        List containing batches of images that should be displayed.
        Each tensor should have the same shape after squeezing, except
        for the batch axis.
    titles : sequence of str, optional
        Titles for the colums in the plot. By default, titles are empty.
    indices : sequence of int, optional
        Object to select the subset of the images that should be shown.
        The subsets are determined by slicing along the batch axis, i.e.,
        as ``displayed = image_batch[indices]``. The default is to show
        everything.
    kwargs :
        Further keyword arguments that are passed on to the Matplotlib
        ``imshow`` function.
    """
    import matplotlib.pyplot as plt

    if indices is None:
        displayed_batches = image_batches
    else:
        displayed_batches = [batch[indices] for batch in image_batches]

    displayed_batches = [batch.detach().cpu() if isinstance(batch, Tensor)
                         else batch for batch in displayed_batches]

    nrows = len(displayed_batches[0])
    ncols = len(displayed_batches)

    if titles is None:
        titles = [''] * ncols

    figsize = 2
    fig, rows = plt.subplots(
        nrows, ncols, sharex=True, sharey=True,
        figsize=(ncols * figsize, figsize * nrows))

    if nrows == 1:
        rows = [rows]

    for i, row in enumerate(rows):
        if ncols == 1:
            row = [row]
        for name, batch, ax in zip(titles, displayed_batches, row):
            if i == 0:
                ax.set_title(name)
            ax.imshow(batch[i].squeeze(), **kwargs)
            ax.set_axis_off()
    plt.show()


def num_model_params(model):
    """Return the number of learnable parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def summarize_model(model, test_input, verbose=False, return_summary=False):
    """Run a minibatch and print a model summary.

    Parameters
    ----------
    model : ``torch.nn.Module``
        The model for which to get the summary.
    test_input : ``torch.Tensor``
        Input minibatch that's being run to get the sizes.
    verbose : bool, optional
        If ``True``, print the full ``repr`` of each module in the model.
    return_summary : bool, optional
        If ``True``, a summary in the form of a nested ordered dictionary
        is returned.

    Returns
    -------
    summary : OrderedDict, optional
        The summary in dictionary form, returned only if
        ``return_summary=True``.

    Examples
    --------
    >>> model = nn.Linear(128, 64)
    >>> summarize_model(model, torch.zeros((3, 128)))
    === Summary of Linear ===
    Input shapes : ['(?, 128)']
    Output shapes: ['(?, 64)']
    # of params  : 8256
    <BLANKLINE>
    0: Linear
    Input shapes : ['(?, 128)']
    Output shapes: ['(?, 64)']
    # of params  : 8256
    """
    from collections import OrderedDict
    summary = OrderedDict()

    # Register hooks in the model to dump stuff. We include `modules`
    # since `model.modules()` includes the model itself as well as
    # container classes like `nn.Sequential` which we don't want to have
    # in the summary.
    modules, shapes_in, shapes_out, handles = [], [], [], []

    def hook(m, ins, outs):
        modules.append(m)
        if not isinstance(ins, (list, tuple)):
            ins = [ins]
        shapes_in.append([i.shape for i in ins])
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        shapes_out.append([o.shape for o in outs])

    for mod in model.modules():
        handles.append(mod.register_forward_hook(hook))

    # Populate the lists
    test_output = model(test_input)

    # Remove the hooks again
    for handle in handles:
        handle.remove()

    # Generate the summary
    if not isinstance(test_input, (list, tuple)):
        test_input = [test_input]
    if not isinstance(test_output, (list, tuple)):
        test_output = [test_output]

    summary[0] = {
        'name': model.__class__.__name__,
        'repr': '',
        'shapes_in': [i.shape for i in test_input],
        'shapes_out': [o.shape for o in test_output],
        'num_params': num_model_params(model)
    }

    for i, (mod, ishp, oshp) in enumerate(
        zip(modules, shapes_in, shapes_out)
    ):
        summary[i + 1] = {
            'name': mod.__class__.__name__,
            'repr': repr(mod),
            'shapes_in': ishp,
            'shapes_out': oshp,
            'num_params': num_model_params(mod)
        }

    def shape_str(shape):
        shape = tuple(shape)
        if len(shape) == 1:
            return '(?,)'
        else:
            return '(?, ' + ', '.join(str(n) for n in shape[1:]) + ')'

    num_digits = int(np.ceil(np.log10(max(len(modules) - 1, 1))))
    idx_fmt = '{{:<{}}}'.format(num_digits)

    for i, entry in summary.items():
        if i == 0:
            print('=== Summary of {} ==='.format(entry['name']))
            print('Input shapes :',
                  [shape_str(s) for s in entry['shapes_in']])
            print('Output shapes:',
                  [shape_str(s) for s in entry['shapes_out']])
            print('# of params  :', entry['num_params'])
        else:
            print()
            if verbose:
                print(idx_fmt.format(i - 1) + ': ' + entry['repr'])
            else:
                print(idx_fmt.format(i - 1) + ': ' + entry['name'])
            print('Input shapes :',
                  [shape_str(s) for s in entry['shapes_in']])
            print('Output shapes:',
                  [shape_str(s) for s in entry['shapes_out']])
            print('# of params  :',
                  entry['num_params'])

    if return_summary:
        return summary


if __name__ == '__main__':
    from dipp.util.testutils import run_doctests
    import torch
    run_doctests(extraglobs={'torch': torch, 'nn': torch.nn})
