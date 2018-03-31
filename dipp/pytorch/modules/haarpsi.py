# Copyright 2017,2018 The DIPP contributors
#
# This file is part of DIPP.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Pytorch modules for the Haar perceptual similarity index FOM."""

import numpy as np
import torch
from torch import nn, autograd
from dipp.pytorch.modules.generic import InvSigmoid

__all__ = ('HaarPSI',)


class HaarPSIScore(nn.Module):

    """Pointwise :math:`(2xy + c^2) / (x^2 + y^2 + c^2)`.

    The constant :math:`c > 0` determines the penalization of
    dissimilarity. It should be of the same order of magnitude as
    :math:`x` and :math:`y`.
    """

    def __init__(self, c=None, init_c=None):
        """Initialize a new instance.

        Parameters
        ----------
        c : positive float or `torch.nn.parameter.Parameter`, optional
            The :math:`c` parameter in the function. If not provided, it
            is registered as a learnable parameter. A provided
            ``Parameter`` is registered as-is.
        c_init : positive float, optional
            If ``c`` is ``None`` and thus learnable, this value must be
            provided as an initial value. Unused otherwise.

        Examples
        --------
        Initialize with fixed ``c``:

        >>> sim = HaarPSIScore(2)
        >>> x = autograd.Variable(torch.Tensor([0, 1, 2]))
        >>> y = autograd.Variable(torch.Tensor([0, 0, 0]))
        >>> sim(x, y)  # should be [1, 4/5, 1/2]
        Variable containing:
         1.0000
         0.8000
         0.5000
        [torch.FloatTensor of size 3]

        Make ``c`` learnable:

        >>> sim = HaarPSIScore(init_c=2)
        >>> params = list(sim.parameters())
        >>> len(params)
        1
        >>> params[0]
        Parameter containing:
         2
        [torch.FloatTensor of size 1]
        """
        super(HaarPSIScore, self).__init__()
        if c is None:
            if init_c is None:
                raise ValueError('`init_c` must be provided for `c=None`')
            self.c = nn.Parameter(torch.Tensor([init_c]))
        elif isinstance(c, nn.Parameter):
            self.c = c
        else:
            assert c > 0
            self.c = float(c)

    def forward(self, x, y):
        return (2 * x * y + self.c ** 2) / (x ** 2 + y ** 2 + self.c ** 2)


class HaarPSISimilarityMap(nn.Module):

    """Pointwise similarity score for the HaarPSI FOM.

    For input images :math:`f_1, f_2`, this module computes

    .. math::
        \mathrm{HS}_{f_1, f_2}^{(k)}(x) =
        l_a \\left(
        \\frac{1}{2} \sum_{j=1}^2
        S\\left(\\left|g_j^{(k)} \\ast f_1 \\right|(x),
        \\left|g_j^{(k)} \\ast f_2 \\right|(x), c\\right)
        \\right),

    see `[Rei+2016] <https://arxiv.org/abs/1607.06140>`_ equation (10).

    Here, :math:`l_a` is the logistic function,
    :math:`S(x, y, c) = (2xy + c) / (x^2+y^2 + c)` the pointwise similarity
    score, and the superscript :math:`(k)` refers to the axis (0 or 1)
    in which edge features are compared.

    The :math:`j`-th level Haar wavelet filters :math:`g_j^{(k)}` are
    high-pass in the axis :math:`k` and low-pass in the other axis,
    respectively.

    References
    ----------
    [Rei+2016] Reisenhofer, R, Bosse, S, Kutyniok, G, and Wiegand, T.
    *A Haar Wavelet-Based Perceptual Similarity Index for Image Quality
    Assessment*. arXiv:1607.06140 [cs], Jul. 2016.
    """

    def __init__(self, axis, a=None, c=None, init_c=None):
        """Initialize a new instance.

        Parameters
        ----------
        axis : {0, 1}
            The axis :math:`k` along which high-pass filters should be
            applied.
        a : positive float or `torch.nn.parameter.Parameter`, optional
            The :math:`a` parameter in the function. If not provided, it
            is registered as a learnable parameter and initialized randomly
            between 0.5 and 5. A provided ``Parameter`` is stored
            as-is.
        c : positive float, optional
            The :math:`c` parameter in the function. If not provided, it
            is registered as a learnable `torch.nn.parameter.Parameter`.
        c_init : positive float, optional
            If ``c`` is ``None`` and thus learnable, this value must be
            provided as an initial value. Otherwise it is not used.
        """
        super(HaarPSISimilarityMap, self).__init__()
        assert axis in (0, 1)
        self.axis = int(axis)
        self.sim_score = HaarPSIScore(c, init_c)

        if a is None:
            self.a = nn.Parameter(torch.Tensor(1))
            self.a.data.uniform_(0.5, 5)
        elif isinstance(a, nn.Parameter):
            self.a = a
        else:
            assert a > 0
            self.a = float(a)

    def forward(self, x, y):
        # Stack input to do them as a batch, add empty in_channels dim
        xy = torch.stack([x, y], dim=0)[:, None, ...]

        # Init 1d filters for levels 1 and 2. Since pytorch computes
        # correlation, not convolution, we flip the filters.
        filt_lo_lvl1 = np.array([np.sqrt(2), np.sqrt(2)], dtype='float32')
        filt_hi_lvl1 = np.array([np.sqrt(2), -np.sqrt(2)], dtype='float32')
        filt_lo_lvl2 = np.repeat(filt_lo_lvl1, 2)
        filt_hi_lvl2 = np.repeat(filt_hi_lvl1, 2)

        # Make 2D filters by outer products and add empty dims for
        # out_channels and in_channels
        if self.axis == 0:
            # Horizontal high-pass
            f_l1_arr = np.multiply.outer(filt_hi_lvl1, filt_lo_lvl1)
            f_l1 = autograd.Variable(
                torch.from_numpy(f_l1_arr[None, None, ...]))

            f_l2_arr = np.multiply.outer(filt_hi_lvl2, filt_lo_lvl2)
            f_l2 = autograd.Variable(
                torch.from_numpy(f_l2_arr[None, None, ...]))

        else:
            # Vertical high-pass
            f_l1_arr = np.multiply.outer(filt_lo_lvl1, filt_hi_lvl1)
            f_l1 = autograd.Variable(
                torch.from_numpy(f_l1_arr[None, None, ...]))

            f_l2_arr = np.multiply.outer(filt_lo_lvl2, filt_hi_lvl2)
            f_l2 = autograd.Variable(
                torch.from_numpy(f_l2_arr[None, None, ...]))

        # Do the convolution. Padding must be `kernel_size - 1` to
        # compute the convolution using all input values. The padding
        # will be distributed evenly to the "left" and the "right", with
        # odd paddings adding one extra to the right.
        conv_l1 = nn.functional.conv2d(xy, f_l1, padding=1)
        conv_l2 = nn.functional.conv2d(xy, f_l2, padding=3)

        conv_x_l1 = conv_l1[0, 0, :-1, :-1]
        conv_y_l1 = conv_l1[1, 0, :-1, :-1]
        conv_x_l2 = conv_l2[0, 0, 1:-2, 1:-2]
        conv_y_l2 = conv_l2[1, 0, 1:-2, 1:-2]

        # Evaluate score per level
        score_l1 = self.sim_score(torch.abs(conv_x_l1), torch.abs(conv_y_l1))
        score_l2 = self.sim_score(torch.abs(conv_x_l2), torch.abs(conv_y_l2))

        # Return logistic function applied to the mean
        return nn.functional.sigmoid(self.a * (score_l1 + score_l2) / 2)


class HaarPSIWeightMap(nn.Module):

    """Pointwise weight map for computation of the HaarPSI FOM.

    For input images :math:`f_1, f_2`, this module computes

    .. math::
        \mathrm{W}_{f_1, f_2}^{(k)}(x) =
        \max \\left\{
        \\left|g_3^{(k)} \\ast f_1 \\right|(x),
        \\left|g_3^{(k)} \\ast f_2 \\right|(x)
        \\right\},

    see `[Rei+2016] <https://arxiv.org/abs/1607.06140>`_ equations (11)
    and (13).
    Here, the superscript :math:`(k)` refers to the axis (0 or 1) in which
    edge features are compared, and the 3rd-level Haar wavelet filters
    :math:`g_3^{(k)}` are high-pass in the axis :math:`k` and low-pass in
    the other axis, respectively.

    References
    ----------
    [Rei+2016] Reisenhofer, R, Bosse, S, Kutyniok, G, and Wiegand, T.
    *A Haar Wavelet-Based Perceptual Similarity Index for Image Quality
    Assessment*. arXiv:1607.06140 [cs], Jul. 2016.
    """

    def __init__(self, axis):
        """Initialize a new instance.

        Parameters
        ----------
        axis : {0, 1}
            The axis :math:`k` along which high-pass filters should be
            applied.
        """
        super(HaarPSIWeightMap, self).__init__()
        assert axis in (0, 1)
        self.axis = int(axis)

    def forward(self, x, y):
        # Stack input to do them as a batch, add empty in_channels dim
        xy = torch.stack([x, y], dim=0)[:, None, ...]

        # Init 1d filters for level 3. Since pytorch computes
        # correlation, not convolution, we flip the filters.
        filt_lo_lvl1 = np.array([np.sqrt(2), np.sqrt(2)], dtype='float32')
        filt_hi_lvl1 = np.array([np.sqrt(2), -np.sqrt(2)], dtype='float32')
        filt_lo_lvl3 = np.repeat(filt_lo_lvl1, 4)
        filt_hi_lvl3 = np.repeat(filt_hi_lvl1, 4)

        # Make 2D filters by outer products and add empty dims for
        # out_channels and in_channels
        if self.axis == 0:
            # Horizontal high-pass
            f_l3_arr = np.multiply.outer(filt_hi_lvl3, filt_lo_lvl3)
            f_l3 = autograd.Variable(
                torch.from_numpy(f_l3_arr[None, None, ...]))

        else:
            # Vertical high-pass
            f_l3_arr = np.multiply.outer(filt_lo_lvl3, filt_hi_lvl3)
            f_l3 = autograd.Variable(
                torch.from_numpy(f_l3_arr[None, None, ...]))

        # Do the convolution. Padding must be `kernel_size - 1` to
        # compute the convolution using all input values. The padding
        # will be distributed evenly to the "left" and the "right", with
        # odd paddings adding one extra to the right.
        conv_l3 = nn.functional.conv2d(xy, f_l3, padding=7)

        conv_x_l3 = conv_l3[0, 0, 3:-4, 3:-4]
        conv_y_l3 = conv_l3[1, 0, 3:-4, 3:-4]

        # Return maximum of the absolute values
        return torch.max(torch.abs(conv_x_l3), torch.abs(conv_y_l3))


class HaarPSI(nn.Module):

    """The Haar Perceptual Similarity Index for image comparison.

    For input images :math:`f_1, f_2`, this module computes the scalar

    .. math::
        \mathrm{HaarPSI}_{f_1, f_2} =
        l_a^{-1} \\left(
        \\frac{
        \sum_x \sum_{k=1}^2 \mathrm{HS}_{f_1, f_2}^{(k)}(x) \cdot
        \mathrm{W}_{f_1, f_2}^{(k)}(x)}{
        \sum_x \sum_{k=1}^2 \mathrm{W}_{f_1, f_2}^{(k)}(x)}
        \\right)^2

    see `[Rei+2016] <https://arxiv.org/abs/1607.06140>`_ equation (12).
    Here :math:`l_a^{-1}` is the inverse logistic function, and the
    functions :math:`\mathrm{HS}` and :math:`\mathrm{W}` are defined as

    .. math::
        \mathrm{HS}_{f_1, f_2}^{(k)}(x) &=
        l_a \\left(
        \\frac{1}{2} \sum_{j=1}^2
        S\\left(\\left|g_j^{(k)} \\ast f_1 \\right|(x),
        \\left|g_j^{(k)} \\ast f_2 \\right|(x), c\\right)
        \\right),

        \mathrm{W}_{f_1, f_2}^{(k)}(x) &=
        \max \\left\{
        \\left|g_3^{(k)} \\ast f_1 \\right|(x),
        \\left|g_3^{(k)} \\ast f_2 \\right|(x)
        \\right\}.

    Here, :math:`l_a` is the logistic function,
    :math:`S(x, y, c) = (2xy + c) / (x^2+y^2 + c)` the pointwise similarity
    score, and the superscript :math:`(k)` refers to the axis (0 or 1)
    in which edge features are compared.

    The :math:`j`-th level Haar wavelet filters :math:`g_j^{(k)}` are
    high-pass in the axis :math:`k` and low-pass in the other axis,
    respectively.

    References
    ----------
    [Rei+2016] Reisenhofer, R, Bosse, S, Kutyniok, G, and Wiegand, T.
    *A Haar Wavelet-Based Perceptual Similarity Index for Image Quality
    Assessment*. arXiv:1607.06140 [cs], Jul. 2016.
    """

    def __init__(self, a=None, c=None, init_c=None):
        """Initialize a new instance.

        Parameters
        ----------
        a : positive float or `torch.nn.parameter.Parameter`, optional
            The :math:`a` parameter in the function. If not provided, it
            is registered as a learnable parameter and initialized randomly
            between 0.5 and 5. A provided ``Parameter`` is stored
            as-is.
        c : positive float or `torch.nn.parameter.Parameter`, optional
            The :math:`c` parameter in the function. If not provided, it
            is registered as a learnable parameter. A provided
            ``Parameter`` is registered as-is.
        c_init : positive float, optional
            If ``c`` is ``None`` and thus learnable, this value must be
            provided as an initial value. Otherwise it is not used.
        """
        super(HaarPSI, self).__init__()

        if a is None:
            self.a = nn.Parameter(torch.Tensor(1))
            self.a.data.uniform_(0.5, 5)
        elif isinstance(a, nn.Parameter):
            self.a = a
        else:
            assert a > 0
            self.a = float(a)

        if c is None:
            if init_c is None:
                raise ValueError('`init_c` must be provided for `c=None`')
            self.c = nn.Parameter(torch.Tensor([init_c]))
        elif isinstance(c, nn.Parameter):
            self.c = c
        else:
            assert c > 0
            self.c = float(c)

        # Sharing the `a` and `c` parameters with the constituting modules
        self.inv_sigmoid = InvSigmoid()
        self.local_sim_ax0 = HaarPSISimilarityMap(0, self.a, self.c)
        self.local_sim_ax1 = HaarPSISimilarityMap(1, self.a, self.c)
        self.wmap_ax0 = HaarPSIWeightMap(0)
        self.wmap_ax1 = HaarPSIWeightMap(1)

    def forward(self, x, y):
        wmap_ax0 = self.wmap_ax0(x, y)
        wmap_ax1 = self.wmap_ax1(x, y)

        numer = torch.sum(
            self.local_sim_ax0(x, y) * wmap_ax0 +
            self.local_sim_ax1(x, y) * wmap_ax1)
        denom = torch.sum(wmap_ax0 + wmap_ax1)

        return (self.inv_sigmoid(numer / denom) / self.a) ** 2


if __name__ == '__main__':
    from dipp.util.testutils import run_doctests
    run_doctests()
