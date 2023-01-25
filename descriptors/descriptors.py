#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is dedicated to the computation of compositional fingerprints 
that are used by the Random Forest models used in the paper

@author: matteo
"""

import numpy as np


def computeWeightedProperties(comp):
    """
    Compute property weighted compositional features.

    Parameters
    ----------
    comp : Composition
        The function espects as input a pymatgen Composition object.

    Returns
    -------
    properties: ndarray
        array of property weighted compositional features (15).

    """

    # Properties: Z, m, r, col, row, eneg, Pn, nval, s, p, d, f, V, Tm, EA
    properties = np.zeros(15)

    for el in comp.elements:
        xi = comp.get_atomic_fraction(el)

        # Atomic number
        properties[0] += el.number * xi

        # Atomic mass
        properties[1] += el.atomic_mass * xi

        # Van Der Waals Radius
        properties[2] += el.van_der_waals_radius * xi

        # Period
        properties[3] += el.row * xi

        # Group
        properties[4] += el.group * xi

        # electronegativity
        properties[5] += el.X * xi

        # Pettifor number
        properties[6] += el.mendeleev_no * xi

        # valence electrons
        es = el.electronic_structure.split(".")
        valence = 0.0
        s = 0.0
        p = 0.0
        d = 0.0
        f = 0.0

        for i in range(1, len(es)):
            valence += np.int8(es[i][2:])

            if "s" in es[i]:
                s += np.int8(es[i][2:])

            if "p" in es[i]:
                p += np.int8(es[i][2:])

            if "d" in es[i]:
                d += np.int8(es[i][2:])

            if "f" in es[i]:
                f += np.int8(es[i][2:])

        properties[7] += valence * xi
        properties[8] += s * xi
        properties[9] += p * xi
        properties[10] += d * xi
        properties[11] += f * xi

        # Molar volume
        properties[12] += el.molar_volume * xi

        # Melting T
        properties[13] += el.melting_point * xi

        # Electron Affinity
        properties[14] += el.electron_affinity * xi

    return np.asarray(properties)


def computeCompEmbeddings(compositions, types_index=None):
    """
    Compute compositional embeddings.

    Parameters
    ----------
    compositions : list of Composition
        The function espects as input a list of Composition objects.
    types_index : list of int
        List of index associated to the elements to be removed from the output.
        If the features are used as input of a trained model this argument should 
        contain the list of elements that are not present in the training set of the model.

    Returns
    -------
    embeddings: ndarray
        2D array of compositional embeddings.

    indx: list of int
        List of index associated to the elements that were not used in the the embeddings 
        because not present within the compositions.
        Designed to be used as type_index input for subsequents call to computeCompEmbeddings.

    """

    embeddings = np.zeros((len(compositions), 150))

    for i in range(len(compositions)):
        comp = compositions[i]

        for el in comp.elements:

            xi = comp.get_atomic_fraction(el)
            embeddings[i][el.number - 1] = xi

    if types_index is None:
        indx = np.argwhere(np.all(embeddings[..., :] == 0, axis=0))
    else:
        indx = types_index

    embeddings = np.delete(embeddings, indx, axis=1)

    return embeddings, indx


def dumpCompositionalDescriptors(X, out_name="descriptors", types_index=None):
    """
    Dump compositional features.

    Parameters
    ----------
    X : list of Composition
        The function espects as input a list of Composition objects.
    out_name : str
        name of the .npy file on witch the compositional features are written.
    types_index : list of int
        List of index associated to the elements to be removed from the compositional embeddings.
        If the features are used as input of a trained model this argument should contain 
        the list of elements that are not present in the training set of the model.

    Returns
    -------
    indx: list of int
        List of index associated to the elements that were not used in the the embeddings 
        because not present within the compositions contained in X.
        Designed to be used as type_index input for subsequents call to dumpCompositionalDescriptors

    """

    weightedprop = []

    for x in X:
        weightedprop.append(computeWeightedProperties(x))
    embeddings, indx = computeCompEmbeddings(X, types_index=types_index)

    np.save(out_name + ".npy", np.concatenate([weightedprop, embeddings], axis=1))

    return indx