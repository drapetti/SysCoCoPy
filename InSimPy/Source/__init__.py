#!/usr/bin/env python
# coding: utf-8

"""
File: InSimPy/Source/__init__.py
Author: David Rapetti
Date: 12 Jul 2023

Description: InSimPy contains classes to generarte and inject stellar 
variability at the pixel level for the purpose of realistically simulating
target pixel files that can be used to test the ability of systematic 
correctors in identifying this data component.

It utilizes the TESS PRF (pixel response function) as implemented in Python 
in the TESS_PRF code by Keaton Bell.
"""

from .Simulations_Injector import *
