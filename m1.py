#!/usr/bin/env python
# 
# Copyright (c) 2012 Kyle Gorman
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# ibm.py: IBM Model 1 machine translation
# Kyle Gorman <kgorman@ling.upenn.edu
# 
# Trains an IBM model one translation table from bitexts. See the included
# m1.py script for an example application. A small tokenized and uppercased
# portion of the Canadian Hansards, parlimentary proceedings in French and 
# English, is included in the data/ directory. The full set of data can be 
# found at the following URL:
# 
# <http://www.isi.edu/natural-language/download/hansard/>
# 
# The tokenization itself is taken from the following GitHub project:
# 
# <http://github.com/madmaze/Python-IBM-Model-1>
# 
# IBM Model 1 is described in the following paper: 
# 
# Brown, P., Della Pietra, V., Della Pietra, S., and Mercer, R. 1993. The
# mathematics of statistical machine translation: Parameter estimation. 
# Computational Linguistics 19(2): 263-312.
# 
# This paper is just one of many that include a pseudocode description of the 
# t-table estimation algorithm. However, my experience is that many of these 
# pseudocode algorithms are not efficient or are simply incorrect. The 
# complexity of t-table estimation depends primarily on the E-step. During the
# E-step, for each training pair, this module takes an outer loop over the 
# source words and an inner loop over the target words, as proposed by Brown 
# et al. However, some published descriptions of Model 1 call for three nested 
# loops, making training intractable for even moderate amounts of data.
# 
# In the standard description of this model my "s" ("source") is called "f" 
# ("French") and my "t" ("target") is called "e" ("English"). The Python
# None type is used to represent alignments to nulls (i.e., insertion or 
# deletion).
# 
# You will probably want to translate your training data into all uppercase or
# all lowercase characters if you're using languages that distinguish between
# the two. 
# 
# Given a trained M1 instance `model1`, `model1[None]` returns the distribution
# over possible insertions. Given a source word `X`, its deletion 
# probability is given by `model1[X][None]`. `1 - model1[None][None]` is the 
# insertion probability itself. There is no unary deletion probability: all
# deletion probabilities are lexically conditioned.
# 
# To limit memory requirements of this code, sentence-pairs are not stored in 
# memory, but rather are read online. The IO penalty appears to be minimal, as
# runs of m1.py tend to sit at near 100% CPU utilization.
# 
# That said, this code is very slow and since it is CPU-bound but also involves
# high-level Python operations, it cannot be parallelized in CPython.
# A parallel implementation of this code would involve a pool of threads 
# pulling sentence pairs from a queue during the E-step. Once the queue is 
# empty, the M-step is done by a single thread. Decoding can also be done in 
# parallel in a similar fashion. To produce a full Model 1, this code must 
# also be extended to compute a source-target fertility table (which maps from
# the number of words in the source sentence to a distribution of number of 
# words in the target sentence) and be "composed" with a language model.
#
# FIXME labels follow lines that might be better iterating over a set object,
# which contains only unique counts. I'm not sure it matters, though. 
# 
# NB: the code for decoding alignments has not been carefully debugged yet. 
# Alignments are lazily computed: an alignment for a single source/target pair
# s, t is a generator. And, a set of alignments is also a generator.

from sys import stderr
from math import log, exp
from collections import defaultdict


def bitext(source, target):
    """
    Run through the bitext files, yielding one sentence at a time.
    """
    sourcef = source if hasattr(source, 'read') else open(source, 'r')
    targetf = target if hasattr(target, 'read') else open(target, 'r')
    for (s, t) in zip(sourcef, targetf):
        yield ([None] + s.strip().split(), [None] + t.strip().split())


class M1(object):
    """
    A class wrapping an IBM Model 1 t-table. After initialization, training
    is performed by calling the iterate() method on the instance.

    >>> model = M1('data/hansards.f.small', 'data/hansards.e.small')
    >>> print round(model['gouvernement']['government'], 5) # before
    0.03069
    >>> model.iterate(5)
    >>> print round(model['gouvernement']['government'], 5) # after
    0.98719
    """

    def __init__(self, source, target):
        """
        Takes two arguments, specifying the paths (or a file-like objects
        with appropriate 'read' methods) of source and target bitexts
        """
        self.source = source
        self.target = target
        self.ttable = defaultdict(lambda: defaultdict(float)) # p(s|t)
        # compute raw co-occurrence frequencies 
        for (s, t) in bitext(self.source, self.target):
            for sw in s: # FIXME
                for tw in t: # FIXME
                    self.ttable[sw][tw] += 1
        # normalize them
        self._normalize()
        self.n = 0 # number of iterations thus far
 
    def __repr__(self):
        return 'M1({0}, {1})'.format(self.source, self.target)

    def __getitem__(self, item):
        return exp(self.ttable[item])

    def _normalize(self):
        for (sw, twtable) in self.ttable.iteritems():
            Z = sum(twtable.values())
            for tw in twtable:
                twtable[tw] = twtable[tw] / Z

    def iterate(self, n=1, verbose=False):
        """
        Perform n iterations of EM training
        """
        for i in xrange(n):
            if verbose:
                print >>stderr, 'iteration {0}...'.format(self.n)
            acounts = defaultdict(float)
            tcounts = defaultdict(float)
            ## E-step
            for (s, t) in bitext(self.source, self.target):
                for sw in s: # FIXME
                    Z = sum(self.ttable[sw].values())
                    for tw in t: # FIXME
                        # compute expectation and preserve it
                        c = self.ttable[sw][tw] / Z
                        acounts[(sw, tw)] += c
                        tcounts[tw] += c
            ## M-step
            for ((sw, tw), val) in acounts.iteritems():
                self.ttable[sw][tw] = val / tcounts[tw]
            self._normalize()
            ## wrap up
            self.n += 1

    def decode_pair(self, s, t):
        """
        Given a pair of source/target sentences s, t, output the optimal 
        alignment.
        """
        for sw in s:
            best_p = 0.
            best_a = -1
            for (i, tw) in enumerate(t):
                p = self.ttable[sw][tw]
                if p > best_p: # best one so far
                    best_p = p
                    best_a = i
            yield sw, t[best_a]

    def decode_training(self):
        """
        Generator of the optimal decodings for the training sentences
        """
        for s, t in bitext(self.source, self.target):
            yield self.decode_pair(s, t)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
