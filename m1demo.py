#!/usr/bin/env python
# Sample application

from sys import stderr, argv
from operator import itemgetter
from getopt import getopt, GetoptError

from m1 import M1

MIN = 0.0001

if __name__ == '__main__':
    ## parse args
    (opts, args) = getopt(argv[1:], 's:t:n:hv')
    # defaults
    n = 0
    source = None
    target = None
    # get vals from args
    try:
        for (opt, val) in opts:
            if opt == '-s':
                source = val
            elif opt == '-t':
                target = val
            elif opt == '-n':
                try:
                    n = int(val)
                except ValueError:
                    exit('-n value invalid')
            else:
                raise(GetoptError)
    except GetoptError, err:
        exit(str(err))

    ## check args
    if not source:
        exit('Source file not specified (use -s)')
    elif not target:
        exit('Target file not specified (use -t)')
    elif n < 1:
        exit('Number of iterations not specified (use -n)')

    ## train
    model = M1(source, target)
    model.iterate(n, verbose=True)

    """
    ## dump t-table
    for (sw, twtable) in model.ttable.iteritems():
        print '{0}'.format(sw),
        for (tw, val) in sorted(twtable.iteritems(), reverse=True,
                                                     key=itemgetter(1)):
            if val < MIN: continue
            print '{0}:{1:.4f}'.format(tw, val),
        print
    """
