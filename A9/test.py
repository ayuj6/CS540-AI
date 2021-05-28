from funny_puzzle import *
import sys
import time

if __name__ == '__main__':

    # set this to "new" for updated version
    # that demands requeuing
    # set this to "original"
    # for the crappy one
    ref_version = "new"

    # this is necessary to print to file
    backup_stdout = sys.stdout
    sys.stdout = open('test.txt', 'w')
    starttime = time.time()

    # these are a bunch of random test cases
    
    solve([6, 4, 7, 8, 5, 0, 3, 2, 1])
    solve([8, 2, 7, 3, 1, 6, 4, 0, 5])
    solve([6, 5, 7, 1, 8, 2, 3, 4, 0])
    solve([8, 6, 5, 7, 1, 4, 0, 2, 3])
    solve([6, 4, 8, 2, 3, 5, 1, 7, 0])
    solve([7, 8, 6, 5, 4, 0, 2, 3, 1])
    solve([6, 5, 7, 4, 0, 2, 3, 1, 8])
    solve([3, 6, 2, 5, 8, 0, 4, 7, 1])
    #solve([8, 6, 7, 2, 5, 4, 3, 0, 1])

    print_succ([6, 4, 7, 8, 5, 0, 3, 2, 1])
    print_succ([8, 2, 7, 3, 1, 6, 4, 0, 5])
    print_succ([6, 5, 7, 1, 8, 2, 3, 4, 0])
    print_succ([8, 6, 5, 7, 1, 4, 0, 2, 3])
    print_succ([6, 4, 8, 2, 3, 5, 1, 7, 0])
    print_succ([7, 8, 6, 5, 4, 0, 2, 3, 1])
    print_succ([6, 5, 7, 4, 0, 2, 3, 1, 8])
    print_succ([3, 6, 2, 5, 8, 0, 4, 7, 1])

    # These are unsolvable
    #solve([7, 2, 8, 5, 0, 4, 3, 6, 1])
    #solve([1, 2, 3, 4, 0, 6, 7, 8, 5])
    #solve([8, 7, 5, 6, 0, 4, 3, 2, 1])

    # these are examples from the writeup
    print_succ([1, 2, 3, 4, 5, 0, 6, 7, 8])
    solve([4, 3, 8, 5, 1, 6, 7, 2, 0])
    print_succ([8, 7, 6, 5, 4, 3, 2, 1, 0])
    solve([1, 2, 3, 4, 5, 6, 7, 8, 0])
    solve([1, 2, 3, 4, 5, 6, 7, 0, 8])

    endtime = time.time()

    sys.stdout.close()
    sys.stdout = backup_stdout
    live = open('test.txt', 'r')
    if ref_version == "new":
        ref = open('ref.txt', 'r')
    else:
        ref = open('badref.txt', 'r')

    live_l = live.readlines()
    ref_l = ref.readlines()

    live.close()
    ref.close()

    line_no = 0
    for l, r in zip(live_l, ref_l):
        line_no += 1
        if not l == r:
            print("Yours: " + l.strip())
            print("Ref:   " + r.strip())
            print("Line number: " + str(line_no))
            print("\n")
    if not live_l:
        print("Did you... print anything? Check test.txt")
    elif len(live_l) < len(ref_l):
        print("Looks like you didn't print enough stuff")
    elif len(live_l) > len(ref_l):
        print("Looks like you printed too much stuff")
    print("If the time below is all you see, your code is good.")

    print("Elapsed time was: " + str((endtime - starttime)) + "s")
    print("Ref elapsed time: 0.62s")