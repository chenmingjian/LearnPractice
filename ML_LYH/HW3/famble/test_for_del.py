import os

def del_file(path):
    count = 0
    min = 1000
    for fn in os.listdir(path):
        count = count+1
        if min > int(fn[6:-3]):
            min = int(fn[6:-3])
            print(min)
            min_fn = fn
        if count > 2:
            os.remove(path+'/'+min_fn)
    return

del_file("ec_model")
