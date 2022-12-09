import csv
import random

def generate_random_array(row, col):
    a = []
    for i in range(col):
        l = [i]
        for j in range(row):
            l.append(random.sample(range(-10, 10), 2))
        a.append(l)
    return a

if __name__ == '__main__':
    row = 500
    col = 1

    array = generate_random_array(row, col)

    f = open('sample.csv', 'w')
    w = csv.writer(f, lineterminator='\n')
    w.writerows(array)
    f.close()