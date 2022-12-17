import csv
import random

def generate_random_array(row, col):
    a = []
    for i in range(100):
        l = [i]
        for j in range(row):
            l.append(random.sample(range(-10, -5), 2))
        a.append(l)
        
    for i in range(200):
        l = [i]
        for j in range(row):
            l.append(random.sample(range(-5, 5), 2))
        a.append(l)
        
    for i in range(200):
        l = [i]
        for j in range(row):
            l.append(random.sample(range(5, 10), 2))
        a.append(l)
        
    return a

if __name__ == '__main__':
    row = 1
    col = 500

    array = generate_random_array(row, col)

    f = open('sample.csv', 'w')
    w = csv.writer(f, lineterminator='\n')
    w.writerows(array)
    f.close()