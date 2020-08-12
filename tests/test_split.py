import json

with open('./qlist.txt', 'r') as f:
    q_list = f.read().split(',')

num = 42906

x = 0
done = False
while not done:
    x += 1
    person_list = ''
    for i in range(num):
        try:
            person_list += str(q_list.pop(0)) + ','
        except IndexError:
            done = True
            break
    person_list += '\b'
    with open('./nums%s.txt'%x, 'w') as f:
        f.write(person_list)