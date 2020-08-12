import json

with open('data.json', 'r') as f:
    data = json.load(f)


with open('qlist.txt', 'r') as f:
    q_list = f.read().split(',')

print(q_list.index("tt0363335"))
        
print(len(q_list))