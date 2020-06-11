read_rele = open("./rele.txt")
count = 0

for line in read_rele.readlines():
    query_id, doc_id, flag = line.strip().split('\t')
    count += int(flag)

print(count)