d= [140,154,154,154,155,180,192,192,139]
d.sort()
print(d)
x= int((len(d)+1) /2)
print(x)
print(d[x])

sum=0
for i in d:
    sum= sum+i
print(sum)
print(sum/len(d))