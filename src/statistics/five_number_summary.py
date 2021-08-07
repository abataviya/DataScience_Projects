l= [10,11,12,25,25,27,31,33,34,34,35,36,43,50,59]
print(len(l))
pos= int((len(l)+1)/2)
print(pos)
median_l= l[pos-1]
print(median_l)
pos_q1= int((pos+1)/2)
print(pos_q1)
quartile_1= l[pos_q1]
print(quartile_1)
pos_q3= pos+pos_q1
print(pos_q3)
quartile_3= l[pos_q3-1]
print(quartile_3)
IQR= quartile_3- quartile_1
print(IQR)
outlier_1= quartile_1-(1.5*IQR)
print(outlier_1)
outlier_2= quartile_3+(1.5*IQR)
print(outlier_2)