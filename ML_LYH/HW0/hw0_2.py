#-*-code:utf-8-*-
counter = {}

file_obj = open('data/words.txt')
try:
	file_context = file_obj.read()
finally:
	file_obj.close()
fc_list = file_context.split()


for i in fc_list:
	if i in counter:
		counter[i] += 1
	else:
		counter[i] = 1

file_obj = open("output.txt",'w')

j=0
for i in counter:
	j+=1
	print(i+' '+ '%s'%j + ' ' + '%s'%counter[i], file = file_obj)

file_obj.close;
