import os
import random


for name in os.listdir("/data/"):
	if 'txt' in name:
		continue

	path, dirs, files = next(os.walk("/data/"+name))
	file_count = len(files)
	print(file_count)
	if(file_count==2):
		a = []
		for file in os.listdir("/data/" + name):
			a.append(file)
		with open("pairs.txt","a") as f:
			pair_two_list=[[0,1]]
			temp = random.choice(a).split("_")[0] # This line may vary depending on how your images are named.
			n0 = a[pair_two_list[0][0]].split("_")[1].lstrip("0").rstrip(".png")
			n1 = a[pair_two_list[0][0]].split("_")[1].lstrip("0").rstrip(".png")
			f.write(temp + "\t" + n0 + "\t" +n1 + "\n")
	if (file_count == 3):
		a = []
		for file in os.listdir("/data/" + name):
			a.append(file)
		with open("pairs.txt", "a") as f:
			pair_two_list = [[0, 1],[0,2],[1,2]]
			temp = random.choice(a).split("_")[0] # This line may vary depending on how your images are named.
			for l1 in pair_two_list:
				n0 = a[l1[0]].split("_")[1].lstrip("0").rstrip(".png")
				n1 = a[l1[1]].split("_")[1].lstrip("0").rstrip(".png")
				f.write(temp + "\t" + n0 + "\t" + n1 + "\n")
	if (file_count == 4):
		a = []
		for file in os.listdir("/data/" + name):
			a.append(file)
		with open("pairs.txt", "a") as f:
			pair_two_list = [[0, 1],[0,2],[0,3],[1,2],[1,3],[2,3]]
			temp = random.choice(a).split("_")[0] # This line may vary depending on how your images are named.
			for l1 in pair_two_list:
				n0 = a[l1[0]].split("_")[1].lstrip("0").rstrip(".png")
				n1 = a[l1[1]].split("_")[1].lstrip("0").rstrip(".png")
				f.write(temp + "\t" + n0 + "\t" + n1 + "\n")
	if (file_count >= 5):
		a = []
		for file in os.listdir("/data/" + name):
			a.append(file)
		with open("pairs.txt", "a") as f:
			pair_two_list = [[0, 1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
			temp = random.choice(a).split("_")[0] # This line may vary depending on how your images are named.
			for l1 in pair_two_list:
				n0 = a[l1[0]].split("_")[1].lstrip("0").rstrip(".png")
				n1 = a[l1[1]].split("_")[1].lstrip("0").rstrip(".png")
				f.write(temp + "\t" + n0 + "\t" + n1 + "\n")



###   make all mismatches    ###
for i,name in enumerate(os.listdir("/data/")):
	remaining = os.listdir("/data/")
	del remaining[i] # deletes the file from the list, so that it is not chosen again
	other_dir = random.choice(remaining)
	with open("pairs.txt","a") as f:
		flag=1
		for i in range(5):
			file1 = random.choice(os.listdir("/data/" + name))
			file2 = random.choice(os.listdir("/data/" + other_dir))
#path, dirs, files = next(os.walk("datasets/clfnew160//" + name))
#file_count = len(files)
#if(file_count ==1 and flag==1):
# f.write(name + "\t" + file1.split("")[1].lstrip("0").rstrip(".png") + "\t" +other_dir + "\t" + file2.split("")[1].lstrip("0").rstrip(".png") + "\n")
# flag=0
#if (file_count !=1):
			f.write(name + "\t" + file1.split("_")[1].lstrip("0").rstrip(".png") + "\t" + other_dir + "\t" + file2.split("_")[1].lstrip("0").rstrip(".png") + "\n")
