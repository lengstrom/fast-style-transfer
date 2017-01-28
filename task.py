import evaluate
import glob
import os

'''
By Feilong
Copyright (C) 2017 FeiLong"
This is a small conttribution to the porject 
task.py helps in processing multiple images. You just need 
to place all your images inside the images folder and create
a done folder for storing the generated images.

Note that I have already added the rain-princess.ckpt, scream.ckpt, wave.ckpt, udnie.ckpt, la-muse.ckpt and wreck.ckpt in the root folder.
'''

dir_path = os.path.dirname(os.path.realpath(__file__))
ckpt_names =  glob.glob(dir_path+"/*.ckpt")
image_names = glob.glob(dir_path+"/images/*.*")

print "Starting"
for image_path in image_names:
	temp = image_path.split('/')
	in_path = temp[len(temp)-1]
	for file in ckpt_names:
		temp2 = file.split('/')
		checkpoint_dir = temp2[len(temp2)-1]
		suffix = checkpoint_dir.split('.')
		out_path= suffix[0]+"_"+in_path
		evaluate.ffwd_to_img("./images/"+in_path, "./done/art_"+out_path, "./"+checkpoint_dir)	
print "Completed!"



