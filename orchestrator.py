import os

if __name__=='__main__':

	i,b = 0,5
	while True:
		os.system('./python-dock.sh ./code/mainProject.py')
		i+=1
		if i>=b:
			c=input('Continue? [Y/n]')
			if c.strip().lower()=='n':
				break
