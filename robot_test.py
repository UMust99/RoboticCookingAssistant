import subprocess

python2_command = 'C:\Python27\python.exe part1.py'
process = subprocess.Popen(python2_command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
#print(output)

python3_command = 'C:/Users/jozef/anaconda3/python.exe nn.py'
process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
#print(output)

python2_command = 'C:\Python27\python.exe part2.py'
process = subprocess.Popen(python2_command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
#print(output)
