with open('interactions.txt', 'r') as file :
  filedata = file.read()

filedata = filedata.replace(',', ' ')

with open('interactions.txt', 'w') as file:
  file.write(filedata)
