gif_file = open('raw_gifs.tsv')
lines = gif_file.readlines()
for line in lines:
    print(line.split()[0])
