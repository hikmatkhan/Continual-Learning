import re
import os
import imageio

in_dir = 'C:/Users/chris/PycharmProjects/VDPMain/models/VDP/04_11_21/02_56_complete/KDEs/'
layers = os.listdir(in_dir)
for layer in layers:
    layer_dir = in_dir + layer + '/'
    types = os.listdir(layer_dir)
    for type in types:
        type_dir = layer_dir + type + '/'
        filenames = os.listdir(type_dir)
        images = []
        for filename in filenames:
            images.append(imageio.imread(type_dir + filename))
        imageio.mimsave(layer_dir + '/' + type + '.gif', images, 'GIF', duration=0.05)

#
# num = int(re.search(r'Epoch_(.*?)_End', filename).group(1))
# parts = re.split(r'_\d{1,4}_', filename)
# newfile = parts[0] + f'_{num:04d}_' + parts[1]
# newPath = type_dir + newfile
# old_path = type_dir + filename
# os.rename(old_path, newPath)
# #if re.search(r'Epoch_(\d{4})_End', filename) == None:
#    os.remove(type_dir + filename)


#  if not re.search(r'Epoch_(\d{4})_End', filename) == None:
#      os.remove(type_dir + filename)