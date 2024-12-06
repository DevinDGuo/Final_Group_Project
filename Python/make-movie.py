import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import struct
from matplotlib import colormaps
import argparse
import sys

# Parse command-line arguments
if len(sys.argv) != 3:
    print('Usage: python make-movie.py <input .dat file> <output movie MP4 file>')
    sys.exit(1)

input_file = sys.argv[1]
output_movie = sys.argv[2]

# Read binary matrices
matrices = []
with open(input_file, 'rb') as f:
    dim_data = f.read(8)
    if len(dim_data) < 8:
        sys.exit("Error: File is too short to contain valid dimensions.")
    
    num_rows = struct.unpack('i', dim_data[:4])[0]
    num_cols = struct.unpack('i', dim_data[4:])[0]
    print(f"First matrix dimensions: num_rows={num_rows}, num_cols={num_cols}")

    num_elements = num_rows * num_cols
    iteration = 0  
    while True:
        matrix_data = f.read(num_elements * 8)
        if len(matrix_data) < num_elements * 8:
            break
        
        matrix = np.frombuffer(matrix_data, dtype=np.float64).reshape((num_rows, num_cols))
        matrices.append(matrix)
        print(f"Iteration {iteration}")
        iteration += 1  

# Create movie
fig, ax = plt.subplots()
ims = []

cmap = colormaps['coolwarm']
for matrix in matrices:
    im = ax.imshow(matrix, animated=True, cmap=cmap, vmin=0, vmax=1)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
ani.save(output_movie, writer='ffmpeg', fps=60)

print("Movie created at, " + output_movie + ".")
