import os
import subprocess

# Change to the 'code' directory
os.chdir("../code")

# Run 'make clean all' to compile the code
subprocess.run(["make", "clean", "all"])

print("Making matrix of size 5 x 4.")

# Run './make-2d 5000 5000 initial.dat'
subprocess.run(["./make-2d", "5", "4", "initial.dat"])

# Run './print-2d initial.dat'
subprocess.run(["./print-2d", "initial.dat"])

print("-------------------------------------")

print("Running serial stencil...")
subprocess.run(["./stencil-2d", "10", "initial.dat", "final.dat", "all.dat"])

subprocess.run(["python3", "../python/make-movie.py", "all.dat", "sample_movie.mp4"])

print("-------------------------------------")

# Run './pth-stencil-2d 25 initial.dat final.dat 1 4'
print("Running pthread stencil...")
subprocess.run(["./pth-stencil-2d", "10", "initial.dat", "final.dat", "1", "4"])

# Run 
subprocess.run(["./print-2d", "final.dat"])

print("-------------------------------------")

# Run './omp-stencil-2d 25 initial.dat final.dat 1 4'
print("Running OpenMP stencil...")
subprocess.run(["./omp-stencil-2d", "10", "initial.dat", "final1.dat", "1", "4"])

# Run 
subprocess.run(["./print-2d", "final1.dat"])

# Run 'make clean all' to compile the code
subprocess.run(["make", "clean"])

os.chdir("../Python")

# Ask if the user wants to run the experimental gather and plot data
experimentalGather_Plot = input("Would you like to run the programs for gathering and plotting the experimental data? (Y/N): ")

while True:
    experimentalGather_Plot = input("Would you like to run the programs for gathering and plotting the experimental data? (Y/N): ").upper()
    
    if experimentalGather_Plot == "Y":
        subprocess.run(["python3", "gather_and_plot.py"])
        break  # Exit loop after running the script
    elif experimentalGather_Plot == "N":
        print("Skipping the gathering and plotting process.")
        break  # Exit loop if user chooses to skip
    else:
        print("Your answer must be Y or N.")

# Change back to the original directory if needed
os.chdir("..")