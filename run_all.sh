#!/bin/bash

#Duffing

# start_time=$(date +"%s")

# echo "Running DyRC_VG.py..."
# python Duffing/DyRC_VG.py
# end_time=$(date +"%s")
# duration=$((end_time - start_time))
# echo "Script execution time: ${duration} seconds"

# start_time=$(date +"%s")

# echo "Running DyRC_VG_16.py..."
# python Duffing/DyRC_VG_16.py
# end_time=$(date +"%s")
# duration=$((end_time - start_time))
# echo "Script execution time: ${duration} seconds"

# start_time=$(date +"%s")

# echo "Running DyRC_VG_16_compare_sparsity.py..."
# python Duffing/DyRC_VG_16_compare_sparsity.py
# end_time=$(date +"%s")
# duration=$((end_time - start_time))
# echo "Script execution time: ${duration} seconds"

# #Lorenz

# start_time=$(date +"%s")

# echo "Running Lorenz_DyRC_VG.py..."
# python Lorenz/Lorenz_DyRC_VG.py
# end_time=$(date +"%s")
# duration=$((end_time - start_time))
# echo "Script execution time: ${duration} seconds"

# start_time=$(date +"%s")

# echo "Running Lorenz_DyRC_VG_16.py..."
# python Lorenz/Lorenz_DyRC_VG_16.py
# end_time=$(date +"%s")
# duration=$((end_time - start_time))
# echo "Script execution time: ${duration} seconds"

# start_time=$(date +"%s")

# echo "Running Lorenz_DyRC_VG_16_compare_sparsity.py..."
# python Lorenz/Lorenz_DyRC_VG_16_compare_sparsity.py
# end_time=$(date +"%s")
# duration=$((end_time - start_time))
# echo "Script execution time: ${duration} seconds"



# #Mackey-Glass

# start_time=$(date +"%s")

# echo "Running MG_DyRC_VG.py..."
# python MackeyGlass/MG_DyRC_VG.py
# end_time=$(date +"%s")
# duration=$((end_time - start_time))
# echo "Script execution time: ${duration} seconds"

start_time=$(date +"%s")

echo "Running MG_DyRC_VG_16.py..."
python MackeyGlass/DyRC_VG_16.py
end_time=$(date +"%s")
duration=$((end_time - start_time))
echo "Script execution time: ${duration} seconds"

start_time=$(date +"%s")
echo "Running MG_DyRC_VG_16_compare_sparsity.py..."
python MackeyGlass/MG_DyRC_VG_16_compare_sparsity.py
end_time=$(date +"%s")
duration=$((end_time - start_time))
echo "Script execution time: ${duration} seconds"


echo "All programs finished."