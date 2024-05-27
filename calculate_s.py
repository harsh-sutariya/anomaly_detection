import os
import torch
from pretrain_data import preprocess_modelnet10, compute_average_neighbor_distance

def main():
    destination_dir = "./modelnet10"
    num_points = 1024
    k = 8
    output_file = "average_distance.txt"

    point_clouds = preprocess_modelnet10(download=False, num_points=num_points)
    print(f"Processed {len(point_clouds)} point clouds.")
    
    average_distance = compute_average_neighbor_distance(point_clouds, k=k)
    print(f"Computed average neighbor distance: {average_distance}")

    with open(output_file, 'w') as f:
        f.write(f"Average Neighbor Distance: {average_distance}")

    print(f"Average distance saved to {output_file}")

if __name__ == "__main__":
    main()
