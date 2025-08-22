# Multiprocessing
import os
import h5py
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path


from plato_copilot.utils.time_utils import Timer

def run_script(args):
    process_id, data_folder, num_rollouts = args
    # Construct the command to run your script with the desired arguments
    command = [
        'python', 'kinodynamic_scripts/jenga_data_generation_sp.py',
        '--process-id', str(process_id),
        '--data-folder', data_folder,
        '--num-rollouts', str(num_rollouts)
    ]

    # Run the command
    subprocess.run(command)

def main():
    parser = argparse.ArgumentParser(description='Generate Jenga data') 
    parser.add_argument('--num-processes', type=int, default=10, help='number of processes')
    parser.add_argument('--data-folder', required=True, help='data folder')
    parser.add_argument('--num-rollouts', default=4, type=int)
    args = parser.parse_args()

    num_processes = args.num_processes
    data_folder = args.data_folder
    os.makedirs(data_folder, exist_ok=True)
    num_processes = args.num_processes
    arguments = [(i, data_folder, args.num_rollouts) for i in range(num_processes)]

    with Timer(verbose=True):
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(run_script, arguments)

    file_names = []
    final_demo_file = h5py.File(f"{args.data_folder}/data.hdf5", "w")
    data_grp = final_demo_file.create_group("data")
    num_traj = 0
    for demo_file in sorted(Path(args.data_folder).glob("0*.hdf5")):
        file_names.append(demo_file)
        demo_file = h5py.File(demo_file, "r")
        # copy the data group directly
        for key in demo_file["data"]:
            ep_grp = data_grp.create_group(f"traj_{num_traj}")
            ep_grp.attrs["xml"] = demo_file["data"][key].attrs["xml"]
            ep_grp.attrs["block_name"] = demo_file["data"][key].attrs["block_name"]
            # ep_grp.attrs["info"] = demo_file["data"][key].attrs["info"]
            for k in demo_file["data"][key]:
                ep_grp.create_dataset(k, data=demo_file["data"][key][k])

            num_traj += 1

    print(final_demo_file["data"].keys())

    final_demo_file.close()

if __name__ == "__main__":
    main()
    