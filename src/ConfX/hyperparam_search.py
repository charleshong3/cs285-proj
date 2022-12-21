import multiprocessing as mp
import copy
import traceback
import itertools

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd

from main import main

def run_main(kwargs):
    print(kwargs)
    try:
        return main(**kwargs)
    except:
        print(traceback.format_exc())
        return None

def best_point():
    args = {
        "outdir": "outdir_best_point",
        "model": "resnet50",
        "df": "shi", 
        "dropout": 0.3,
        "d_model": 10,
        "d_hid": 200,
        "nhead": 2,
        "nlayers": 3,
        "gpu": 0,
    }
    perf = main(**args)
    return perf


def search(param, values):
    print(f"hyperparam search for param {param}, values {values}")
    args = {
        "outdir": "outdir",
        "model": "resnet50",
        "df": "shi", 
        "dropout": 0.2,
        "d_model": 10,
        "d_hid": 200,
        "nhead": 2,
        "nlayers": 2,
        "gpu": 2,
    }
    args_per_val = []
    for i, val in enumerate(values):
        val_args = copy.deepcopy(args)
        val_args[param] = val
        val_args["outdir"] = f"outdir_{param}_{val}"
        # val_args["gpu"] = i % 4
        args_per_val.append(val_args)

    # pool = mp.get_context("spawn").Pool(8)
    # perfs = pool.map(run_main, args_per_val)
    
    failing_vals = []
    working_vals = []
    perfs = []
    for i, val_args in enumerate(args_per_val):
        try:
            perf = main(**val_args)
            perfs.append(perf)
            working_vals.append(values[i])
        except:
            failing_vals.append(values[i])
            print(traceback.format_exc())
    
    print("failing vals:", failing_vals)
    print("working vals:", working_vals)
    print("perfs:", perfs)
    plt.figure()
    plt.title(param)
    plt.plot(working_vals, perfs)
    plt.xlabel("value")
    plt.ylabel("latency (cycles)")
    plt.savefig(f"hyperparam_{param}.png", bbox_inches="tight")

def grid_search():
    param_value_dict = {
        "dropout": [0.3],
        "d_model": [8, 16, 32],
        "d_hid": [64, 128, 256],
        "nhead": [1, 2, 4],
        "nlayers": [1, 2, 3],
    }
    # param_value_dict = {
    #     "dropout": [0, 0.2],
    #     "d_model": [4],
    #     "d_hid": [32],
    #     "nhead": [1],
    #     "nlayers": [1],
    # }
    print(f"grid search for params {param_value_dict}")
    args_choices = list(itertools.product(*(param_value_dict.values())))
    args = {
        "outdir": "outdir",
        "model": "resnet50",
        "df": "shi", 
        "dropout": 0.2,
        "d_model": 10,
        "d_hid": 200,
        "nhead": 2,
        "nlayers": 2,
        # "gpu": 0,
    }
    args_per_val = []
    for choice_idx, choice in enumerate(args_choices):
        val_args = copy.deepcopy(args)
        outdir = "outdir"
        
        for param_idx, param in enumerate(param_value_dict.keys()):
            val_args[param] = choice[param_idx]
            outdir += f"_{param}_{choice[param_idx]}"
        val_args["outdir"] = outdir
        # val_args["gpu"] = choice_idx % 4
        args_per_val.append(val_args)

    # pool = mp.get_context("spawn").Pool(16)
    # perfs = pool.map(run_main, args_per_val)
    perfs = []
    df = pd.DataFrame([])
    for i, val_args in enumerate(args_per_val):
        try:
            perf = main(**val_args)
            perfs.append(perf)
            val_args.update({"latency (cycles)": perf})
            row_df = pd.DataFrame([val_args])
            df = pd.concat([df, row_df])
            df.to_csv("grid_search.csv")
        except:
            perfs.append(None)
    
    # df = pd.DataFrame(args_per_val)
    # df.insert(0, "latency (cycles)", perfs)
    # df.to_csv("grid_search.csv")
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    grid_search()

    # search("dropout", [0, 0.2, 0.3, 0.4, 0.5])
    # search("d_model", [2, 4, 10, 20, 100])
    # search("d_hid", [16, 32, 64, 128, 256, 512, 1024])
    # search("nhead", [1, 2, 3, 4, 6, 8, 10])
    # search("nlayers", [1, 2, 3, 4, 6, 8, 10])