from src.pytorch.main import ModelTypes, run

if __name__ == "__main__":
    # Optional but fine:
    # import torch.multiprocessing as mp
    # mp.set_start_method("spawn", force=True)

    run(ModelTypes.DCGAN)