import torch
torch.cuda.empty_cache()

import world
from utils.utils import set_seed
from utils.dataloader import load_data
from model.RecModel import socialRecModel, DenoiseModel
from utils.train_loop import TrainLoop

import sys

if __name__ == "__main__":
    print(f"Using {world.device} device: {torch.cuda.current_device()}")
    set_seed(world.seed)
    
    dataset = load_data(world.dataset)
    print('Max: ', torch.cuda.max_memory_allocated())
    print('Memory: ', torch.cuda.max_memory_allocated())
    model = socialRecModel(world.args, 
                           dataset.n_users, dataset.n_items,
                           dataset.getInterGraph())
    model = model.to(world.device)
    print('Max: ', torch.cuda.max_memory_allocated())
    print('Memory: ', torch.cuda.max_memory_allocated())
    
    
    model.interGraph = model.interGraph.to(world.device)
    denoise_model = DenoiseModel(world.args)
    denoise_model = denoise_model.to(world.device)
    TrainLoop(world, 
              dataset,
              model,
              denoise_model).run_loop()
    