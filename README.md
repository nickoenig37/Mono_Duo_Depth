# Mono_Duo_Depth

## How to activate venv (in the neural_network directory) 
- For windows: `source venv/Scripts/activate`
- For ubuntu: 

## Which pytorch to install in venv
- 

## Problems
- Depth sensor is a bit above the two mono cameras, so it is capturing a bit more than what the mono cameras see
- All cameras are not the same FOV. This is a problem since the depth sensor is seeing more than the monocular cameras

## Speed Parameters tested
- num_workers: Between 2-6 seems to work well, as these consistently use under 90% RAM. Using 8 pushed the RAM to the limit which made training slow.
    - In testing, using anything between 2-6 had the same training speed, but this is because having extra workers were not needed. If extra CPU computations were added (such as colour jitter), then these extra workers woul be needed, which is why setting it to the max is best.
    - Using 2 workers -> 6-7 min
    - Using 4 workers -> 5 min
    - Using 6 workers -> 5 min (since there is no change in time, this means the GPU is running as fast as it can, we have fully optimized it)
    - Using 8 workers -> 6 min
- batch_size: 8 seems to work well, as it uses 4.9/8GB of GPU RAM. Using 16 pushed it to 8/8GB which made training slow, and using 4 clearly would make it less efficient as more RAM could be used. There could be a possibility to move it to using 12 in the future.
    - Using < 8 batch size -> we know will be longer than 6 min
    - Using 8 batch size -> 6 min
    - Using 16 batch size -> 12 min
- overall: These set of parameters consistently pushes the GPU to 100% usage and under 75% GPU usage, while pushing CPU usage to 20-40% and 80-90% RAM usage, which are safe thresholds

## Hyperparameters tested
- Type of scheduler: Both ReduceLROnPlateau and CosineAnnealingLR were tested, with ReduceLROnPlateau providing approximately 7% better loss.