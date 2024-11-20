from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

def read_tensorboard_log(log_file1, log_file2, log_file3):
    event_acc1 = EventAccumulator(log_file1)
    event_acc1.Reload()
    event_acc2 = EventAccumulator(log_file2)
    event_acc2.Reload()
    event_acc3 = EventAccumulator(log_file3)
    event_acc3.Reload()

    tags1 = event_acc1.Tags()['scalars']
    tags2 = event_acc2.Tags()['scalars']
    tags3 = event_acc3.Tags()['scalars']
    print("Available scalar tags in the log file:", tags1)
    print("Available scalar tags in the log file:", tags2)
    print("Available scalar tags in the log file:", tags3)

    # Assuming 'loss' is the tag for the loss function
    loss_events1 = event_acc1.Scalars('intensity_recon_loss')
    steps1 = [event.step for event in loss_events1]
    values1 = [event.value for event in loss_events1]

    loss_events2 = event_acc2.Scalars('intensity_recon_loss')
    steps2 = [event.step for event in loss_events2]
    values2 = [event.value for event in loss_events2]

    loss_events3 = event_acc3.Scalars('intensity_recon_loss')
    steps3 = [event.step for event in loss_events3]
    values3 = [event.value for event in loss_events3]


    plt.figure(figsize=(10, 6))
    plt.plot(steps1[:500], values1[:500], label='TPD-MAE Loss Curve', color='red')
    plt.plot(steps2[:500], values2[:500], label='CMAE Loss Curve', color='blue')
    plt.plot(steps3[:500], values3[:500], label='AE Loss Curve', color='green')
    plt.xlabel('Steps', fontsize=15)
    plt.ylabel('Total loss', fontsize=15)
    plt.title('Loss Curve on TotalSegmentator', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.savefig('/home/jerry/paper/loss_curve.png', format='png')
    plt.show()
# Path to the TensorBoard log file
log_file_path1 = '/home/jerry/github/CTAE-semimask/segmentation/checkpoints/mit_b2/mask_ratio_0.25_0.5/5/events.out.tfevents.1723921135.jerry.94076.0'
log_file_path2 = '/home/jerry/github/CTAE-semimask/segmentation/checkpoints/mit_b2/mask_ratio_0.25_0.5/6/events.out.tfevents.1723946286.jerry.112823.0'
log_file_path3 = '/home/jerry/github/CTAE-semimask/segmentation/checkpoints/mit_b2/mask_ratio_0.25_0.5/2/events.out.tfevents.1723583995.jerry.9022.0'

read_tensorboard_log(log_file_path1, log_file_path2, log_file_path3)
