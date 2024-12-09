import argparse
from os.path import join

import numpy as np
from matplotlib import pyplot as plt, patches

box_colours = ['g',
               'b',
               'r',
               'magenta']


def get_arg_parser():
    """
    Set up the parser for the main script.

    :return: parser.parse_args()
    """
    # Path parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp',
                        '--train_path',
                        type=str,
                        required=True,
                        help='Path to training folder')
    parser.add_argument('-vp',
                        '--val_path',
                        type=str,
                        required=True,
                        help='Path to validation folder')
    parser.add_argument('-sp',
                        '--save_path',
                        type=str,
                        required=True,
                        help='Path to save folder')
    # Training parameters.
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=1000,
                        help='Training epochs')
    parser.add_argument('-we',
                        '--warmup_epochs',
                        type=int,
                        default=5,
                        help='Epochs before checking for early stopping.')
    parser.add_argument('-is',
                        '--image_size',
                        type=int,
                        default=600,
                        help='Scaled image size (image_size, image_size)')
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=8,
                        help='Training batch size')
    parser.add_argument('-p',
                        '--patience',
                        type=int,
                        default=100,
                        help='Training patience before early stopping')
    parser.add_argument('-pd',
                        '--patience_delta',
                        type=int,
                        default=0.005,
                        help='Training patience delta')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=0.01,
                        help='Optimiser learning rate')
    parser.add_argument('-lres',
                        '--learning_restart',
                        type=int,
                        default=150,
                        help='Learning rate schedular restart frequency.')
    parser.add_argument('-m',
                        '--momentum',
                        type=float,
                        default=0.9,
                        help='Optimiser momentum')
    parser.add_argument('-wd',
                        '--weight_decay',
                        type=float,
                        default=0.005,
                        help='Optimiser weight decay')
    parser.add_argument('-bw',
                        '--box_weight',
                        type=float,
                        default=7.5,
                        help='Weight applied to box loss')
    parser.add_argument('-cw',
                        '--cls_weight',
                        type=float,
                        default=0.5,
                        help='Weight applied to classification loss')
    parser.add_argument('-of',
                        '--oversampling_factor',
                        type=int,
                        default=1,
                        help='How much oversampling is desired')

    return parser.parse_args()


def plot_losses(best_epoch, training_losses, training_cls_losses, training_bbox_losses, validation_losses,
                validation_cls_losses,
                validation_bbox_losses, training_learning_rates, save_path):
    """
    Plot the training losses (combined, cls, and bbox) and validation losses (combined, cls, and bbox) along with the
    learning rates. The figure will be saved at save_path/losses.png. The losses and rates should be in a list
    that grows as the epochs increase.


    :param best_epoch: Best epoch for special marker.
    :param training_losses: List of training losses.
    :param training_cls_losses:
    :param training_bbox_losses:
    :param validation_losses: List of validation losses.
    :param validation_cls_losses:
    :param validation_bbox_losses:
    :param training_learning_rates: List of optimiser learning rates.
    :param save_path: Directory to save image into.
    """
    epochs = range(1, len(training_losses) + 1)
    _, ax = plt.subplots(nrows=2, ncols=3, layout='constrained', figsize=(16, 9), dpi=200)

    ax[0, 0].set_title('Training Classification Losses')
    ax[0, 0].plot(epochs, training_cls_losses, marker='*')
    ax[0, 0].axvline(x=best_epoch, color='green', linestyle='--')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')

    ax[0, 1].set_title('Training Box Regression Losses')
    ax[0, 1].plot(epochs, training_bbox_losses, marker='*')
    ax[0, 1].axvline(x=best_epoch, color='green', linestyle='--')

    ax[0, 2].set_title('Training Losses (weighted)\n'
                       'with Learning Rate')
    ax[0, 2].plot(epochs, training_losses, marker='*')
    ax_lr = ax[0, 2].twinx()
    ax_lr.plot(epochs, [i * 100 for i in training_learning_rates], color='red', label='learning rate')
    ax[0, 2].axvline(x=best_epoch, color='green', linestyle='--')
    ax_lr.set_ylabel('Learning Rate x10$^{-2}$')
    ax_lr.legend(loc='upper right')

    ax[1, 0].set_title('Validation Classification Losses')
    ax[1, 0].plot(epochs, validation_cls_losses, marker='*')
    ax[1, 0].axvline(x=best_epoch, color='green', linestyle='--')
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel('Loss')

    ax[1, 1].set_title('Validation Box Regression Losses')
    ax[1, 1].plot(epochs, validation_bbox_losses, marker='*')
    ax[1, 1].axvline(x=best_epoch, color='green', linestyle='--')
    ax[1, 1].set_xlabel('Epoch')

    ax[1, 2].set_title('Validation Losses (unweighted)')
    ax[1, 2].plot(epochs, validation_losses, marker='*')
    ax[1, 2].axvline(x=best_epoch, color='green', linestyle='--', label='Best Validation Epoch')
    ax[1, 2].set_xlabel('Epoch')
    ax[1, 2].legend(loc='upper right')

    plt.savefig(join(save_path, 'losses.png'))
    plt.close()


def plot_validation_results(validation_detections, validation_images, starting_label, counter, save_path):
    """
    Draw input images with detected bounding boxes on them. Only the top scoring box of each label/class
    is displayed. Since FasterRCNN using label 0 for background and RetinaNet using label 0 for the first
    class, there is an offset that is set using starting_label (for selecting box colour).

    :param validation_detections: Detection returned by the model in eval() mode.
    :param validation_images: Images that were given to the model for detection.
    :param starting_label: Lowest label value (RetinaNet = 0, FasterRCNN = 1 since 0 is background).
    :param counter: Image counter, based on batch_size, for saving images with unique names while maintaining
                    validation dataset size.
    :param save_path: Save directory.
    """
    batch_number = counter
    # Since batches are used, detections per image are delt with incrementally.
    for index, output in enumerate(validation_detections):
        # Highest scoring box per label.
        highest_scoring_boxes = {}

        for i in range(len(output['scores'])):
            label = output['labels'][i].item()
            score = output['scores'][i].item()
            box = output['boxes'][i].tolist()

            if label not in highest_scoring_boxes:
                highest_scoring_boxes[label] = {'score': score, 'box': box}
            else:
                if score > highest_scoring_boxes[label]['score']:
                    highest_scoring_boxes[label] = {'score': score, 'box': box}

        # Sort by label.
        sorted_highest_scoring_boxes = dict(sorted(highest_scoring_boxes.items()))

        _, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(np.transpose(validation_images[index].to('cpu'), (1, 2, 0)))

        for label, result in sorted_highest_scoring_boxes.items():
            box = result['box']
            patch = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                      edgecolor=box_colours[label - starting_label], facecolor='none')
            ax.add_patch(patch)
            ax.text(box[0], box[1], f'{label}', ha='left', color=box_colours[label - starting_label], weight='bold',
                    va='bottom')

        plt.savefig(join(save_path, f'val_result_{batch_number}.png'))
        plt.close()
        batch_number += 1


def test_transforms():
    # _, ax = plt.subplots(2)
    # pb = target['boxes'].data[0].to('cpu')
    # bb = target['boxes'].data[1].to('cpu')
    # ax[0].imshow(img)
    # pp = patches.Rectangle((pb[0], pb[1]), pb[2] - pb[0], pb[3] - pb[1], linewidth=1,
    #                        edgecolor='g', facecolor='none')
    # bp = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1,
    #                        edgecolor='b', facecolor='none')
    # ax[0].add_patch(pp)
    # ax[0].add_patch(bp)
    # pb = target['boxes'].data[0].to('cpu')
    # bb = target['boxes'].data[1].to('cpu')
    # ax[1].imshow(np.transpose(img, (1, 2, 0)))
    # pp = patches.Rectangle((pb[0], pb[1]), pb[2] - pb[0], pb[3] - pb[1], linewidth=1,
    #                        edgecolor='g', facecolor='none')
    # bp = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1,
    #                        edgecolor='b', facecolor='none')
    # ax[1].add_patch(pp)
    # ax[1].add_patch(bp)
    # plt.show()
    pass
