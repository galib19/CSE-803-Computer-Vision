import pickle
import matplotlib.pyplot as plt
from softmax import *
from train import *


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict


def gradient_ascent(model, target_class, init, learning_rate=1):
    """
    Inputs:
    - model: Image classifier.
    - target_class: Integer, representing the target class the fooling image
      to be classified as.
    - init: Array, shape (1, Din), initial value of the fooling image.
    - learning_rate: A scalar for initial learning rate.

    Outputs:
    - image: Array, shape (1, Din), fooling images classified as target_class
      by model
    """
    scores_correct = []
    scores_target = []

    image = init.copy()
    y = np.array([target_class])
    ###########################################################################
    # TODO: perform gradient ascent on your input image until your model      #
    # classifies it as the target class, get the gradient of loss with        #
    # respect to your input image by model.forwards_backwards(imgae, y, True) #
    ###########################################################################
    i = 0
    while True:
        i += 1
        scores = model.forwards_backwards(image)
        scores_correct.append(scores[0][y + 1])
        scores_target.append(scores[0][y])
        index = np.argmax(scores, axis=1)
        if (index != y):
            dx = model.forwards_backwards(image, y, True)
            image = image - learning_rate * dx
            dx = 0
        else:
            break
            ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    plt.plot(scores_correct, label='Max. Score of Correct Class')
    plt.plot(scores_target, label='Max. Score of Target Class')
    plt.legend(loc='upper right')
    plt.xlabel("Epochs")
    plt.ylabel("Max. Scores of Classes")
    plt.savefig('t.jpg')


    return image


def img_reshape(flat_img):
    # Use this function to reshape a CIFAR 10 image into the shape 32x32x3,
    # this should be done when you want to show and save your image.
    return np.moveaxis(flat_img.reshape(3, 32, 32), 0, -1)


def main():
    # Initialize your own model
    model = SoftmaxClassifier(hidden_dim=64)
    config = {'learning_rate': 0.1,
              'lr_decay': 0.95, 'num_epochs': 100,
              'batch_size': 500, 'print_every': 1000}
    target_class = None
    correct_class = None
    correct_image = None
    ###########################################################################
    # TODO: load your trained model, correctly classified image and set your  #
    # hyperparameters, choose a different label as your target class          #
    ###########################################################################
    # loading the model and setting:
    model_loaded = unpickle('model3')
    for k in model_loaded:
        model.params[k] = model_loaded[k]
    # getting correctly classified image
    test_batch = unpickle("cifar-10-batches-py/test_batch")
    X_test = test_batch['data']
    Y_test = test_batch['labels']
    correct_image_found = False
    i = 5
    while not correct_image_found:
        temp_image_scores = model.forwards_backwards(X_test[i, :].reshape(1, -1))
        if np.argmax(temp_image_scores, axis=1) == Y_test[i]:
            correct_image = X_test[i, :].reshape(1, -1)
            correct_class = Y_test[i]
            correct_image_found = True
        i += 1
    # choosing a differnt level as target class:  target class = actual_class - 1
    if correct_class == 0:
        target_class = 9
    else:
        target_class = correct_class - 1
    # making fooling image
    fooling_image = gradient_ascent(model, target_class, init=correct_image)
    correct_image = img_reshape(correct_image)
    plt.imshow((correct_image))
    plt.imshow((correct_image * 255).astype(np.uint8))
    # plt.savefig('real.jpg')
    plt.show()
    fooling_image = img_reshape(fooling_image)
    plt.imshow((fooling_image))
    plt.imshow((fooling_image * 255).astype(np.uint8))
    plt.show()
    # plt.savefig('fake.jpg')
    # print(fooling_image[0], correct_image[0])

    # plt.show()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    ###########################################################################
    # TODO: compute the (magnified) difference of your original image and the #
    # fooling image, save all three images for your report                    #
    ###########################################################################
    diff = 10 * (fooling_image - correct_image)
    plt.imshow((diff * 255).astype(np.uint8))
    plt.show()
    # plt.savefig('diff.jpg')
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


if __name__ == "__main__":
    main()
