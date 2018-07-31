import sys
import os

from model import *
from data_list import *
from utils import *

imageSize = 64
crop_size = 128
DATA_DIM = 1 * imageSize * imageSize
CLASS_DIM = 1036
BATCH_SIZE = 128
model_name = "baseline"


paddle.init(use_gpu=1, trainer_count=1)
def model_def():
    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(DATA_DIM))
    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(CLASS_DIM))

    fea, fc = resnet_baseline(image, class_dim=CLASS_DIM)
    return fea, fc, lbl


def main():
    fea, fc, lbl = model_def()
    cost = paddle.layer.classification_cost(input=fc, label=lbl)

    myReader = MyReader(imageSize=imageSize,center_crop_size=crop_size)
    trainer_reader = myReader.train_reader(train_list="./train.txt")

    reader = paddle.batch(reader=paddle.reader.shuffle(reader=trainer_reader, buf_size=2048), batch_size=BATCH_SIZE)
    print('reader done')

    # from scratch
    num_id = 0
    parameters = paddle.parameters.create(cost)

    optimizer = paddle.optimizer.Momentum(momentum=0.9, regularization=paddle.optimizer.L2Regularization(rate=0.0002 * BATCH_SIZE),
                                          learning_rate=0.1 / BATCH_SIZE,
                                          learning_rate_schedule="pass_manual",
                                          learning_rate_args="20:1, 40:0.1, 60:0.01")

    trainer = paddle.trainer.SGD(cost=cost, parameters=parameters, update_equation=optimizer)

    def event_handler(event):

        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 50 == 0:
                print "\nPass %d, Batch %d, Cost %f, Error %s" % (
                    event.pass_id + num_id, event.batch_id, event.cost, event.metrics['classification_error_evaluator'])
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

            if event.batch_id % 1000 == 0:
                save_parameters_name = model_name
                if not os.path.exists(save_parameters_name):
                    os.mkdir(save_parameters_name)

                with open(os.path.join(save_parameters_name, 'model' + '_' + str(event.pass_id + num_id)), 'w') as f:
                    trainer.save_parameter_to_tar(f)
                print("saved model")

    trainer.train(reader=reader, num_passes=100, event_handler=event_handler)


def val(start, end, modelpath=''):
    fea, fc, lbl = model_def()
    def load_image(file):
        img = cv2.imread(file, 0)
        img = np.reshape(img, [img.shape[0], img.shape[1], 1])
        img = paddle.image.center_crop(img, 128, is_color=False)
        img = cv2.resize(img, (imageSize, imageSize)).flatten()
        return img

    for i in range(start, end, 1):
        print("=======================", i, "=====================")
        modelpath = os.path.join(r'.', model_name)
        modelpath += '/model_' + str(i)
        with open(modelpath, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)


        infile = open('./test.txt')
        testdata1, testdata2 = [], []
        fea1, fea2, fea1_flip, fea2_flip, labels = [], [], [], [], []
        count = 0
        for line in infile:
            count += 1
            line = line.strip()
            line = line.split('\t')

            imgpath1 = line[0]
            imgpath2 = line[1]

            label = line[2]

            img1 = load_image(imgpath1)
            img2 = load_image(imgpath2)

            testdata1.append([img1])
            testdata2.append([img2])

            if count % 100 == 0:
                for i in paddle.infer(output_layer=fea, parameters=parameters, input=testdata1):
                    fea1.append(i)
                for i in paddle.infer(output_layer=fea, parameters=parameters, input=testdata2):
                    fea2.append(i)
                testdata1, testdata2 = [], []

            labels.append(int(label))

        caculate_roc(fea1, fea2, labels)


if __name__ == '__main__':
    main()
    val(0,100)
