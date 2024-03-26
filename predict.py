import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import onnxruntime as rt
import numpy as np
import time


def main():

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "./img/test.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0).numpy()

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    t1 = time.time()

    # onnx模型前向推理
    sess = rt.InferenceSession('./model/swin-transformer-sim.onnx')
    # 模型的输入和输出节点名，可以通过netron查看
    input_name = 'input.1'
    outputs_name = ['6347']
    # 模型推理:模型输出节点名，模型输入节点名，输入数据(注意节点名的格式！！！！！)
    net_outs = sess.run(outputs_name, {input_name: img})
    t2 = time.time()
    print("inf_time:%.3f"%(t2 -t1))
    net_outs = np.array(net_outs).reshape(5).astype(np.float32)
    print(net_outs)
    net_outs = torch.from_numpy(net_outs)
    predict = torch.softmax(net_outs, dim=0)
    predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    print('Running Time: {}'.format(t_end - t_start))
