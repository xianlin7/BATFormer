import torch
import torch.onnx
import onnx
from onnx_tf.backend import prepare
from onnx2keras import onnx_to_keras
import keras
import tensorflow as tf


def pth_to_onnx(input_model, onnx_path):
    '''
    1)声明：使用本函数之前，必须保证你手上已经有了.pth模型文件.
    2)功能：本函数功能四将pytorch训练得到的.pth文件转化为onnx文件。
    '''
    model = input_model  # pytorch模型加载,此处加载的模型包含图和参数
    model.eval()
    x = torch.randn(1, 1, 256, 256).to(device='cuda')
    torch.onnx.export(model, x, onnx_path, verbose=True, input_names=['input'], output_names=['output'])  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")
    '''
    torch.onnx.export(torch_model,
                      x,
                      export_onnx_file,
                      opset_version=9,  # 操作的版本，稳定操作集为9
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names=["input"],  # 输入名
                      output_names=["output"],  # 输出名
                      dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                                    "output": {0: "batch_size"}}
                      )
    '''
    # onnx_model = onnx.load('model_all.onnx')    #加载.onnx文件
    # onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))       #打印.onnx文件信息


def onnx_to_pb(output_path):
    '''
    将.onnx模型保存为.pb文件模型
    '''
    model = onnx.load(output_path)  # 加载.onnx模型文件
    tf_rep = prepare(model)
    tf_rep.export_graph('model_all.pb')  # 保存最终的.pb文件


def onnx_to_h5(output_path):
    '''
    将.onnx模型保存为.h5文件模型,并打印出模型的大致结构
    '''
    onnx_model = onnx.load(output_path)
    k_model = onnx_to_keras(onnx_model, ['input'])
    keras.models.save_model(k_model, 'kerasModel.h5', overwrite=True, include_optimizer=True)  # 第二个参数是新的.h5模型的保存地址及文件名
    # 下面内容是加载该模型，然后将该模型的结构打印出来
    model = tf.keras.models.load_model('kerasModel.h5')
    model.summary()
    print(model)

