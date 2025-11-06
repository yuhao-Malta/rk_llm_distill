from rknn.api import RKNN
import numpy as np

MODEL_ONNX = 'models/tiny_transformer_student.onnx'
OUTPUT_RKNN = 'models/tiny_transformer_student_int8.rknn'
DATASET_TXT = 'dataset/calibration_samples.txt'

def main():
    # ✅ 创建 RKNN 对象
    rknn = RKNN(verbose=True)

    # ✅ 配置量化参数
    rknn.config(
        mean_values=[[0]],
        std_values=[[1]],
        quantized_dtype='asymmetric_quantized-8',
        optimization_level=3,
        target_platform='rv1126'
    )

    print('➡️ 加载 ONNX 模型...')
    ret = rknn.load_onnx(model=MODEL_ONNX)
    if ret != 0:
        print('❌ 加载 ONNX 失败')
        exit(1)

    print('➡️ 构建 RKNN 模型 (含 INT8 量化)...')
    ret = rknn.build(do_quantization=True, dataset=DATASET_TXT)
    if ret != 0:
        print('❌ 构建失败')
        exit(1)

    print('➡️ 导出 RKNN 模型...')
    ret = rknn.export_rknn(OUTPUT_RKNN)
    if ret != 0:
        print('❌ 导出失败')
        exit(1)

    print(f'✅ 模型已成功导出: {OUTPUT_RKNN}')
    rknn.release()

if __name__ == '__main__':
    main()