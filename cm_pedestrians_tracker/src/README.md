# Blogs

- https://github.com/OpenJetson/tensorrt-yolov5/blob/main/yolov5_trt.py
- https://github.com/Linaom1214/TensorRT-For-YOLO-Series

# Models
- https://docs.ultralytics.com/tasks/detect/#models
- https://pytorch.org/get-started/locally/#windows-pip

# Ultralytics

```ps
C:\Users\valen\dev\_venv>py -m venv ultralytics
C:\Users\valen\dev> .\ultralytics\Scripts\activate
(ultralytics) C:\Users\valen\dev\_venv>py -m pip install --upgrade pip
(ultralytics) C:\Users\valen\dev\_venv>pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
(ultralytics) C:\Users\valen\dev\_venv>pip install ultralytics
(ultralytics) C:\Users\valen\dev\ultralytics>py test.py
```

# TensorRT

```ps
C:\Program Files\NVIDIA\TensorRT-10.7.0.23\bin>trtexec.exe --onnx=C:\Users\valen\dev\ultralytics\yolo11x.onnx --saveEngine=C:\Users\valen\dev\ultralytics\yolo11x.engine
```

# TensorRT for YOLO series

```ps
C:\Users\valen\dev\_venv>py -m venv tensorrt-for-yolo-series
C:\Users\valen\dev\_venv>.\tensorrt-for-yolo-series\Scripts\activate
(tensorrt-for-yolo-series) C:\Users\valen\dev\_venv>py -m pip install --upgrade pip
(tensorrt-for-yolo-series) C:\Users\valen\dev\_venv>py -m pip install "C:\Program Files\NVIDIA\TensorRT-10.7.0.23\python\tensorrt-10.7.0-cp312-none-win_amd64.whl"
(tensorrt-for-yolo-series) C:\Users\valen\dev\TensorRT-For-YOLO-Series>pip install cuda-python opencv-python matplotlib

(tensorrt-for-yolo-series) PS C:\Users\valen\dev\TensorRT-For-YOLO-Series> py .\export.py -o C:\Users\valen\dev\ultralytics\yolo11x.onnx -e yolo11x.trt --end2end --v8 -p fp32
(tensorrt-for-yolo-series) PS C:\Users\valen\dev\TensorRT-For-YOLO-Series> py .\trt.py -e .\yolo11x.trt -i C:\Users\valen\dev\ultralytics\bus.jpg -o bus_11.jpg --end2end

(tensorrt-for-yolo-series) PS C:\Users\valen\dev\TensorRT-For-YOLO-Series> py .\export.py -o C:\Users\valen\dev\ultralytics\yolov8x.engine -e yolov8x.trt --end2end --v8 -p fp32
(tensorrt-for-yolo-series) PS C:\Users\valen\dev\TensorRT-For-YOLO-Series> py .\trt.py -e .\yolov8x.trt -i C:\Users\valen\dev\ultralytics\bus.jpg -o bus_8.jpg --end2end
```
