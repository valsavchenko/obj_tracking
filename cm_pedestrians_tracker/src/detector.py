import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

import _ped_trk.timer


class _Logger(trt.ILogger):
    """
    """

    def __init__(self, logger):
        super().__init__()
        self.__logger = logger

    def log(self, severity, msg):
        method_per_severity = {
            trt.ILogger.VERBOSE: self.__logger.debug,
            trt.ILogger.INFO: self.__logger.info,
            trt.ILogger.WARNING: self.__logger.warning,
            trt.ILogger.ERROR: self.__logger.error,
            trt.ILogger.INTERNAL_ERROR: self.__logger.critical
        }

        method_per_severity[severity](f'{msg}')


class Detector(_ped_trk.timer.Timeable):
    """
    """

    def __init__(self, logger, settings):
        """
        Sets up a model to infer with
        """
        super().__init__(logger=logger)

        log = _Logger(logger=logger)
        trt.init_libnvinfer_plugins(logger=log, namespace='')

        with open(settings['weights_file_path'], "rb") as weights_file, trt.Runtime(logger=log) as runtime:
            weights = weights_file.read()
            self.__engine = runtime.deserialize_cuda_engine(weights)

        if self.__engine is None:
            raise ValueError(f'Fail to deserialize CUDA engine from {settings["weights_file_path"]}')

        self.__context = self.__engine.create_execution_context()

        self.__inputs, self.__outputs, self.__bindings = [], [], []

        for binding in self.__engine:
            size = trt.volume(self.__engine.get_tensor_shape(name=binding))
            dtype = trt.nptype(self.__engine.get_tensor_dtype(name=binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.__bindings.append(int(device_mem))

            puts = self.__inputs if trt.TensorIOMode.INPUT == self.__engine.get_tensor_mode(binding) else self.__outputs
            puts.append({'host': host_mem, 'device': device_mem})

        self.__stream = cuda.Stream()

    @_ped_trk.timer.timer(label='obj_det')
    def detect(self, frame):
        """
        """
        height, width, *_ = frame.shape

        boxes = []
        num = np.random.randint(low=0, high=5)
        ls = np.random.randint(low=0, high=width, size=num)
        ts = np.random.randint(low=0, high=height, size=num)
        for l, t in zip(ls, ts):
            if (width - l) < 2 or (height - t) < 2:
                continue

            w = np.random.randint(low=1, high=(width - l))
            h = np.random.randint(low=1, high=(height - t))
            boxes.append((l, t, w, h))
        scores = np.random.rand(len(boxes))

        class_objs = [{'lt_wh': b, 'score': s} for b, s in zip(boxes, scores)]
        return class_objs
