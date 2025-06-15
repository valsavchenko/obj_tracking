import cv2
import numpy as np
import pycuda.autoinit  # noqa
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

        inputs, outputs = [], []
        for binding in self.__engine:
            shape = trt.volume(self.__engine.get_tensor_shape(name=binding))
            dtype = trt.nptype(self.__engine.get_tensor_dtype(name=binding))
            host_memory = cuda.pagelocked_empty(shape=shape, dtype=dtype)
            device_memory = cuda.mem_alloc(host_memory.nbytes)

            puts = inputs if trt.TensorIOMode.INPUT == self.__engine.get_tensor_mode(binding) else outputs
            puts.append({'name': binding, 'host': host_memory, 'device': device_memory, 'shape': shape})

        if 1 != len(inputs):
            raise ValueError(f'Fail to parse CUDA engine with {len(inputs)} inputs and {len(outputs)} outputs')

        self.__input, *_ = inputs
        self.__outputs = outputs
        self.__stream = cuda.Stream()

    def _pre_process(self, frame):
        """
        """
        # Resize the frame, while maintaining aspect ratio, to match the input tensor
        tensor_name = self.__input['name']
        tensor_height, tensor_width = self.__engine.get_tensor_shape(tensor_name)[2:]
        frame_height, frame_width = frame.shape[:2]
        scale = min(tensor_width / frame_width, tensor_height / frame_height)
        resized_width, resized_height = int(scale * frame_width), int(scale * frame_height)
        resized = cv2.resize(src=frame, dsize=(resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        # Pad the input tensor
        pad_top = (tensor_height - resized_height) // 2
        pad_bottom = tensor_height - resized_height - pad_top
        pad_left = (tensor_width - resized_width) // 2
        pad_right = tensor_width - resized_width - pad_left
        padded = cv2.copyMakeBorder(src=resized, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))

        rgbed = cv2.cvtColor(src=padded, code=cv2.COLOR_BGR2RGB)

        normalized = rgbed.astype(dtype=np.float32) / 255.0

        transposed = np.transpose(normalized, axes=(2, 0, 1))

        batched = np.expand_dims(transposed, axis=0)

        tensor = np.ascontiguousarray(batched, dtype=np.float32)

        return tensor, scale, (pad_top, pad_left)

    def _post_process(self, scale, pad):
        """
        """
        class_objs = []

        pad_top, pad_left = pad
        for d in range(self.__outputs[0]['host'][0]):
            left, top, right, bottom = self.__outputs[1]['host'][4 * d:4 * (d + 1)]
            left = int((left - pad_left) / scale)
            top = int((top - pad_top) / scale)
            right = int((right - pad_left) / scale)
            bottom = int((bottom - pad_top) / scale)
            lt_wh = left, top, right - left, bottom - top

            score = self.__outputs[2]['host'][d]

            class_objs.append({'lt_wh': lt_wh, 'score': score})

        return class_objs


    @_ped_trk.timer.timer(label='obj_det')
    def detect(self, frame):
        """
        """
        input_tensor, scale, pad = self._pre_process(frame=frame)

        np.copyto(dst=self.__input['host'], src=input_tensor.ravel())
        cuda.memcpy_htod_async(dest=self.__input['device'], src=self.__input['host'], stream=self.__stream)
        for put in [self.__input] + self.__outputs:
            self.__context.set_tensor_address(put['name'], int(put['device']))
        self.__context.execute_async_v3(stream_handle=self.__stream.handle)

        for output in self.__outputs:
            cuda.memcpy_dtoh_async(dest=output['host'], src=output['device'], stream=self.__stream)
        self.__stream.synchronize()

        class_objs = self._post_process(scale=scale, pad=pad)
        return class_objs
