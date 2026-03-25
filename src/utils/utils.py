import gc
import psutil
import threading
import cv2
import torch

from diffusers.utils.torch_utils import is_compiled_module

from accelerate.utils import is_npu_available, is_xpu_available


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def get_memory_statistics(logger, precision=3):
    memory_allocated = None
    memory_reserved = None
    max_memory_allocated = None
    max_memory_reserved = None

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)

    elif torch.mps.is_available():
        memory_allocated = torch.mps.current_allocated_memory()

    else:
        logger.warning("No CUDA, MPS, or ROCm device found. Memory statistics are not available.")

    return {
        "memory_allocated": round(bytes_to_gigabytes(memory_allocated), ndigits=precision),
        "memory_reserved": round(bytes_to_gigabytes(memory_reserved), ndigits=precision),
        "max_memory_allocated": round(bytes_to_gigabytes(max_memory_allocated), ndigits=precision),
        "max_memory_reserved": round(bytes_to_gigabytes(max_memory_reserved), ndigits=precision),
    }


def bytes_to_gigabytes(x):
    if x is not None:
        return x / 1024**3


def free_memory(device):
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device):
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory_allocated=:.3f} GB")
    print(f"{max_memory_allocated=:.3f} GB")
    print(f"{max_memory_reserved=:.3f} GB")


# New Code #
# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# New Code #
# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.cuda.memory_allocated()
        elif is_xpu_available():
            torch.xpu.empty_cache()
            torch.xpu.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.xpu.memory_allocated()
        elif is_npu_available():
            torch.npu.empty_cache()
            torch.npu.reset_max_memory_allocated()  # reset the peak gauge to zero
            self.begin = torch.npu.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.end = torch.cuda.memory_allocated()
            self.peak = torch.cuda.max_memory_allocated()
        elif is_xpu_available():
            torch.xpu.empty_cache()
            self.end = torch.xpu.memory_allocated()
            self.peak = torch.xpu.max_memory_allocated()
        elif is_npu_available():
            torch.npu.empty_cache()
            self.end = torch.npu.memory_allocated()
            self.peak = torch.npu.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def resize_numpy_image_long(image, resize_long_edge=768):
    """
    Resize the input image to a specified long edge while maintaining aspect ratio.

    Args:
        image (numpy.ndarray): Input image (H x W x C or H x W).
        resize_long_edge (int): The target size for the long edge of the image. Default is 768.

    Returns:
        numpy.ndarray: Resized image with the long edge matching `resize_long_edge`, while maintaining the aspect
        ratio.
    """

    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def tensor_to_pil(self, src_img_tensor):
    """
    Converts a tensor image to a PIL image.

    This function takes an input tensor with the shape (C, H, W) and converts it
    into a PIL Image format. It ensures that the tensor is in the correct data
    type and moves it to CPU if necessary.

    Parameters:
        src_img_tensor (torch.Tensor): Input image tensor with shape (C, H, W),
            where C is the number of channels, H is the height, and W is the width.

    Returns:
        PIL.Image: The converted image in PIL format.
    """

    img = src_img_tensor.clone().detach()
    if img.dtype == torch.bfloat16:
        img = img.to(torch.float32)
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.uint8)
    pil_image = Image.fromarray(img)
    return pil_image
