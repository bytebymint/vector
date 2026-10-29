"""Measured host telemetry; unavailable GPU data stays null."""
import platform
import time
import psutil
import torch


class Performance:
    def __init__(self):
        self.process=psutil.Process()
        self.process.cpu_percent(None)
        psutil.cpu_percent(None)
        self.last_sample=0
        self.cached={}
        self.cuda=torch.cuda.is_available()
        self.device=torch.cuda.get_device_name(0) if self.cuda else platform.processor()
        self.logical_cpus=psutil.cpu_count()
        self.physical_cpus=psutil.cpu_count(logical=False)
        self.nvml=None
        self.nvml_handle=None
        if self.cuda:
            try:
                import pynvml
                pynvml.nvmlInit();self.nvml=pynvml
                self.nvml_handle=pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:pass
    def sample(self):
        if time.monotonic()-self.last_sample<1:return self.cached
        self.last_sample=time.monotonic()
        gpu_util=None;gpu_mem=None
        if self.nvml:
            try:
                gpu_util=self.nvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu
                gpu_mem=self.nvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used/1048576
            except Exception:pass
        self.cached=dict(cpu_percent=psutil.cpu_percent(None),process_cpu_percent=self.process.cpu_percent(None),
            memory_mb=self.process.memory_info().rss/1048576,gpu_percent=gpu_util,gpu_memory_mb=gpu_mem,
            cuda=self.cuda,device=self.device,logical_cpus=self.logical_cpus,physical_cpus=self.physical_cpus)
        return self.cached
