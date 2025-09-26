from abc import ABC, abstractmethod


class BaseModel(ABC):
    """统一模型接口"""

    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass

    @abstractmethod
    def translate(self, text: str, src_lang: str = "zh", tgt_lang: str = "en") -> str:
        """翻译接口"""
        pass

    @abstractmethod
    def get_model_size(self) -> str:
        """返回模型大小"""
        pass

    @abstractmethod
    def get_latency(self) -> float:
        """返回推理延迟（ms）"""
        pass