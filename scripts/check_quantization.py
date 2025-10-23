import torch
import torch.quantization
print(torch.__version__)
print(hasattr(torch.quantization, 'QConfigMapping'))
try:
    qconfig_mapping = torch.quantization.QConfigMapping()
    print("✅ QConfigMapping 可用")
except Exception as e:
    print(f"❌ QConfigMapping 不可用: {e}")