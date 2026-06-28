# quantize_model.py
import torch
import torch.nn as nn
import os

# 1. Define a lightweight CNN (Simulating an edge-vision surveillance model)
class SurveillanceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Quantization requires wrapping operations that change tensor formats
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 112 * 112, 2) # Binary classification: Person vs No Person
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x) # Convert inputs to INT8
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x) # Convert outputs back to FP32 for the API
        return x

def print_size_of_model(model, label=""):
    """Helper function to print the file size of a model in MB."""
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    print(f"{label} Model Size: {size:.2f} MB")

# --- Quick Test & Execution ---
if __name__ == "__main__":
    print("🚀 Booting Edge Quantization Pipeline...")
    
    # 1. Instantiate the baseline FP32 Model
    fp32_model = SurveillanceCNN()
    fp32_model.eval() # Must be in eval mode for quantization
    
    print_size_of_model(fp32_model, "🟦 Baseline (FP32)")

    # 2. Attach the QConfig (Quantization Configuration)
    # We use fbgemm for x86 CPUs, or qnnpack for ARM (like Raspberry Pi/Mobile)
    fp32_model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    
    # 3. Prepare the model (inserts observers to track tensor statistics)
    torch.ao.quantization.prepare(fp32_model, inplace=True)
    
    print("\n📊 Calibrating model with dummy surveillance data...")
    # 4. Calibrate using representative data (crucial for finding the min/max ranges for INT8)
    dummy_calibration_data = torch.randn(10, 3, 224, 224)
    fp32_model(dummy_calibration_data)
    
    # 5. Convert the model to INT8
    int8_model = torch.ao.quantization.convert(fp32_model, inplace=False)
    
    print("\n✅ Quantization Complete.")
    print_size_of_model(int8_model, "🟩 Quantized (INT8)")
    
    # Save the compressed model for our C++ deployment tomorrow
    dummy_input = torch.randn(1, 3, 224, 224)
    traced_model = torch.jit.trace(int8_model, dummy_input)
    traced_model.save("edge_surveillance_int8.pt")
    print("\n💾 Exported optimized model to edge_surveillance_int8.pt")