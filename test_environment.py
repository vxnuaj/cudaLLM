#!/usr/bin/env python3

"""
Test script for the CUDA kernel generation environment.
This script tests the basic functionality without requiring actual CUDA compilation; being useful for testing init. setup.
"""

import sys
import traceback
import os
sys.path.append('environments/cuda_kernel_generation')

from cuda_kernel_generation import load_environment, CudaCodeParser, CudaResourceManager

def test_environment_loading():
    """Test basic environment loading"""
    print("\n--- Testing Environment Loading ---")
    try:
        env = load_environment(max_concurrent_compilations=1)
        print(f"Environment loaded successfully")
        print(f"Dataset size: {len(env.dataset)}")
        print(f"System prompt length: {len(env.system_prompt)}")
        print(f"Environment has resource_manager: {hasattr(env, 'resource_manager')}")
        print(f"Environment has config: {hasattr(env, 'config')}")
        return env
    except Exception as e:
        print(f"Failed to load environment: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def test_parser():
    """Test the CUDA code parser"""
    print("\n--- Testing Parser ---")
    parser = CudaCodeParser()
    
    cuda_response = '''Here's the optimized CUDA implementation:

```python
import torch
from torch.utils.cpp_extension import load_inline

source = """
__global__ void my_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // CUDA kernel code here
}
"""

op = load_inline(name="test", cpp_sources="", cuda_sources=source)
```
'''
    
    parsed = parser.parse_answer(cuda_response)
    if parsed:
        print("Successfully parsed CUDA code from response")
        print(f"Parsed code length: {len(parsed)}")
        print(f"Contains load_inline: {'load_inline' in parsed}")
    else:
        print("Failed to parse CUDA code")
    
    # Test format reward
    format_reward_func = parser.get_format_reward_func()
    reward = format_reward_func(parser, cuda_response, "", {})
    print(f"Format reward computed: {reward}")
    
    return parser

def test_resource_manager():
    """Test the resource manager (without actual compilation)"""
    print("\n--- Testing Resource Manager ---")
    manager = CudaResourceManager(max_concurrent_compilations=1)
    
    test_code = '''
```python
import torch
from torch.utils.cpp_extension import load_inline

class ModelNew(torch.nn.Module):
    def forward(self, x):
        return x * 2
```
'''
    
    extracted = manager._extract_cuda_code(test_code)
    if extracted:
        print("Code extraction working")
        print(f"Extracted {len(extracted)} characters")
    else:
        print("Code extraction failed")
    
    valid_code = "import torch\nclass Model: pass"
    is_valid, msg = manager._validate_cuda_code(valid_code)
    print(f"Code validation working: {is_valid}, {msg}")
    
    return manager

def test_dataset_sample():
    """Test dataset sample processing"""
    print("\n--- Testing Dataset Sample ---")
    try:
        env = load_environment(max_concurrent_compilations=1)
        if len(env.dataset) > 0:
            sample = env.dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Prompt preview: {sample['prompt'][:100]}...")
            if 'info' in sample:
                print(f"Info keys: {sample['info'].keys()}")
                print(f"Has PyTorch module: {bool(sample['info'].get('py_code'))}")
        else:
            print("Dataset is empty - this is expected if no parquet file is available")
        return True
    except Exception as e:
        print(f"Dataset sampling failed: {e}")
        print(f"Tracebakc: {traceback.format_exc()}")
        return False

def main():
    """Run all tests"""
    print("Testing CUDA Kernel Generation Environment")
    print("=" * 50)
    
    # Test individual components
    env = test_environment_loading()
    parser = test_parser()
    manager = test_resource_manager()
    test_dataset_sample()
    
    print("\n--- Summary ---")
    if env and parser and manager:
        print("All basic tests passed!")
    else:
        print("Some tests failed - check the output above")
        sys.exit(1)

if __name__ == "__main__":
    main()