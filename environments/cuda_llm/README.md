# cuda_llm

### Overview
- **Environment ID**: `cuda_llm`
- **Short description**: RL environment for evaluating models on CUDA kernel generation tasks
- **Tags**: cuda, kernel-generation, compilation, bytedance, prime-intellect

### Datasets
- **Primary dataset(s)**: ByteDance CUDA LLM dataset with real CUDA kernel generation problems
- **Source links**: [ByteDance-Seed/cudaLLM-data](https://huggingface.co/datasets/ByteDance-Seed/cudaLLM-data/tree/main)
- **Split sizes**: Variable based on parquet file contents

### Task
- **Type**: single-turn
- **Parser**: CudaCodeParser (custom parser for extracting CUDA code from model responses)
- **Rubric overview**: 
  - Compilation reward (1.0x) - code compiles and runs successfully
  - Code validation reward (0.3x) - has CUDA keywords, proper structure, valid syntax
  - Format reward (0.1x) - code properly extracted from markdown blocks

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval cuda_llm
```

Configure model and sampling:

```bash
uv run vf-eval cuda_llm -m gpt-4o -n 20 -r 3 -t 1024 -T 0.7
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of compilation, validation, and format) |
| `compilation_reward` | Binary reward for successful CUDA code compilation |
| `code_validation_reward` | Reward for code structure and syntax validity |
| `format_reward` | Reward for proper code extraction and formatting |

### Testing

test the environment setup:
```bash
python test_environment.py
```

test rewards on specific outputs:
```bash
python test_rewards.py --file some_output.txt
python test_rewards.py --interactive
python test_rewards.py 'your code here'
```

