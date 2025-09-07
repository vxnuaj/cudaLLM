# cudaLLM rl env

rl env. for eval'ing models ( or training if you fancy [training on the test set](https://arxiv.org/pdf/2309.08632) lmao ) to write CUDA kernels.

originally an env by bytedance, ported to prime intellect's [verifiers framework](https://github.com/willccbb/verifiers) for usage on the environments hub.

## setup

### install dependencies, setup env, etc.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init
source .venv/bin/activate
uv sync
uv pip install -e .
```

### get the data
```bash
# put your parquet file here:
# environments/eval_data/rl_cuda_llm_0424.parquet
# you can download from https://huggingface.co/datasets/ByteDance-Seed/cudaLLM-data/tree/main
```

don't forget to export your oai api key.

### test setup
```bash
uv run test_environment.py
```

### what's in here

- **cuda_llm environment** - the main RL env that loads problems from parquet files
- **reward functions** - compilation, validation, format checking 
- **test script** - debug rewards on specific outputs
- **parser** - extracts code from model responses

## how it works

1. loads real CUDA kernel problems from `prime_cuda_llm/rl_cuda_llm_0424.parquet`
2. model tries to write CUDA code using `load_inline` 
3. we parse the code, check syntax, try to compile it
4. give rewards: compilation (1.0x), validation (0.3x), format (0.1x)

## usage

### run evaluation
```bash
cd environments
vf-eval cuda_llm -m gpt-4o -n 10
```

common flags:
- `-m` model name (default: gpt-4.1-mini)
- `-n` number of problems to test (default: 5)
- `-r` rollouts per problem (default: 3)
- `-c` max concurrent requests (default: 32)
- `-t` max tokens per response (default: model default)
- `-T` temperature (default: model default)
- `-v` verbose output (default: false)
- `-b` API base URL (default: https://api.openai.com/v1)
- `-k` API key env var name (default: OPENAI_API_KEY)

### test rewards on specific individual outputs if u want.
```bash
python test_rewards.py --file some_output.txt
python test_rewards.py --interactive
python test_rewards.py 'your code here'
```

## rewards breakdown

- **compilation_reward (1.0x)** - does the CUDA code actually compile and run?
- **code_validation_reward (0.3x)** - has CUDA keywords, load_inline, reasonable length, valid syntax
- **format_reward (0.1x)** - is the code properly extracted from markdown blocks?

## files

- `environments/cuda_llm/` - main environment code
- `test_rewards.py` - debug reward calculations  
- `test_environment.py` - basic env testing
- `environments/eval_data/rl_cuda_llm_0424.parquet` - real eval data