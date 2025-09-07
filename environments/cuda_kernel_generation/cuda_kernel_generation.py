import verifiers as vf
import tempfile
import os
import subprocess
import torch
import re
import json
import asyncio
import logging
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple, List, Optional
from unittest.mock import patch
import glob
from pathlib import Path
import pandas as pd

logger = logging.getLogger("verifiers.env.cuda_kernel_generation")


TEST_CODE_TMPL = r'''
import torch
import torch.nn.functional as F
import ast
from pathlib import Path
import sys
from contextlib import contextmanager

def rewrite_cuda_model_code(src_path, dst_path):
    """Replace "op = load_inline" with "import op" to separate compilation and execution"""

    model_src = Path(src_path).read_text()
    tree = ast.parse(model_src)

    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and isinstance(call := node.value, ast.Call) and \
            ((isinstance(call.func, ast.Attribute) and call.func.attr == 'load_inline') or (isinstance(call.func, ast.Name) and call.func.id == 'load_inline')):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            ext_alias = node.targets[0].id
            for kw in call.keywords:
                if kw.arg == 'name':
                    assert isinstance(kw.value, ast.Constant)
                    ext_name = kw.value.value
                    break
            else:
                raise RuntimeError("Cannot find extension name from model_new.py")
            tree.body[i] = ast.parse(f'import {ext_name} as {ext_alias}').body[0]

    model_src = ast.unparse(tree)
    Path(dst_path).write_text(model_src)

rewrite_cuda_model_code(src_path='model_new.py', dst_path='model_new_patch.py')

from model import Model, get_inputs, get_init_inputs
from model_new_patch import ModelNew


def transform_tensors(tensors, fn):
    if not isinstance(tensors, (list, tuple)):
        return tensors
    outputs = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor = fn(tensor)
        elif isinstance(tensor, (list, tuple)):
            tensor = transform_tensors(tensor, fn)
        elif isinstance(tensor, dict):
            tensor = {k:transform_tensors(v, fn) for k, v in tensor.items()}

        outputs.append(tensor)
    return outputs


def check_equal(actual, expected):
    assert isinstance(actual, (list, tuple)) == isinstance(expected, (list, tuple))
    if not isinstance(actual, (list, tuple)):
        actual = [actual]
        expected = [expected]
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-2, rtol=1e-2)


@contextmanager
def block_torch_functional(excludes=None):
    if excludes is None:
        excludes = set()

    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith('_') and name not in excludes:
            originals[name] = attr
            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(
                    f"Function {F.__name__}.{__name} is not allowed in this context."
                )
            setattr(F, name, wrapper)

    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


init_inputs = get_init_inputs()
if not isinstance(init_inputs, (list, tuple)):
    init_inputs = [init_inputs]
torch_model = Model(*init_inputs).cuda()
cuda_model = ModelNew(*init_inputs).cuda()
cuda_model.load_state_dict(torch_model.state_dict())

torch_inputs = get_inputs()
if not isinstance(torch_inputs, (list, tuple)):
    torch_inputs = [torch_inputs]
torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())

with block_torch_functional():
    cuda_outputs = cuda_model(*cuda_inputs)
torch_outputs = torch_model(*torch_inputs)
check_equal(cuda_outputs, torch_outputs)
'''


class CudaResourceManager:
    def __init__(self, max_workers: int = 4, max_concurrent_compilations: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.compilation_semaphore = asyncio.Semaphore(max_concurrent_compilations)
        self.gpu_semaphore = asyncio.Semaphore(1)
        self.compilation_cache = {}
    
    async def compute_cuda_reward(
        self, 
        completion_text: str, 
        ground_truth_info: Dict, 
        compilation_timeout: int = 180, 
        execution_timeout: int = 60
    ) -> Dict[str, Any]:
        async with self.compilation_semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._sync_compute_score,
                completion_text,
                ground_truth_info,
                compilation_timeout,
                execution_timeout
            )
    
    def _sync_compute_score(
        self,
        solution_str: str,
        ground_truth_info: Dict,
        compilation_timeout: int,
        execution_timeout: int
    ) -> Dict[str, Any]:
        cuda_code = self._extract_cuda_code(solution_str)
        if cuda_code is None:
            return {"score": 0.0, "msg": "No CUDA code found in response", "stage": "extraction"}
        
        validate_ret, validate_msg = self._validate_cuda_code(cuda_code)
        if not validate_ret:
            return {"score": 0.0, "msg": f"Code validation failed: {validate_msg}", "stage": "validation"}
        
        compile_success, compile_result = self._compile_cuda_code(cuda_code, timeout=compilation_timeout)
        if not compile_success:
            return {"score": 0.0, "msg": compile_result["msg"], "stage": "compilation"}
        
        pytorch_module = ground_truth_info.get("py_code", ground_truth_info.get("pytorch_module", ""))
        if not pytorch_module:
            return {"score": 0.0, "msg": "No PyTorch module provided for testing", "stage": "missing_baseline"}
        
        test_success, test_msg = self._test_cuda_execution(
            compile_result["ext_filename"],
            compile_result["ext_content"],
            cuda_code,
            pytorch_module,
            timeout=execution_timeout
        )
        
        if test_success:
            return {
                "score": 1.0,
                "msg": "success",
                "stage": "execution",
                "compile_time": compile_result.get("compile_time", 0),
                "test_time": compile_result.get("test_time", 0)
            }
        else:
            return {
                "score": 0.0,
                "msg": test_msg,
                "stage": "execution_failed",
                "compile_time": compile_result.get("compile_time", 0)
            }
    
    def _extract_cuda_code(self, text: str) -> Optional[str]:
        codeblock_seps = ['python']
        languages_pattern = '|'.join(map(re.escape, codeblock_seps))
        codeblock_start = f'```({languages_pattern})'
        pattern = re.compile(codeblock_start + r'\n(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
        matches = list(pattern.finditer(text))

        if matches:
            last_match = matches[-1]
            code_content = last_match.group(2).rstrip()
            return code_content
        return None
    
    def _validate_cuda_code(self, code: str) -> Tuple[bool, str]:
        all_ops = set(torch.ops.aten.__dict__.keys())
        allowed_ops = set(['empty', 'empty_like', 'empty_strided', 'zeros', \
                           'zeros_like', 'ones', 'ones_like', 'numel', 'view',
                           'copy', 'dim', 'eye', 'full', 'full_like', 'mode',
                           'new_empty', 'new_empty_strided', 'new_full', 'new_ones', 'new_zeros',
                           'randn', 'rand'])
        forbidden_ops = all_ops - allowed_ops
        pattern = re.compile(r"(torch::|aten::|torch\.)(" + "|".join(forbidden_ops) + r")\(", flags=re.DOTALL)
        matched = re.search(pattern, code)
        if matched is not None:
            return False, f'Using {matched.group(0)[:-1]} is forbidden'
        return True, 'success'
    
    def _compile_cuda_code(self, cuda_code: str, timeout: int = 180) -> Tuple[bool, Dict]:
        ret = {
            'ext_filename': None,
            'ext_content': None,
            'msg': None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "model_new.py"), 'w') as fout:
                fout.write(cuda_code)

            compile_log = ''
            success = True
            try:
                compile_cmd = f"python3 model_new.py"
                with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "9.0", "TORCH_EXTENSIONS_DIR": "build", "MAX_JOBS": "1"}):
                    compile_result = subprocess.run(compile_cmd,
                                                    timeout=timeout,
                                                    stdout=subprocess.PIPE,
                                                    stderr=subprocess.STDOUT,
                                                    shell=True,
                                                    cwd=tmpdir)
                compile_log = compile_result.stdout.decode()
                so_files = glob.glob(f"{tmpdir}/build/**/*.so")
                assert len(so_files) == 1, f"should generate 1 .so file, got {so_files}"
                with open(so_files[0], 'rb') as fin:
                    bin_content = fin.read()
                ret['ext_filename'] = os.path.basename(so_files[0])
                ret['ext_content'] = bin_content
                ret['msg'] = "compile success"
                success = True
            except subprocess.TimeoutExpired as e:
                success = False
                ret['msg'] = "failed: compilation timed out"
            except Exception as e:
                success = False
                ret['msg'] = f"failed: compilation error: [{e}] log: [{compile_log}]"
            return success, ret
    
    def _test_cuda_execution(
        self, 
        ext_filename: str, 
        ext_content: bytes, 
        cuda_code: str, 
        pytorch_module: str,
        timeout: int = 60
    ) -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, ext_filename), 'wb') as fout:
                fout.write(ext_content)
            with open(os.path.join(tmpdir, "model_new.py"), 'w') as fout:
                fout.write(cuda_code)
            with open(os.path.join(tmpdir, "model.py"), 'w') as fout:
                fout.write(pytorch_module)
            with open(os.path.join(tmpdir, "test.py"), 'w') as fout:
                fout.write(TEST_CODE_TMPL)

            test_log = ''
            try:
                test_cmd = f"python3 test.py"
                test_result = subprocess.run(test_cmd,
                                             timeout=timeout,
                                             stderr=subprocess.STDOUT,
                                             stdout=subprocess.PIPE,
                                             shell=True,
                                             cwd=tmpdir)
                test_log = test_result.stdout.decode()
            except subprocess.TimeoutExpired as e:
                return False, "failed: test timed out"
            except Exception as e:
                return False, f"failed: test error: [{e}] log: [{test_log}]"
            if test_result.returncode != 0:
                return False, f"failed: test error: [{test_log}]"

        return True, "test success"


class CudaCodeParser(vf.Parser):
    def __init__(self, supported_languages=None, strict_extraction=False, validate_syntax=True):
        self.supported_languages = supported_languages or ['python', 'cpp', 'cuda']
        self.strict_extraction = strict_extraction
        self.validate_syntax = validate_syntax
        self._compile_patterns()
    
    def _compile_patterns(self):
        languages_pattern = '|'.join(map(re.escape, self.supported_languages))
        self.codeblock_pattern = re.compile(
            f'```({languages_pattern})\\n(.*?)(?:\\n```)?(?=\\n```|$)', 
            re.DOTALL
        )
        cuda_keywords = ['load_inline', '__global__', '__device__', '__shared__', 'blockIdx', 'threadIdx', 'gridDim', 'blockDim']
        self.cuda_indicators = re.compile('|'.join(cuda_keywords), re.IGNORECASE)
    
    def parse_answer(self, completion) -> Optional[str]:
        if completion is None:
            return None
        
        if isinstance(completion, list):
            if not completion:
                return None
            text_content = completion[-1].get('content', str(completion[-1]))
        else:
            text_content = str(completion)
        
        return self._extract_cuda_code(text_content)
    
    def _extract_cuda_code(self, text: str) -> Optional[str]:
        matches = list(self.codeblock_pattern.finditer(text))
        if not matches:
            return None
        
        cuda_matches = []
        for match in matches:
            code_content = match.group(2).rstrip()
            if self.cuda_indicators.search(code_content):
                cuda_matches.append(match)
        
        selected_match = cuda_matches[-1] if cuda_matches else matches[-1]
        code = selected_match.group(2).rstrip()
        
        if self.validate_syntax and not self._validate_python_syntax(code):
            return None
        
        return code
    
    def _validate_python_syntax(self, code: str) -> bool:
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def get_format_reward_func(self):
        def format_reward(parser, completion, answer, state):
            code = parser.parse_answer(completion)
            if not code:
                return 0.0
            
            reward_components = []
            
            try:
                compile(code, '<string>', 'exec')
                reward_components.append(0.4)
            except SyntaxError:
                reward_components.append(0.0)
            
            
            if self.cuda_indicators.search(code):
                reward_components.append(0.3)
            else:
                reward_components.append(0.0)
            
            structure_indicators = [
                'def ' in code,
                'import ' in code or 'from ' in code,
                len(code.split('\n')) >= 5
            ]
            structure_score = 0.2 * sum(structure_indicators) / len(structure_indicators)
            reward_components.append(structure_score)
            
            code_length = len(code.strip())
            if 50 <= code_length <= 5000:
                reward_components.append(0.1)
            else:
                reward_components.append(0.0)
            
            return sum(reward_components)
        
        return format_reward
    
    def extract_metadata(self, completion) -> Dict[str, Any]:
        code = self.parse_answer(completion)
        if not code:
            return {}
        
        return {
            'code_length': len(code),
            'line_count': len(code.split('\n')),
            'has_cuda_keywords': bool(self.cuda_indicators.search(code)),
            'has_imports': 'import ' in code or 'from ' in code,
            'has_functions': 'def ' in code,
            'has_classes': 'class ' in code
        }


def load_default_cuda_dataset():
    # Use path relative to this module's directory
    module_dir = Path(__file__).parent.parent
    dataset_path = str(module_dir / "eval_data" / "rl_cuda_llm_0424.parquet")
    if os.path.exists(dataset_path):
        try:
            logger.info(f"Loading dataset from {dataset_path}")
            df = pd.read_parquet(dataset_path)
            converted_data = []
            sample_df = df.head(10) if len(df) > 10 else df
            
            for _, row in sample_df.iterrows():
                prompt_text = row.get("prompt", "")
                if isinstance(prompt_text, (list, tuple)) and len(prompt_text) > 0:
                    
                    prompt_content = prompt_text[0].get('content', '') if isinstance(prompt_text[0], dict) else str(prompt_text[0])
                else:
                    prompt_content = str(prompt_text)
                
                if len(prompt_content.strip()) > 10:
                    converted_row = {
                        "prompt": [{"role": "user", "content": prompt_content.strip()}],
                        "info": {
                            "py_code": row.get("py_code", ""),
                            "pytorch_module": row.get("py_code", ""),
                            "level": row.get("level", "unknown"),
                            "type": row.get("type", "unknown"),
                            "ops": row.get("ops", ""),
                            "ability": row.get("ability", ""),
                            "data_source": row.get("data_source", "")
                        }
                    }
                    if row.get("answer"):
                        converted_row["answer"] = row["answer"]
                    converted_data.append(converted_row)
            
            from datasets import Dataset
            return Dataset.from_list(converted_data)
        except Exception as e:
            logger.warning(f"Failed to load parquet dataset: {e}")
            
            return create_mock_dataset()
    else:
        
        return create_mock_dataset()

def create_mock_dataset(mock:bool = False):
    """Create a small mock dataset using a single sample from parquet if available"""
    from datasets import Dataset
    
    # Try to get one sample from the parquet file first
    module_dir = Path(__file__).parent.parent
    parquet_path = str(module_dir / "eval_data" / "rl_cuda_llm_0424.parquet")
    if os.path.exists(parquet_path):
        try:
            full_dataset = load_dataset("parquet", data_files=parquet_path, split='train')
           
            print(f"Full dataset size: {len(full_dataset)}") 
            
            if len(full_dataset) > 0:
                # Use first sample from parquet
                sample = full_dataset[0]
                logger.info(f"Using single sample from parquet file as mock data")
                return Dataset.from_list([sample])
        except Exception as e:
            logger.warning(f"Failed to load sample from parquet: {e}, using hardcoded mock")
   
    logger.warning("Using hardcoded data")
    mock_data = [
        {
            "prompt": [{"role": "user", "content": "Write a CUDA kernel to add two vectors"}],
            "info": {
                "py_code": '''import torch
class Model(torch.nn.Module):
    def forward(self, a, b):
        return a + b

def get_inputs():
    return [torch.randn(1024).cuda(), torch.randn(1024).cuda()]

def get_init_inputs():
    return []''',
                "pytorch_module": '''import torch
class Model(torch.nn.Module):
    def forward(self, a, b):
        return a + b

def get_inputs():
    return [torch.randn(1024).cuda(), torch.randn(1024).cuda()]

def get_init_inputs():
    return []''',
                "level": "easy",
                "type": "vector_add"
            }
        }
    ]
    return Dataset.from_list(mock_data)


def load_environment(
    dataset_path: Optional[str] = None,
    max_concurrent_compilations: int = 2,
    compilation_timeout: int = 180,
    execution_timeout: int = 60,
    enable_validation: bool = True,
    **kwargs
) -> vf.SingleTurnEnv:
    
    if dataset_path:
        try:
            dataset = load_dataset("parquet", data_files=dataset_path, split='train')
            logger.info(f"Loaded dataset from {dataset_path} with {len(dataset)} examples")
        except Exception as e:
            logger.error(f"Failed to load dataset from {dataset_path}: {e}")
            dataset = load_default_cuda_dataset()
    else:
        module_dir = Path(__file__).parent.parent
        parquet_path = str(module_dir / "eval_data" / "rl_cuda_llm_0424.parquet")
        if os.path.exists(parquet_path):
            try:
                dataset = load_dataset("parquet", data_files=parquet_path, split='train')
                logger.info(f"Using parquet dataset from {parquet_path} with {len(dataset)} examples")
            except Exception as e:
                logger.error(f"Failed to load parquet dataset: {e}, falling back to mock data")
                dataset = load_default_cuda_dataset()
        else:
            logger.info("Parquet file not found, using mock dataset")
            dataset = load_default_cuda_dataset()
    
    
    resource_manager = CudaResourceManager(
        max_concurrent_compilations=max_concurrent_compilations
    )
    
    
    parser = CudaCodeParser(
        supported_languages=['python', 'cpp', 'cuda'],
        validate_syntax=True
    )
    
    
    async def cuda_compilation_reward(parser, completion, info, state):
        completion_text = parser.parse_answer(completion) or str(completion)
        result = await resource_manager.compute_cuda_reward(
            completion_text, info, compilation_timeout, execution_timeout
        )
        return result.get('score', 0.0)
    
    def code_validation_reward(parser, completion, info, state):
        if not enable_validation:
            return 0.5  
        
        code = parser.parse_answer(completion)
        if not code:
            return 0.0
        
        
        try:
            compile(code, '<string>', 'exec')
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False
        
        has_cuda = parser.cuda_indicators.search(code) is not None
        has_load_inline = 'load_inline' in code
        reasonable_length = 50 <= len(code.strip()) <= 10000
        
        score = 0.0
        if syntax_ok:
            score += 0.4
        if has_cuda:
            score += 0.3
        if has_load_inline:
            score += 0.2
        if reasonable_length:
            score += 0.1
        
        return score
    
    rubric = vf.Rubric(
        funcs=[cuda_compilation_reward, code_validation_reward, parser.get_format_reward_func()],
        weights=[1.0, 0.3, 0.1],
        parser=parser
    )
    
    env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt="",
        parser=parser,
        rubric=rubric,
        **kwargs
    )
    
    env.resource_manager = resource_manager
    env.config = {
        'compilation_timeout': compilation_timeout,
        'execution_timeout': execution_timeout,
        'enable_validation': enable_validation,
        'max_concurrent_compilations': max_concurrent_compilations
    }
    
    return env
