#!/usr/bin/env python3
"""
Test reward functions on existing model outputs.
Load actual model outputs and verify reward calculations.
"""

import sys
import os
import asyncio
import json
import logging
from datetime import datetime
sys.path.append('environments/cuda_kernel_generation')
from environments.cuda_kernel_generation import load_environment

# Set up logging
def setup_logging():
    """Set up logging to console only."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )

async def test_output_rewards(output_text: str, info_dict: dict = None):
    """Test reward functions on a specific model output."""
    
    # Load environment
    env = load_environment(max_concurrent_compilations=1)
    
    # Default info if not provided
    if info_dict is None:
        info_dict = {"py_code": "class Model(torch.nn.Module): pass"}
    
    logging.info(f"Testing output (first 200 chars): {output_text[:200]}...")
    logging.info("-" * 50)
    
    # Test parser
    parsed = env.parser.parse_answer(output_text)
    logging.info(f"Parser extracted: {bool(parsed)} ({'Yes' if parsed else 'No'})")
    if parsed:
        logging.info(f"  - Code length: {len(parsed)} characters")
        logging.info(f"  - Has load_inline: {'load_inline' in parsed}")
        logging.info(f"  - Has CUDA indicators: {bool(env.parser.cuda_indicators.search(parsed))}")
    
    # Create completion format
    completion = [{"role": "assistant", "content": output_text}]
    prompt = [{"role": "user", "content": "Generate CUDA code"}]
    
    # Score with rubric
    try:
        logging.info("Starting reward calculation...")
        result = await env.rubric.score_rollout(
            prompt=prompt,
            completion=completion,
            answer="",
            state={},
            info=info_dict,
            task="cuda_generation"
        )
        
        logging.info(f"Reward Scores:")
        logging.info(f"  Total Reward: {result.reward:.4f}")
        logging.info(f"  Individual Metrics:")
        for name, score in result.metrics.items():
            logging.info(f"    - {name}: {score:.4f}")
            
    except Exception as e:
        logging.error(f"Error calculating rewards: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    logging.info("=" * 60)

def load_output_from_file(file_path: str):
    """Load output from a plain text file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return ""

async def main():
    """Main function to test outputs."""
    
    # Set up logging first
    setup_logging()
    logging.info("Starting reward testing")
    
    if len(sys.argv) < 2:
        logging.info("No arguments provided - running interactive mode by default")
        logging.info("Usage:")
        logging.info("  python test_rewards.py 'your output text here'")
        logging.info("  python test_rewards.py --file output.txt")
        logging.info("  python test_rewards.py --interactive")
        logging.info("\nStarting interactive mode...")
        
        while True:
            output = input("\nEnter model output (or 'quit' to exit): ")
            if output.lower() in ['quit', 'exit', 'done']:
                break
            if output.strip():
                await test_output_rewards(output)
        return
    
    if sys.argv[1] == '--file':
        if len(sys.argv) < 3:
            logging.error("Please provide file path: --file output.txt")
            return
            
        file_path = sys.argv[2]
        logging.info(f"Loading output from file: {file_path}")
        output_text = load_output_from_file(file_path)
        if output_text:
            await test_output_rewards(output_text)
        else:
            logging.error(f"Failed to load output from {file_path}")
                
    elif sys.argv[1] == '--interactive':
        logging.info("Interactive mode: Enter outputs (type 'done' to finish)")
        while True:
            output = input("\nEnter model output: ")
            if output.lower() == 'done':
                break
            await test_output_rewards(output)
            
    else:
        # Single output from command line
        output_text = sys.argv[1]
        logging.info("Testing single output from command line")
        await test_output_rewards(output_text)
    
    logging.info("Testing complete")

if __name__ == "__main__":
    asyncio.run(main())
