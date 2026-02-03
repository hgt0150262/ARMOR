"""Test that verl_mini modules can be imported correctly."""
import sys
sys.path.insert(0, '.')

def test_imports():
    print("Testing imports...")
    
    # Test trainer import
    try:
        from verl_mini.trainer import train_qwen7b_grpo_multigpu
        print("✓ verl_mini.trainer.train_qwen7b_grpo_multigpu")
    except Exception as e:
        print(f"✗ verl_mini.trainer.train_qwen7b_grpo_multigpu: {e}")
    
    # Test models import
    try:
        from verl_mini.models import model_manager
        print("✓ verl_mini.models.model_manager")
    except Exception as e:
        print(f"✗ verl_mini.models.model_manager: {e}")
    
    # Test tools import
    try:
        from verl_mini.tools import reward_functions
        print("✓ verl_mini.tools.reward_functions")
    except Exception as e:
        print(f"✗ verl_mini.tools.reward_functions: {e}")
    
    print("\nImport tests completed!")

if __name__ == "__main__":
    test_imports()
