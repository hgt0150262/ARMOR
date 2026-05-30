"""Test that ARMOR modules can be imported correctly."""
import sys
sys.path.insert(0, '.')

def test_imports():
    print("Testing imports...")
    
    # Test trainer import
    try:
        from ARMOR.trainer import train_qwen7b_grpo_multigpu
        print("✓ ARMOR.trainer.train_qwen7b_grpo_multigpu")
    except Exception as e:
        print(f"✗ ARMOR.trainer.train_qwen7b_grpo_multigpu: {e}")
    
    # Test models import
    try:
        from ARMOR.models import model_manager
        print("✓ ARMOR.models.model_manager")
    except Exception as e:
        print(f"✗ ARMOR.models.model_manager: {e}")
    
    # Test tools import
    try:
        from ARMOR.tools import reward_functions
        print("✓ ARMOR.tools.reward_functions")
    except Exception as e:
        print(f"✗ ARMOR.tools.reward_functions: {e}")
    
    print("\nImport tests completed!")

if __name__ == "__main__":
    test_imports()
