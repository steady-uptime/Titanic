import sys
from src.config.config_loaderv1 import config

### NOTE: This must be run in the project root folder as -> python -m scripts.verify_config

# Define required schema
REQUIRED_KEYS = {
    'env': {
        'mode': dict(type=str, exists=True),
        'compute': {
            'num_workers': dict(type=int, min=1),
            'gpu_id': dict(type=int, min=0)
        }
    },
    'data': ['raw_path', 'processed_path', 'external_path'],
    'training': ['learning_rate', 'batch_size'],
    'artifacts': ['input_file', 'output_file']
}

def main():
    print(f"{'='*40}")
    print("      CONFIGURATION VERIFICATION")
    print(f"{'='*40}")
    
    try:
        for section, keys in REQUIRED_KEYS.items():
            print(f"\nSection: [{section.upper()}]")
            # Access the nested dictionary for this section
            section_config = config.get(section, {})
            
            for key in keys:
                # Retrieve value; if it doesn't exist, it will trigger the KeyError
                value = section_config.get(key)
                if value is None:
                    raise KeyError(f"Missing key: '{key}' in section '{section}'")
                
                # Print the key and value for reference
                print(f"  > {key}: {value}")
        
        print(f"\n{'-'*40}")
        print("SUCCESS: All required configuration keys are present.")
        print(f"{'='*40}")
        sys.exit(0) 
        
    except KeyError as e:
        print(f"\nFAILURE: Configuration validation failed: {e}")
        print(f"{'='*40}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILURE: An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
