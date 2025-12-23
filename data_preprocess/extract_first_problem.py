import json
import argparse


def extract_first_problem_entries(input_file, output_file):
    """
    Extract first occurrence of entries with 'problem' key from JSONL file
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
    """
    seen_problems = set()
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                # Parse JSON line
                entry = json.loads(line.strip())
                
                # Check if entry has 'problem' key and it hasn't been seen before
                if 'problem' in entry and entry['problem'] not in seen_problems:
                    # Add problem to seen set
                    seen_problems.add(entry['problem'])
                    
                    entry['cot'] = entry['cot'][-200:]
                    if entry['ground_truth'] == '':
                        continue
                    # Write entry to output file
                    outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first occurrence of problems from JSONL file")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_file", help="Path to output JSONL file")
    # input_file = "partial_right_samples_cot_filter.jsonl"  # Replace with your input file path
    # output_file = "partial_right_problem.jsonl"  # Replace with your desired output file path
    input_file = parser.parse_args().input_file
    output_file = parser.parse_args().output_file   
    
    extract_first_problem_entries(input_file, output_file)
    print(f"Processing complete. First occurrence entries saved to {output_file}")
