import json

def process_json(input_json_file, output_json_file):
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output = []
    for item in data:
        instruction = item.get('prompt', '')
        output_value = item.get('completion', '')
        output.append({
            "instruction": instruction,
            "input": "",
            "output": output_value,
            "history": ""
        })

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


input_json_file = ''
input_json_file = ""
output_json_file = ''

process_json(input_json_file, output_json_file)

# Example usage
# Provide the path to the input JSON file and the output JSON file
