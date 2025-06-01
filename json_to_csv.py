import csv
import json

if __name__ == "__main__":
    # Define the input and output file paths
    input_json_path = './prompt/video_prompts.json'
    output_csv_path = './data/metadata.csv'

    # Read the JSON file
    with open(input_json_path, 'r') as json_file:
        data = json.load(json_file)

    with open(output_csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header
        writer.writerow(['cam_video', 'lidar_video', 'prompt'])
        
        # Write the data
        for name, prompt in data.items():
            writer.writerow(['cam_video/'+name, 'lidar_video/'+name, prompt.replace("\n", " ").replace("  ", " ")])
    