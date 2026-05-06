import csv
import os

def filter_csv_by_whitelist(source_csv, whitelist_csv, output_csv):
    """
    Filter source dataset utilizing a whitelist of allowed filenames
    """
    
    # Load whitelisted base filenames
    allowed_names = set()
    try:
        with open(whitelist_csv, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    # Extract base filename excluding extensions and whitespace
                    name = os.path.splitext(row[0].strip())[0]
                    allowed_names.add(name)
    except FileNotFoundError:
        print(f"Error missing whitelist file {whitelist_csv}")
        return

    print(f"Successfully loaded whitelist containing {len(allowed_names)} base filenames.")

    # Read source CSV and apply filtering logic
    filtered_rows = []
    header = None
    
    try:
        with open(source_csv, mode='r', encoding='utf-8-sig') as f:
            # Use DictReader to accurately locate FileName property
            reader = csv.DictReader(f)
            header = reader.fieldnames
            
            for row in reader:
                # Retrieve FileName attribute value
                original_filename = row.get('FileName', '')
                # Remove extension for matching purposes
                pure_name = os.path.splitext(original_filename.strip())[0]
                
                if pure_name in allowed_names:
                    filtered_rows.append(row)
    except FileNotFoundError:
        print(f"Error missing source file {source_csv}")
        return
    except KeyError:
        print("Error missing FileName column in source CSV")
        return

    # Write filtered results to new file
    with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Processing complete")
    print(f"Original row count: {reader.line_num - 1}")
    print(f"Retained row count: {len(filtered_rows)}")
    print(f"Results saved to {output_csv}")

# Configuration parameters
source_file = '/home/wangchenhao/Github/DGFMT/csv_files/MSP_Podcast.csv'
whitelist = 'filenames.csv'
output = 'MSP_Podcast_Filtered.csv'

if __name__ == "__main__":
    filter_csv_by_whitelist(source_file, whitelist, output)