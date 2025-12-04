import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

def parse_apple_health_export(export_xml_path):
    """
    Parse the Apple Health export.xml file and convert <Record> items into a DataFrame.
    """

    print(f"Parsing: {export_xml_path} ...")
    
    tree = ET.parse(export_xml_path)
    root = tree.getroot()

    records = []

    # Apple records are under <HealthData><Record ... />
    for record in tqdm(root.findall("Record"), desc="Processing Records"):
        record_type = record.attrib.get("type")
        start_date = record.attrib.get("startDate")
        end_date = record.attrib.get("endDate")
        value = record.attrib.get("value")
        unit = record.attrib.get("unit")

        records.append({
            "type": record_type,
            "start_date": start_date,
            "end_date": end_date,
            "value": value,
            "unit": unit
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    # Path to your Apple Health export.xml file
    EXPORT_FILE = "export.xml"  # modify if needed

    df = parse_apple_health_export(EXPORT_FILE)

    print("\nPreview of parsed data:")
    print(df.head())

    # Save output
    output_file = "apple_health_records.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved CSV → {output_file}")
