from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import logging

# Get DAG directory
dag_path = os.path.dirname(__file__)

def extract_data():
    """Extract data from CSV with flexible parsing"""
    try:
        # File path
        csv_file = os.path.join(dag_path, 'Expanded100Data_Dataset.csv')
        logging.info(f"Reading CSV from: {csv_file}")

        # Check if file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        # Try different delimiters
        delimiters = [',', ';', '\t']
        df = None

        for delimiter in delimiters:
            try:
                df = pd.read_csv(csv_file, sep=delimiter, encoding='utf-8')
                if df.shape[1] > 1:  # Successfully parsed multiple columns
                    logging.info(f"Successfully parsed with delimiter: '{delimiter}'")
                    break
            except Exception as e:
                logging.warning(f"Failed with delimiter '{delimiter}': {e}")
                continue

        if df is None or df.shape[1] == 1:
            # Try without specifying delimiter (let pandas auto-detect)
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                logging.info("Used pandas auto-detection")
            except:
                df = pd.read_csv(csv_file, encoding='latin-1')
                logging.info("Used latin-1 encoding")

        # Clean column names
        df.columns = df.columns.str.strip()

        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Columns: {list(df.columns)}")

        # Save extracted data
        output_path = '/tmp/extracted_data.csv'
        df.to_csv(output_path, index=False)

        print(f"✓ Successfully extracted {len(df)} records")
        return output_path

    except Exception as e:
        logging.error(f"Extract failed: {str(e)}")
        raise

def transform_data():
    """Transform data to star schema"""
    try:
        # Read extracted data
        df = pd.read_csv('/tmp/extracted_data.csv')
        logging.info(f"Transform input shape: {df.shape}")

        # Dimension Tables creation
        dimensions = {}

        # Time dimension (if Year/Quarter exist)
        time_cols = ['Year', 'Quarter']
        time_dim = df[time_cols].drop_duplicates().reset_index(drop=True)
        time_dim['time_id'] = range(1, len(time_dim) + 1)
        dimensions['time'] = time_dim
        logging.info(f"Created time dimension with {len(time_dim)} records")

        # Product dimension (Product Model, 5G Capability)
        product_cols = ['Product Model', '5G Capability']
        product_dim = df[product_cols].drop_duplicates().reset_index(drop=True)
        product_dim['product_id'] = range(1, len(product_dim) + 1)
        dimensions['product'] = product_dim
        logging.info(f"Created product dimension with {len(product_dim)} records")

        # Region dimension (Region, 5G Coverage, Subscribers, etc.)
        region_cols = ['Region', 'Regional 5G Coverage (%)', '5G Subscribers (millions)', 'Avg 5G Speed (Mbps)', 'Preference for 5G (%)']
        region_dim = df[region_cols].drop_duplicates().reset_index(drop=True)

        # Ensure unique values in Region
        region_dim = region_dim.drop_duplicates(subset='Region').reset_index(drop=True)

        region_dim['region_id'] = range(1, len(region_dim) + 1)
        dimensions['region'] = region_dim
        logging.info(f"Created region dimension with {len(region_dim)} records")

        # Fact Tables creation (with product_id, region_id, time_id)
        # Sales_Fact_Model
        fact_table_model = df[['Product Model', 'Year', 'Quarter', 'Units Sold', 'Revenue ($)']].copy()
        fact_table_model['fact_id'] = range(1, len(fact_table_model) + 1)
        # Map Product Model to product_id
        fact_table_model['product_id'] = fact_table_model['Product Model'].map(product_dim.set_index('Product Model')['product_id'])
        fact_table_model['time_id'] = fact_table_model[['Year', 'Quarter']].apply(lambda row: time_dim[(time_dim['Year'] == row['Year']) & (time_dim['Quarter'] == row['Quarter'])]['time_id'].values[0], axis=1)

        # Sales_Fact_Region
        fact_table_region = df[['Region', 'Year', 'Quarter', 'Units Sold', 'Revenue ($)', 'Market Share (%)']].copy()
        fact_table_region['fact_id'] = range(1, len(fact_table_region) + 1)
        # Map Region to region_id
        fact_table_region['region_id'] = fact_table_region['Region'].map(region_dim.set_index('Region')['region_id'])
        fact_table_region['time_id'] = fact_table_region[['Year', 'Quarter']].apply(lambda row: time_dim[(time_dim['Year'] == row['Year']) & (time_dim['Quarter'] == row['Quarter'])]['time_id'].values[0], axis=1)

        # Sales_Fact_5G
        fact_table_5g = df[['Product Model', 'Region', 'Year', 'Quarter', 'Units Sold', 'Revenue ($)', '5G Subscribers (millions)', 'Avg 5G Speed (Mbps)', 'Preference for 5G (%)']].copy()
        fact_table_5g['fact_id'] = range(1, len(fact_table_5g) + 1)
        # Map Product Model and Region to their respective IDs
        fact_table_5g['product_id'] = fact_table_5g['Product Model'].map(product_dim.set_index('Product Model')['product_id'])
        fact_table_5g['region_id'] = fact_table_5g['Region'].map(region_dim.set_index('Region')['region_id'])
        fact_table_5g['time_id'] = fact_table_5g[['Year', 'Quarter']].apply(lambda row: time_dim[(time_dim['Year'] == row['Year']) & (time_dim['Quarter'] == row['Quarter'])]['time_id'].values[0], axis=1)

        # Save all tables
        fact_table_model.to_csv('/tmp/fact_table_model.csv', index=False)
        fact_table_region.to_csv('/tmp/fact_table_region.csv', index=False)
        fact_table_5g.to_csv('/tmp/fact_table_5g.csv', index=False)

        for dim_name, dim_data in dimensions.items():
            dim_data.to_csv(f'/tmp/dim_{dim_name}.csv', index=False)

        print(f"✓ Created {len(dimensions)} dimensions and 3 fact tables")

    except Exception as e:
        logging.error(f"Transform failed: {str(e)}")
        raise

def load_data():
    """Load data to output directory"""
    try:
        # Create output directory
        output_dir = os.path.join(dag_path, 'star_schema_output_yok')
        os.makedirs(output_dir, exist_ok=True)

        # Copy files from temp to output
        import shutil
        temp_files = [f for f in os.listdir('/tmp') if f.startswith(('dim_', 'fact_'))]

        for temp_file in temp_files:
            src = os.path.join('/tmp', temp_file)
            dst = os.path.join(output_dir, temp_file)
            shutil.copy2(src, dst)
            logging.info(f"Copied {temp_file} to output directory")

        print(f"✓ Loaded {len(temp_files)} files to {output_dir}")

    except Exception as e:
        logging.error(f"Load failed: {str(e)}")
        raise

def quality_check():
    """Basic data quality checks"""
    try:
        output_dir = os.path.join(dag_path, 'star_schema_output')

        # Check if files exist
        files = os.listdir(output_dir)
        logging.info(f"Output files: {files}")

        # Basic file size check
        for file in files:
            file_path = os.path.join(output_dir, file)
            size = os.path.getsize(file_path)
            logging.info(f"{file}: {size} bytes")

        print("✓ Quality checks passed")

    except Exception as e:
        logging.error(f"Quality check failed: {str(e)}")
        raise

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'samsung_galaxy_etl_star_schema',
    default_args=default_args,
    description='Samsung Galaxy ETL Pipeline aligned with Star Schema',
    schedule_interval='@daily',
    catchup=False,
    tags=['samsung', 'etl', 'star_schema']
)

# Define tasks
extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform',
    python_callable=transform_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load',
    python_callable=load_data,
    dag=dag,
)

check_task = PythonOperator(
    task_id='quality_check',
    python_callable=quality_check,
    dag=dag,
)

# Set dependencies
extract_task >> transform_task >> load_task >> check_task
