import csv
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from dashboard.models import DimTime, DimProduct, DimRegion, SalesFactModel, SalesFactRegion, SalesFact5G

DATA_DIR = os.path.join(settings.BASE_DIR, 'dashboard', 'data')

def parse_float(value):
    return float(value.replace('.', '').replace(',', '.'))

class Command(BaseCommand):
    help = 'Import CSV files into the database'

    def handle(self, *args, **options):
        self.import_dim_time()
        self.import_dim_product()
        self.import_dim_region()
        self.import_fact_table_model()
        self.import_fact_table_region()
        self.import_fact_table_5g()

    def import_dim_time(self):
        try:
            with open(os.path.join(DATA_DIR, 'dim_time.csv'), newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    year = int(row['Year'])
                    quarter = int(row['Quarter'].replace('Q', ''))
                    DimTime.objects.get_or_create(year=year, quarter=quarter)
            self.stdout.write(self.style.SUCCESS('✔ dim_time.csv imported'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error importing dim_time: {e}'))

    def import_dim_product(self):
        try:
            with open(os.path.join(DATA_DIR, 'dim_product.csv'), newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    DimProduct.objects.get_or_create(
                        model_name=row['Product Model'],
                        is_5g=row['5G Capability'].strip().lower() == 'yes'
                    )
            self.stdout.write(self.style.SUCCESS('✔ dim_product.csv imported'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error importing dim_product: {e}'))

    def import_dim_region(self):
        try:
            with open(os.path.join(DATA_DIR, 'dim_region.csv'), newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    DimRegion.objects.get_or_create(
                        region_name=row['Region'],
                        regional_5g_coverage=parse_float(row['Regional 5G Coverage (%)']),
                        subscribers_5g_million=parse_float(row['5G Subscribers (millions)']),
                        avg_5g_speed_mbps=parse_float(row['Avg 5G Speed (Mbps)']),
                        preference_5g_percent=parse_float(row['Preference for 5G (%)'])
                    )
            self.stdout.write(self.style.SUCCESS('✔ dim_region.csv imported'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error importing dim_region: {e}'))

    def import_fact_table_model(self):
        try:
            with open(os.path.join(DATA_DIR, 'fact_table_model.csv'), newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    product = DimProduct.objects.get(id=int(row['product_id']))
                    year = int(row['Year'])
                    quarter = int(row['Quarter'].replace('Q', ''))
                    time = DimTime.objects.get(year=year, quarter=quarter)
                    SalesFactModel.objects.create(
                        product=product,
                        time=time,
                        units_sold=int(row['Units Sold']),
                        revenue=parse_float(row['Revenue ($)'])
                    )
            self.stdout.write(self.style.SUCCESS('✔ fact_table_model.csv imported'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error importing fact_table_model: {e}'))

    def import_fact_table_region(self):
        try:
            with open(os.path.join(DATA_DIR, 'fact_table_region.csv'), newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    region = DimRegion.objects.get(id=int(row['region_id']))
                    year = int(row['Year'])
                    quarter = int(row['Quarter'].replace('Q', ''))
                    time = DimTime.objects.get(year=year, quarter=quarter)
                    SalesFactRegion.objects.create(
                        region=region,
                        time=time,
                        units_sold=int(row['Units Sold']),
                        revenue=parse_float(row['Revenue ($)']),
                        market_share=parse_float(row['Market Share (%)'])
                    )
            self.stdout.write(self.style.SUCCESS('✔ fact_table_region.csv imported'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error importing fact_table_region: {e}'))

    def import_fact_table_5g(self):
        try:
            with open(os.path.join(DATA_DIR, 'fact_table_5g.csv'), newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    product = DimProduct.objects.get(id=int(row['product_id']))
                    region = DimRegion.objects.get(id=int(row['region_id']))
                    year = int(row['Year'])
                    quarter = int(row['Quarter'].replace('Q', ''))
                    time = DimTime.objects.get(year=year, quarter=quarter)
                    SalesFact5G.objects.create(
                        product=product,
                        region=region,
                        time=time,
                        units_sold=int(row['Units Sold']),
                        revenue=parse_float(row['Revenue ($)'])
                    )
            self.stdout.write(self.style.SUCCESS('✔ fact_table_5g.csv imported'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error importing fact_table_5g: {e}'))
