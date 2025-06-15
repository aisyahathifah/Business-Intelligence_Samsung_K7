from django.db import models

# --- DIMENSION TABLES ---

class DimProduct(models.Model):
    model_name = models.CharField(max_length=100)
    is_5g = models.BooleanField()  # True = 5G, False = Non-5G

    def __str__(self):
        return self.model_name

class DimRegion(models.Model):
    region_name = models.CharField(max_length=100)
    regional_5g_coverage = models.FloatField()  # in percentage
    subscribers_5g_million = models.FloatField()
    avg_5g_speed_mbps = models.FloatField()
    preference_5g_percent = models.FloatField()

    def __str__(self):
        return self.region_name

class DimTime(models.Model):
    year = models.IntegerField()
    quarter = models.IntegerField(choices=[(1, "Q1"), (2, "Q2"), (3, "Q3"), (4, "Q4")])

    def __str__(self):
        return f"{self.year} Q{self.quarter}"


# --- FACT TABLES ---

class SalesFactModel(models.Model):
    product = models.ForeignKey(DimProduct, on_delete=models.CASCADE)
    time = models.ForeignKey(DimTime, on_delete=models.CASCADE)
    units_sold = models.IntegerField()
    revenue = models.FloatField()

class SalesFactRegion(models.Model):
    region = models.ForeignKey(DimRegion, on_delete=models.CASCADE)
    time = models.ForeignKey(DimTime, on_delete=models.CASCADE)
    units_sold = models.IntegerField()
    revenue = models.FloatField()
    market_share = models.FloatField()  # in percentage

class SalesFact5G(models.Model):
    product = models.ForeignKey(DimProduct, on_delete=models.CASCADE)
    region = models.ForeignKey(DimRegion, on_delete=models.CASCADE)
    time = models.ForeignKey(DimTime, on_delete=models.CASCADE)
    units_sold = models.IntegerField()
    revenue = models.FloatField()
