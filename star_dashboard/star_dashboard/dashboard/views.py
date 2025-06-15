from django.shortcuts import render
from django.db import models
from .models import SalesFactModel, SalesFactRegion, SalesFact5G
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import defaultdict
from django.shortcuts import render
from django.http import JsonResponse
from django.db import models
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from dashboard.models import DimTime, DimProduct, DimRegion, SalesFactModel, SalesFactRegion, SalesFact5G

# =============================
#1. FIXED VERSION - Prediksi Penjualan per Model Produk
# =============================
import json
from collections import defaultdict
from django.shortcuts import render
from django.http import JsonResponse
from django.db import models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from .models import SalesFactModel, DimProduct, DimTime, DimRegion, SalesFactRegion, SalesFact5G

def predict_sales_by_model(request):
    """
    Prediksi penjualan berdasarkan model produk menggunakan star schema
    """
    try:
        # Debug: Cek apakah ada data
        total_records = SalesFactModel.objects.count()
        print(f"Total SalesFactModel records: {total_records}")
        
        if total_records == 0:
            return render(request, 'dashboard/predict_by_model.html', {
                'error': 'Tidak ada data penjualan yang tersedia di database',
                'labels': json.dumps([]),
                'datasets': json.dumps([]),
                'debug_info': f'Total records: {total_records}'
            })
        
        # Ambil total penjualan per model untuk menentukan top 3
        top_models = (
            SalesFactModel.objects
            .select_related('product')
            .values('product__model_name')
            .annotate(
                total_sales=models.Sum('units_sold'),
                total_revenue=models.Sum('revenue')
            )
            .order_by('-total_sales')[:3]
        )
        
        print(f"Top models found: {list(top_models)}")
        
        if not top_models:
            return render(request, 'dashboard/predict_by_model.html', {
                'error': 'Tidak ada data penjualan yang dapat diproses',
                'labels': json.dumps([]),
                'datasets': json.dumps([]),
                'debug_info': 'Top models query returned empty'
            })
        
        top_model_names = [item['product__model_name'] for item in top_models]
        
        # Ambil data detail untuk top 3 models
        data = list(
            SalesFactModel.objects
            .select_related('product', 'time')
            .filter(product__model_name__in=top_model_names)
            .values(
                'product__model_name', 
                'product__is_5g',
                'time__year', 
                'time__quarter', 
                'units_sold',
                'revenue'
            )
            .order_by('product__model_name', 'time__year', 'time__quarter')
        )
        
        print(f"Detailed data found: {len(data)} records")
        
        if not data:
            return render(request, 'dashboard/predict_by_model.html', {
                'error': 'Tidak ada data detail untuk model terpilih',
                'labels': json.dumps([]),
                'datasets': json.dumps([]),
                'debug_info': f'Models: {top_model_names}'
            })
        
        # Kelompokkan berdasarkan model
        grouped = defaultdict(list)
        all_labels = set()
        
        for row in data:
            key = row['product__model_name']
            time_val = row['time__year'] + (row['time__quarter'] - 1) / 4
            quarter_label = f"{row['time__year']} Q{row['time__quarter']}"
            
            grouped[key].append({
                'time_val': time_val,
                'units_sold': row['units_sold'],
                'revenue': row['revenue'],
                'quarter_label': quarter_label,
                'is_5g': row['product__is_5g']
            })
            all_labels.add(quarter_label)
        
        # Konversi set ke list dan sort
        all_labels = sorted(list(all_labels))
        
        # Warna untuk models
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726', '#AB47BC']
        datasets = []
        model_stats = []
        
        for idx, (model_name, points) in enumerate(grouped.items()):
            if len(points) < 2:
                continue
            
            # Prepare data untuk machine learning
            X = np.array([[p['time_val']] for p in points])
            y_units = np.array([p['units_sold'] for p in points])
            
            # Model prediksi untuk units sold
            model_units = LinearRegression().fit(X, y_units)
            y_pred_units = model_units.predict(X)
            
            # Hitung metrics
            r2_units = r2_score(y_units, y_pred_units)
            mae_units = mean_absolute_error(y_units, y_pred_units)
            
            color = colors[idx % len(colors)]
            
            # Dataset untuk UNITS SOLD - Data Aktual
            actual_data = []
            predicted_data = []
            
            for i, point in enumerate(points):
                actual_data.append(point['units_sold'])
                predicted_data.append(float(y_pred_units[i]))
            
            # Dataset untuk data aktual
            datasets.append({
                'label': f'{model_name} - Aktual',
                'data': actual_data,
                'borderColor': color,
                'backgroundColor': color + '30',
                'fill': False,
                'borderWidth': 3,
                'pointRadius': 5,
                'tension': 0.1
            })
            
            # Dataset untuk prediksi
            datasets.append({
                'label': f'{model_name} - Prediksi',
                'data': predicted_data,
                'borderColor': color,
                'backgroundColor': color + '50',
                'fill': False,
                'borderWidth': 2,
                'borderDash': [5, 5],
                'pointRadius': 3,
                'tension': 0.1
            })
            
            # Simpan statistik model
            is_5g = points[0]['is_5g'] if points else False
            model_stats.append({
                'model_name': model_name,
                'is_5g': is_5g,
                'total_units': sum(p['units_sold'] for p in points),
                'total_revenue': sum(p['revenue'] for p in points),
                'avg_units_per_quarter': float(np.mean(y_units)),
                'trend_units': 'Naik' if model_units.coef_[0] > 0 else 'Turun',
                'r2_score': round(r2_units, 3),
                'mae': round(mae_units, 2),
                'quarters_count': len(points)
            })
        
        # Prediksi untuk quarter berikutnya
        future_predictions = []
        if grouped:
            # Get last time value
            all_time_vals = []
            for points in grouped.values():
                all_time_vals.extend([p['time_val'] for p in points])
            
            if all_time_vals:
                last_time_val = max(all_time_vals)
                next_time_val = last_time_val + 0.25
                next_year = int(next_time_val)
                next_quarter = int((next_time_val - next_year) * 4) + 1
                
                if next_quarter > 4:
                    next_year += 1
                    next_quarter = 1
                
                for model_name, points in grouped.items():
                    if len(points) < 2:
                        continue
                    
                    X = np.array([[p['time_val']] for p in points])
                    y_units = np.array([p['units_sold'] for p in points])
                    
                    model_units = LinearRegression().fit(X, y_units)
                    next_pred = model_units.predict([[next_time_val]])[0]
                    
                    future_predictions.append({
                        'model_name': model_name,
                        'predicted_units': max(0, int(next_pred)),
                        'quarter': f"{next_year} Q{next_quarter}"
                    })
        
        # Konversi ke JSON untuk template
        labels_json = json.dumps(all_labels)
        datasets_json = json.dumps(datasets)
        
        print(f"Labels: {all_labels}")
        print(f"Datasets count: {len(datasets)}")
        
        context = {
            'labels': labels_json,
            'datasets': datasets_json,
            'top_models': top_models,
            'model_stats': model_stats,
            'future_predictions': future_predictions,
            'chart_title': 'Prediksi Penjualan per Model Produk (Top 3)',
            'schema_info': {
                'fact_table': 'SalesFactModel',
                'dimensions': ['DimProduct', 'DimTime'],
                'metrics': ['Units Sold', 'Revenue']
            },
            'debug_info': f'Total records: {total_records}, Models: {len(grouped)}, Datasets: {len(datasets)}'
        }
        
        return render(request, 'dashboard/predict_by_model.html', context)
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error in predict_sales_by_model: {error_detail}")
        
        return render(request, 'dashboard/predict_by_model.html', {
            'error': f'Terjadi kesalahan: {str(e)}',
            'labels': json.dumps([]),
            'datasets': json.dumps([]),
            'debug_info': error_detail
        })


def analyze_sales_by_category(request):
    """
    Analisis penjualan berdasarkan kategori 5G capability
    """
    try:
        # Ambil data berdasarkan kategori 5G
        category_data = list(
            SalesFactModel.objects
            .select_related('product', 'time')
            .values(
                'product__is_5g',
                'time__year',
                'time__quarter'
            )
            .annotate(
                total_units=models.Sum('units_sold'),
                total_revenue=models.Sum('revenue')
            )
            .order_by('time__year', 'time__quarter')
        )
        
        if not category_data:
            return render(request, 'dashboard/analyze_by_category.html', {
                'error': 'Tidak ada data kategori yang tersedia',
                'labels': json.dumps([]),
                'datasets': json.dumps([])
            })
        
        # Kelompokkan berdasarkan kategori
        categories = {'5G': [], 'Non-5G': []}
        labels = set()
        
        for row in category_data:
            category = '5G' if row['product__is_5g'] else 'Non-5G'
            quarter_label = f"{row['time__year']} Q{row['time__quarter']}"
            labels.add(quarter_label)
            
            categories[category].append({
                'quarter': quarter_label,
                'units': row['total_units'],
                'revenue': row['total_revenue']
            })
        
        labels = sorted(list(labels))
        
        # Buat datasets untuk chart
        datasets = []
        colors = {'5G': '#FF6B6B', 'Non-5G': '#4ECDC4'}
        
        for category, data in categories.items():
            if not data:
                continue
            
            # Buat mapping untuk semua quarters
            quarter_map = {item['quarter']: item for item in data}
            units_data = []
            
            for label in labels:
                if label in quarter_map:
                    units_data.append(quarter_map[label]['units'])
                else:
                    units_data.append(0)
            
            datasets.append({
                'label': f'{category} (Units)',
                'data': units_data,
                'borderColor': colors[category],
                'backgroundColor': colors[category] + '30',
                'fill': False,
                'borderWidth': 3
            })
        
        return render(request, 'dashboard/analyze_by_category.html', {
            'labels': json.dumps(labels),
            'datasets': json.dumps(datasets),
            'categories': categories,
            'chart_title': 'Analisis Penjualan: 5G vs Non-5G'
        })
    
    except Exception as e:
        return render(request, 'dashboard/analyze_by_category.html', {
            'error': f'Terjadi kesalahan: {str(e)}',
            'labels': json.dumps([]),
            'datasets': json.dumps([])
        })


def dashboard_summary(request):
    """
    Dashboard utama dengan ringkasan dari star schema
    """
    try:
        # Total penjualan keseluruhan
        total_stats = SalesFactModel.objects.aggregate(
            total_units=models.Sum('units_sold'),
            total_revenue=models.Sum('revenue'),
            total_models=models.Count('product__model_name', distinct=True)
        )
        
        # Penjualan per kategori 5G
        category_stats = list(
            SalesFactModel.objects
            .select_related('product')
            .values('product__is_5g')
            .annotate(
                total_units=models.Sum('units_sold'),
                total_revenue=models.Sum('revenue'),
                model_count=models.Count('product__model_name', distinct=True)
            )
        )
        
        # Trend penjualan per quarter
        quarterly_trend = list(
            SalesFactModel.objects
            .select_related('time')
            .values('time__year', 'time__quarter')
            .annotate(
                total_units=models.Sum('units_sold'),
                total_revenue=models.Sum('revenue')
            )
            .order_by('time__year', 'time__quarter')
        )
        
        # Top performing models
        top_models = list(
            SalesFactModel.objects
            .select_related('product')
            .values('product__model_name', 'product__is_5g')
            .annotate(
                total_units=models.Sum('units_sold'),
                total_revenue=models.Sum('revenue')
            )
            .order_by('-total_units')[:5]
        )
        
        return render(request, 'dashboard/summary.html', {
            'total_stats': total_stats,
            'category_stats': category_stats,
            'quarterly_trend': quarterly_trend,
            'top_models': top_models,
            'schema_info': {
                'total_fact_records': SalesFactModel.objects.count(),
                'total_products': DimProduct.objects.count(),
                'total_time_periods': DimTime.objects.count()
            }
        })
    
    except Exception as e:
        return render(request, 'dashboard/summary.html', {
            'error': f'Terjadi kesalahan: {str(e)}'
        })


def api_model_prediction(request, model_name):
    """
    API endpoint untuk prediksi spesifik model
    """
    try:
        data = list(
            SalesFactModel.objects
            .select_related('product', 'time')
            .filter(product__model_name=model_name)
            .values('time__year', 'time__quarter', 'units_sold', 'revenue')
            .order_by('time__year', 'time__quarter')
        )
        
        if not data:
            return JsonResponse({'error': 'Model tidak ditemukan'}, status=404)
        
        # Prepare data untuk prediksi
        points = []
        for row in data:
            time_val = row['time__year'] + (row['time__quarter'] - 1) / 4
            points.append({
                'time_val': time_val,
                'units_sold': row['units_sold'],
                'revenue': row['revenue'],
                'quarter': f"{row['time__year']} Q{row['time__quarter']}"
            })
        
        if len(points) < 2:
            return JsonResponse({'error': 'Data tidak cukup untuk prediksi'}, status=400)
        
        # Prediksi menggunakan Linear Regression
        X = np.array([[p['time_val']] for p in points])
        y = np.array([p['units_sold'] for p in points])
        
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        
        # Prediksi quarter berikutnya
        last_time = max(p['time_val'] for p in points)
        next_time = last_time + 0.25
        next_prediction = model.predict([[next_time]])[0]
        
        return JsonResponse({
            'model_name': model_name,
            'historical_data': points,
            'predictions': y_pred.tolist(),
            'next_quarter_prediction': max(0, int(next_prediction)),
            'r2_score': float(r2_score(y, y_pred)),
            'trend': 'Naik' if model.coef_[0] > 0 else 'Turun'
        })
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
# =============================
# 2. Prediksi Penjualan per Wilayah (Revised untuk Total Unit: 5G + Non-5G)
# =============================
from django.shortcuts import render
from django.db.models import Sum
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression
from dashboard.models import SalesFactRegion, DimRegion, DimTime

def predict_sales_by_region(request):
    # Ambil data penjualan per region dengan join ke dimension tables
    data = (
        SalesFactRegion.objects
        .select_related('region', 'time')
        .values(
            'region__region_name',
            'region__regional_5g_coverage', 
            'region__subscribers_5g_million',
            'region__avg_5g_speed_mbps',
            'region__preference_5g_percent',
            'time__year', 
            'time__quarter', 
            'units_sold',
            'revenue',
            'market_share'
        )
        .order_by('region__region_name', 'time__year', 'time__quarter')
    )

    regional_stats = {}
    region_totals = defaultdict(lambda: {
        'total_units': 0, 
        'total_revenue': 0, 
        'avg_market_share': 0,
        'quarters_count': 0,
        '5g_coverage': 0,
        'subscribers_5g': 0,
        'avg_5g_speed': 0,
        'preference_5g': 0
    })

    grouped = defaultdict(list)
    time_label_totals = defaultdict(int)  # Untuk total units across all regions and types

    for row in data:
        region_name = row['region__region_name']
        time_label = f"{row['time__year']} Q{row['time__quarter']}"
        
        # Sum total units per region
        region_totals[region_name]['total_units'] += row['units_sold']
        region_totals[region_name]['total_revenue'] += row['revenue']
        region_totals[region_name]['avg_market_share'] += row['market_share']
        region_totals[region_name]['quarters_count'] += 1
        
        # Untuk total global units per waktu
        time_label_totals[time_label] += row['units_sold']
        
        # Store regional dimension
        region_totals[region_name]['5g_coverage'] = row['region__regional_5g_coverage']
        region_totals[region_name]['subscribers_5g'] = row['region__subscribers_5g_million']
        region_totals[region_name]['avg_5g_speed'] = row['region__avg_5g_speed_mbps']
        region_totals[region_name]['preference_5g'] = row['region__preference_5g_percent']
        
        time_val = row['time__year'] + (row['time__quarter'] - 1) / 4
        grouped[region_name].append((
            time_val, 
            row['units_sold'], 
            row['revenue'],
            row['market_share'],
            time_label,
            row['time__year'],
            row['time__quarter']
        ))

    # Hitung rata-rata market share
    for region_name in region_totals:
        if region_totals[region_name]['quarters_count'] > 0:
            region_totals[region_name]['avg_market_share'] /= region_totals[region_name]['quarters_count']

    top_3_regions = sorted(region_totals.items(), key=lambda x: x[1]['total_units'], reverse=True)[:3]
    top_3_region_names = [region[0] for region in top_3_regions]

    filtered_grouped = {region: points for region, points in grouped.items() 
                       if region in top_3_region_names}

    all_labels = sorted(list(time_label_totals.keys()))

    datasets = []
    regional_insights = []

    for region_name in top_3_region_names:
        points = filtered_grouped[region_name]
        region_stats = region_totals[region_name]

        actuals_units = [None] * len(all_labels)
        actuals_revenue = [None] * len(all_labels)
        actuals_market_share = [None] * len(all_labels)
        
        time_to_data = {}
        for point in points:
            time_label = point[4]
            time_to_data[time_label] = {
                'units': point[1],
                'revenue': point[2], 
                'market_share': point[3]
            }
        
        for i, label in enumerate(all_labels):
            if label in time_to_data:
                actuals_units[i] = time_to_data[label]['units']
                actuals_revenue[i] = time_to_data[label]['revenue']
                actuals_market_share[i] = time_to_data[label]['market_share']

        # Regression
        X_features = []
        y_values = []
        for point in points:
            time_val, units_sold, revenue, market_share, label, year, quarter = point
            features = [
                time_val,
                quarter,
                np.sin(2 * np.pi * quarter / 4),
                np.cos(2 * np.pi * quarter / 4),
                region_stats['5g_coverage'] / 100,
                region_stats['preference_5g'] / 100,
                region_stats['avg_5g_speed'] / 1000,
            ]
            X_features.append(features)
            y_values.append(units_sold)

        if len(X_features) > 0:
            X = np.array(X_features)
            y = np.array(y_values)

            model = LinearRegression()
            model.fit(X, y)

            r2_score_val = model.score(X, y)
            all_predictions = []
            for label in all_labels:
                if label in time_to_data:
                    year, quarter = label.split(' Q')
                    year, quarter = int(year), int(quarter)
                    time_val = year + (quarter - 1) / 4
                    features = [
                        time_val,
                        quarter,
                        np.sin(2 * np.pi * quarter / 4),
                        np.cos(2 * np.pi * quarter / 4),
                        region_stats['5g_coverage'] / 100,
                        region_stats['preference_5g'] / 100,
                        region_stats['avg_5g_speed'] / 1000,
                    ]
                    pred = model.predict([features])[0]
                    all_predictions.append(max(0, pred))
                else:
                    all_predictions.append(None)

            actual_values = [actuals_units[i] for i in range(len(actuals_units)) if actuals_units[i] is not None]
            pred_values = [all_predictions[i] for i in range(len(all_predictions)) if actuals_units[i] is not None]

            mae = np.mean(np.abs(np.array(actual_values) - np.array(pred_values))) if actual_values else 0
            rmse = np.sqrt(np.mean((np.array(actual_values) - np.array(pred_values))**2)) if actual_values else 0

            datasets.append({
                'region': region_name,
                'actuals_units': actuals_units,
                'actuals_revenue': actuals_revenue,
                'actuals_market_share': actuals_market_share,
                'predictions': all_predictions,
                'model_score': r2_score_val,
                'mae': mae,
                'rmse': rmse
            })

            regional_insights.append({
                'region': region_name,
                'total_units_sold': region_stats['total_units'],
                'total_revenue': region_stats['total_revenue'],
                'avg_market_share': region_stats['avg_market_share'],
                'regional_5g_coverage': region_stats['5g_coverage'],
                'subscribers_5g_million': region_stats['subscribers_5g'],
                'avg_5g_speed_mbps': region_stats['avg_5g_speed'],
                'preference_5g_percent': region_stats['preference_5g'],
                'quarters_analyzed': region_stats['quarters_count']
            })

    overall_stats = {
        'total_regions_analyzed': len(top_3_region_names),
        'total_time_periods': len(all_labels),
        'total_units_sold': sum(time_label_totals.values()),  # Revisi di sini!
        'total_revenue': sum([insight['total_revenue'] for insight in regional_insights]),
        'avg_model_accuracy': np.mean([ds['model_score'] for ds in datasets]) if datasets else 0
    }

    return render(request, 'dashboard/predict_by_region.html', {
        'labels': all_labels,
        'datasets': datasets,
        'regional_insights': regional_insights,
        'overall_stats': overall_stats,
        'top_regions': top_3_region_names,
        'region_totals': {region[0]: region[1] for region in top_3_regions}
    })


# =============================
# 3. Prediksi Penjualan Berdasarkan Dampak 5G (Enhanced)
# =============================
from django.shortcuts import render
from sklearn.linear_model import LinearRegression
import numpy as np
from .models import SalesFact5G, DimProduct, DimRegion, DimTime

def predict_sales_5g(request):
    print("=== CHECKING DATA AVAILABILITY ===")
    total_records = SalesFact5G.objects.count()
    print(f"Total SalesFact5G records: {total_records}")
    
    data = (
        SalesFact5G.objects
        .select_related('product', 'time', 'region')
        .values(
            'product__model_name',
            'product__is_5g',
            'time__year',
            'time__quarter',
            'region__region_name',
            'region__regional_5g_coverage',
            'region__subscribers_5g_million',
            'region__avg_5g_speed_mbps',
            'region__preference_5g_percent',
            'units_sold',
            'revenue'
        )
        .order_by('time__year', 'time__quarter', 'product__is_5g')
    )
    
    data_list = list(data)
    print(f"Query result count: {len(data_list)}")
    for i, row in enumerate(data_list[:3]):
        print(f"Sample Row {i}: {row}")

    grouped_5g = {}
    grouped_non5g = {}
    revenue_5g = {}
    revenue_non5g = {}
    regional_analysis = {}
    all_time_labels = set()
    total_revenue_all = 0  # Penambahan total revenue keseluruhan

    for row in data_list:
        time_label = f"{row['time__year']} Q{row['time__quarter']}"
        units = row['units_sold'] or 0
        revenue = row['revenue'] or 0
        total_revenue_all += revenue  # Akumulasi total revenue
        region = row['region__region_name']
        coverage_5g = row['region__regional_5g_coverage'] or 0
        subscribers_5g = row['region__subscribers_5g_million'] or 0
        avg_speed = row['region__avg_5g_speed_mbps'] or 0
        preference_5g = row['region__preference_5g_percent'] or 0
        all_time_labels.add(time_label)
        is_5g_product = row['product__is_5g']

        if is_5g_product:
            grouped_5g[time_label] = grouped_5g.get(time_label, 0) + units
            revenue_5g[time_label] = revenue_5g.get(time_label, 0) + revenue
        else:
            grouped_non5g[time_label] = grouped_non5g.get(time_label, 0) + units
            revenue_non5g[time_label] = revenue_non5g.get(time_label, 0) + revenue

        if region not in regional_analysis:
            regional_analysis[region] = {
                'regional_5g_coverage': coverage_5g,
                'subscribers_5g_million': subscribers_5g,
                'avg_5g_speed_mbps': avg_speed,
                'preference_5g_percent': preference_5g,
                'total_5g_sales': 0,
                'total_non5g_sales': 0,
                'total_5g_revenue': 0,
                'total_non5g_revenue': 0
            }

        if is_5g_product:
            regional_analysis[region]['total_5g_sales'] += units
            regional_analysis[region]['total_5g_revenue'] += revenue
        else:
            regional_analysis[region]['total_non5g_sales'] += units
            regional_analysis[region]['total_non5g_revenue'] += revenue

    all_labels = sorted(list(all_time_labels))

    print("\n=== REGIONAL 5G IMPACT ANALYSIS ===")
    regional_insights = []
    for region, data_reg in regional_analysis.items():
        total_sales = data_reg['total_5g_sales'] + data_reg['total_non5g_sales']
        total_revenue = data_reg['total_5g_revenue'] + data_reg['total_non5g_revenue']
        if total_sales > 0:
            ratio_5g_sales = (data_reg['total_5g_sales'] / total_sales) * 100
            ratio_5g_revenue = (data_reg['total_5g_revenue'] / total_revenue) * 100 if total_revenue > 0 else 0
            regional_insights.append({
                'region': region,
                'coverage': data_reg['regional_5g_coverage'],
                'subscribers': data_reg['subscribers_5g_million'],
                'avg_speed': data_reg['avg_5g_speed_mbps'],
                'preference': data_reg['preference_5g_percent'],
                'sales_ratio_5g': round(ratio_5g_sales, 1),
                'revenue_ratio_5g': round(ratio_5g_revenue, 1)
            })

    if not all_labels:
        return render(request, 'dashboard/predict_5g.html', {
            'labels': [],
            'actuals': [],
            'predictions': [],
            'actuals_5g': [],
            'actuals_non5g': [],
            'predictions_5g': [],
            'predictions_non5g': [],
            'regional_insights': [],
            'revenue_5g': [],
            'revenue_non5g': [],
            'total_5g_units': 0,
            'total_non5g_units': 0,
            'market_share_5g': 0,
            'total_revenue': 0
        })

    actuals_5g, actuals_non5g = [], []
    revenue_5g_arr, revenue_non5g_arr = [], []
    for label in all_labels:
        actuals_5g.append(grouped_5g.get(label, 0))
        actuals_non5g.append(grouped_non5g.get(label, 0))
        revenue_5g_arr.append(revenue_5g.get(label, 0))
        revenue_non5g_arr.append(revenue_non5g.get(label, 0))

    actuals_total = [a + b for a, b in zip(actuals_5g, actuals_non5g)]

    predictions_5g, predictions_non5g = [], []
    if len(actuals_5g) > 1:
        try:
            X = np.array([[i] for i in range(len(actuals_5g))])
            if len(set(actuals_5g)) > 1:
                model_5g = LinearRegression().fit(X, np.array(actuals_5g))
                pred_5g = model_5g.predict(X).tolist()
            else:
                pred_5g = actuals_5g.copy()

            if len(set(actuals_non5g)) > 1:
                model_non5g = LinearRegression().fit(X, np.array(actuals_non5g))
                pred_non5g = model_non5g.predict(X).tolist()
            else:
                pred_non5g = actuals_non5g.copy()

            predictions_5g = [max(0, p) for p in pred_5g]
            predictions_non5g = [max(0, p) for p in pred_non5g]
        except Exception as e:
            print(f"Error in prediction: {e}")
            predictions_5g = actuals_5g.copy()
            predictions_non5g = actuals_non5g.copy()
    else:
        predictions_5g = actuals_5g.copy()
        predictions_non5g = actuals_non5g.copy()

    predictions_total = [a + b for a, b in zip(predictions_5g, predictions_non5g)]
    total_5g = sum(actuals_5g)
    total_non5g = sum(actuals_non5g)
    grand_total = total_5g + total_non5g

    ratio_5g = (total_5g / grand_total) * 100 if grand_total > 0 else 0

    return render(request, 'dashboard/predict_5g.html', {
        'labels': all_labels,
        'actuals': actuals_total,
        'predictions': predictions_total,
        'actuals_5g': actuals_5g,
        'actuals_non5g': actuals_non5g,
        'predictions_5g': predictions_5g,
        'predictions_non5g': predictions_non5g,
        'regional_insights': regional_insights,
        'revenue_5g': revenue_5g_arr,
        'revenue_non5g': revenue_non5g_arr,
        'total_5g_units': total_5g,
        'total_non5g_units': total_non5g,
        'market_share_5g': round(ratio_5g, 1),
        'total_revenue': total_revenue_all  # <- Hasil akhir total revenue dari semua produk
    })
