from django.urls import path
from . import views

urlpatterns = [
    path('predict/model/', views.predict_sales_by_model, name='predict_by_model'),
    path('predict/region/', views.predict_sales_by_region, name='predict_by_region'),
    path('predict/5g/', views.predict_sales_5g, name='predict_sales_5g'),
]
