# Yapay Zeka ile Veri Analizi Pratik Örnekleri
# Bu kod örnekleri, veri analitiğinde yapay zeka ve istatistiksel yöntemlerin
# nasıl kullanılabileceğini göstermektedir.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, silhouette_score
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
sns.set_style('whitegrid')

# ----------------------------------------------------------------------------
# ÖRNEK 1: ZAMAN SERİSİ ANALİZİ VE SATIŞ TAHMİNİ
# ----------------------------------------------------------------------------

def create_sample_sales_data(n_days=1095):
    """3 yıllık yapay satış verisi oluşturur"""
    
    # Tarih aralığı oluştur (3 yıl)
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    
    # Trend bileşeni
    trend = np.linspace(100, 300, n_days)
    
    # Mevsimsellik bileşeni (haftalık ve yıllık)
    weekly_seasonality = 20 * np.sin(np.arange(n_days) * (2 * np.pi / 7))
    yearly_seasonality = 100 * np.sin(np.arange(n_days) * (2 * np.pi / 365))
    
    # Tatil etkisi
    holidays = pd.Series(0, index=dates)
    # Yılbaşı, Ramazan ve Kurban Bayramı
    holidays['2022-01-01'] = 100
    holidays['2022-05-02'] = 120
    holidays['2022-05-03'] = 150
    holidays['2022-05-04'] = 120
    holidays['2022-07-09'] = 120
    holidays['2022-07-10'] = 150
    holidays['2022-07-11'] = 140
    holidays['2022-07-12'] = 110
    
    holidays['2023-01-01'] = 110
    holidays['2023-04-21'] = 130
    holidays['2023-04-22'] = 160
    holidays['2023-04-23'] = 130
    holidays['2023-06-28'] = 130
    holidays['2023-06-29'] = 160
    holidays['2023-06-30'] = 150
    holidays['2023-07-01'] = 120
    
    holidays['2024-01-01'] = 120
    holidays['2024-04-10'] = 140
    holidays['2024-04-11'] = 170
    holidays['2024-04-12'] = 140
    holidays['2024-06-16'] = 140
    holidays['2024-06-17'] = 170
    holidays['2024-06-18'] = 160
    holidays['2024-06-19'] = 130
    
    # Promosyon etkisi
    promotions = pd.Series(0, index=dates)
    # Yılda 4 kez büyük promosyon (her 3 ayda bir)
    for year in [2022, 2023, 2024]:
        for month in [3, 6, 9, 12]:
            start_date = pd.Timestamp(f"{year}-{month:02d}-01")
            end_date = start_date + pd.Timedelta(days=6)
            mask = (dates >= start_date) & (dates <= end_date)
            promotions[mask] = 80
    
    # Rastgele gürültü
    noise = np.random.normal(0, 20, n_days)
    
    # Tüm bileşenleri birleştir
    sales = trend + weekly_seasonality + yearly_seasonality + holidays.values + promotions.values + noise
    
    # Veri çerçevesi oluştur
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'weekday': dates.dayofweek,
        'month': dates.month,
        'year': dates.year,
        'is_weekend': dates.dayofweek >= 5,
        'is_holiday': holidays > 0,
        'is_promotion': promotions > 0,
        'day_of_year': dates.dayofyear
    })
    
    # Satış değerlerini pozitif yap
    df['sales'] = df['sales'].clip(lower=0)
    
    return df

def analyze_time_series(df):
    """Zaman serisi analizi yapar ve görselleştirir"""
    
    # Tarih sütununu dizin olarak ayarla
    df_ts = df.copy()
    df_ts.set_index('date', inplace=True)
    
    # Mevsimsel ayrıştırma
    result = seasonal_decompose(df_ts['sales'], model='additive', period=30)
    
    # Sonuçları görselleştir
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16))
    result.observed.plot(ax=ax1)
    ax1.set_title('Gözlemlenen Satışlar')
    ax1.set_ylabel('Satış Miktarı')
    
    result.trend.plot(ax=ax2)
    ax2.set_title('Trend Bileşeni')
    ax2.set_ylabel('Trend')
    
    result.seasonal.plot(ax=ax3)
    ax3.set_title('Mevsimsel Bileşen')
    ax3.set_ylabel('Mevsimsellik')
    
    result.resid.plot(ax=ax4)
    ax4.set_title('Artık (Residual) Bileşen')
    ax4.set_ylabel('Artık')
    
    plt.tight_layout()
    
    # Haftalık ortalama satışlar
    plt.figure(figsize=(12, 6))
    weekday_avg = df.groupby('weekday')['sales'].mean()
    weekday_names = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
    
    sns.barplot(x=[weekday_names[i] for i in range(7)], y=weekday_avg.values)
    plt.title('Haftanın Günlerine Göre Ortalama Satışlar')
    plt.xlabel('Gün')
    plt.ylabel('Ortalama Satış')
    plt.xticks(rotation=45)
    
    # Aylık ortalama satışlar
    plt.figure(figsize=(12, 6))
    month_avg = df.groupby('month')['sales'].mean()
    month_names = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 
                   'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
    
    sns.barplot(x=month_names, y=month_avg.values)
    plt.title('Aylara Göre Ortalama Satışlar')
    plt.xlabel('Ay')
    plt.ylabel('Ortalama Satış')
    plt.xticks(rotation=45)
    
    # Tatil günleri vs normal günler
    plt.figure(figsize=(10, 6))
    holiday_impact = df.groupby('is_holiday')['sales'].mean()
    sns.barplot(x=['Normal Gün', 'Tatil Günü'], y=holiday_impact.values)
    plt.title('Tatil Günlerinin Satışlara Etkisi')
    plt.xlabel('Gün Tipi')
    plt.ylabel('Ortalama Satış')
    
    # Promosyon dönemleri vs normal dönemler
    plt.figure(figsize=(10, 6))
    promo_impact = df.groupby('is_promotion')['sales'].mean()
    sns.barplot(x=['Normal Dönem', 'Promosyon Dönemi'], y=promo_impact.values)
    plt.title('Promosyonların Satışlara Etkisi')
    plt.xlabel('Dönem Tipi')
    plt.ylabel('Ortalama Satış')
    
    return result

def forecast_sales(df, forecast_days=30):
    """ARIMA modeli ile satış tahmini yapar"""
    
    # Tarihi dizin olarak ayarla
    df_forecast = df.copy()
    df_forecast.set_index('date', inplace=True)
    
    # ARIMA modeli için sadece satış verilerini al
    sales_series = df_forecast['sales']
    
    # ARIMA parametreleri (p, d, q) - p=otoregressif, d=fark alma, q=hareketli ortalama
    model = ARIMA(sales_series, order=(5, 1, 2))
    model_fit = model.fit()
    
    # Tahmin
def forecast_sales(df, forecast_days=30):
    """ARIMA modeli ile satış tahmini yapar"""

    print("Satış tahmini yapılıyor...")

    # Tarihi dizin olarak ayarla
    df_forecast = df.copy()
    df_forecast.set_index('date', inplace=True)

    # ARIMA modeli için sadece satış verilerini al
    sales_series = df_forecast['sales']

    # ARIMA parametreleri (p, d, q)
    model = ARIMA(sales_series, order=(5, 1, 2))
    model_fit = model.fit()

    # Tahmin ve güven aralığı
    forecast_object = model_fit.get_forecast(steps=forecast_days)
    forecast = forecast_object.predicted_mean
    conf_int = forecast_object.conf_int()

    forecast_index = pd.date_range(start=sales_series.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_series = pd.Series(forecast.values, index=forecast_index)

    # Görselleştirme
    plt.figure(figsize=(14, 7))
    plt.plot(sales_series.index[-90:], sales_series.values[-90:], label='Geçmiş Veriler')
    plt.plot(forecast_index, forecast, color='red', label='Tahmin')
    plt.fill_between(forecast_index,
                     conf_int.iloc[:, 0],
                     conf_int.iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.title('30 Günlük Satış Tahmini (ARIMA Modeli)')
    plt.xlabel('Tarih')
    plt.ylabel('Satış Miktarı')
    plt.legend()
    plt.grid(True)

    return forecast_series

def train_ml_sales_model(df):
    """Makine öğrenmesi modeli ile satış tahmini"""
    
    # Özellikler ve hedef değişken
    X = df.drop(['sales', 'date'], axis=1)
    y = df['sales']
    
    # One-hot encoding için kategorik değişkenler
    categorical_features = []
    numeric_features = ['weekday', 'month', 'year', 'day_of_year']
    binary_features = ['is_weekend', 'is_holiday', 'is_promotion']
    
    # Veri ön işleme pipeline'ı
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('bin', 'passthrough', binary_features)
        ])
    
    # Eğitim ve test verileri
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # RandomForest modeli
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # XGBoost modeli
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42))
    ])
    
    # Modelleri eğit
    rf_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)
    
    # Tahminler
    rf_preds = rf_pipeline.predict(X_test)
    xgb_preds = xgb_pipeline.predict(X_test)
    
    # Değerlendirme
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    
    print(f"RandomForest RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}")
    print(f"XGBoost RMSE: {xgb_rmse:.2f}, MAE: {xgb_mae:.2f}")
    
    # Özellik önemliliği
    if hasattr(rf_pipeline['model'], 'feature_importances_'):
        importances = rf_pipeline['model'].feature_importances_
        feature_names = numeric_features + binary_features
        
        plt.figure(figsize=(12, 6))
        indices = np.argsort(importances)[::-1]
        plt.title('RandomForest Özellik Önemliliği')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
    
    # Çapraz doğrulama
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(xgb_pipeline, X, y, cv=tscv, scoring='neg_root_mean_squared_error')
    print(f"XGBoost Çapraz Doğrulama RMSE: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
    
    return rf_pipeline, xgb_pipeline

# ----------------------------------------------------------------------------
# ÖRNEK 2: ANOMALİ TESPİTİ VE MÜŞTERİ SEGMENTASYONU
# ----------------------------------------------------------------------------

def create_customer_data(n_customers=1000):
    """Müşteri segmentasyonu için örnek veri oluşturur"""
    
    # Ana müşteri segmentleri için merkezler
    centers = [
        [5000, 15, 0.3],  # Yüksek harcama, orta sıklık, düşük iade
        [1000, 30, 0.1],  # Düşük harcama, yüksek sıklık, çok düşük iade
        [8000, 5, 0.05],  # Çok yüksek harcama, düşük sıklık, çok düşük iade
        [500, 2, 0.5],    # Çok düşük harcama, çok düşük sıklık, yüksek iade (potansiyel anomali)
        [3000, 12, 0.2]   # Orta harcama, orta sıklık, düşük iade
    ]
    
    # Cluster büyüklükleri (toplam 1000 müşteri)
    sizes = [300, 250, 50, 100, 300]
    
    # Veri özellikler
    features = ['avg_purchase_value', 'purchase_frequency', 'return_rate']
    
    # Her segment için veri oluştur
    segments = []
    segment_labels = []
    
    for i, (center, size) in enumerate(zip(centers, sizes)):
        # Her özellik için rastgele dağılım
        std_devs = [center[0] * 0.2, center[1] * 0.3, center[2] * 0.1]
        
        # Rastgele nokta oluştur
        segment_data = np.random.normal(loc=center, scale=std_devs, size=(size, len(features)))
        
        # Negatif değerleri düzelt
        segment_data = np.abs(segment_data)
        
        # Return rate'i 0-1 arasına sınırla
        segment_data[:, 2] = np.clip(segment_data[:, 2], 0, 1)
        
        segments.append(segment_data)
        segment_labels.extend([i] * size)
    
    # Tüm segmentleri birleştir
    data = np.vstack(segments)
    
    # Müşteri ID'leri oluştur
    customer_ids = [f'CUST_{i:05d}' for i in range(1, n_customers + 1)]
    
    # Diğer özellikler ekle
    loyalty_years = np.random.uniform(0, 10, size=n_customers)
    avg_basket_size = np.random.uniform(1, 15, size=n_customers)
    pct_discount_used = np.random.uniform(0, 0.7, size=n_customers)
    
    # Veri çerçevesi oluştur
    df = pd.DataFrame(data, columns=features)
    df['customer_id'] = customer_ids
    df['loyalty_years'] = loyalty_years
    df['avg_basket_size'] = avg_basket_size
    df['pct_discount_used'] = pct_discount_used
    df['true_segment'] = segment_labels
    
    # İleride analiz için Kümülatif Değer oluştur
    df['customer_value'] = df['avg_purchase_value'] * df['purchase_frequency'] * (1 - df['return_rate']) * (1 + df['loyalty_years'] * 0.1)
    
    return df

def detect_customer_anomalies(df):
    """Müşteri verilerinde anomali tespiti yapar"""
    
    # Anomali tespiti için kullanılacak özellikler
    features = ['avg_purchase_value', 'purchase_frequency', 'return_rate', 
                'loyalty_years', 'avg_basket_size', 'pct_discount_used']
    
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Isolation Forest modeli
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(X_scaled)
    df['anomaly_score'] = iso_forest.score_samples(X_scaled)
    
    # Anomalileri -1, normal gözlemleri 1 olarak işaretler
    # Bunu 0 ve 1 olarak değiştirelim (1: anomali, 0: normal)
    df['anomaly'] = [1 if x == -1 else 0 for x in df['anomaly']]
    
    # Anomalileri görselleştir
    plt.figure(figsize=(12, 8))
    plt.scatter(df[df['anomaly'] == 0]['avg_purchase_value'], 
                df[df['anomaly'] == 0]['purchase_frequency'],
                alpha=0.7, c='blue', s=30, label='Normal')
    plt.scatter(df[df['anomaly'] == 1]['avg_purchase_value'], 
                df[df['anomaly'] == 1]['purchase_frequency'],
                alpha=0.7, c='red', s=50, label='Anomali')
    plt.xlabel('Ortalama Satın Alma Değeri')
    plt.ylabel('Satın Alma Sıklığı')
    plt.title('Müşteri Davranışında Anomaliler')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Anomali skorlarının dağılımı
    plt.figure(figsize=(12, 6))
    sns.histplot(df['anomaly_score'], bins=50, kde=True)
    plt.title('Anomali Skoru Dağılımı')
    plt.xlabel('Anomali Skoru (Düşük değerler anomali gösterir)')
    plt.axvline(df[df['anomaly'] == 1]['anomaly_score'].max(), color='red', 
                linestyle='--', label='Anomali Eşiği')
    plt.legend()
    
    # En anormal müşterilerin detayları
    anomalies = df[df['anomaly'] == 1].sort_values('anomaly_score')
    print(f"Tespit edilen toplam anomali sayısı: {len(anomalies)}")
    print("\nEn anormal 5 müşteri:")
    print(anomalies.head()[['customer_id', 'avg_purchase_value', 'purchase_frequency', 
                            'return_rate', 'customer_value', 'anomaly_score']])
    
    return df

def segment_customers(df, n_clusters=4):
    """K-means ile müşteri segmentasyonu yapar"""
    
    # Segmentasyon için kullanılacak özellikler
    features = ['avg_purchase_value', 'purchase_frequency', 'return_rate', 
                'loyalty_years', 'customer_value']
    
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Optimal küme sayısı için silhouette skoru
    silhouette_scores = []
    range_n_clusters = range(2, 9)
    
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Silhouette skorlarını görselleştir
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, 'o-')
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Silhouette Skoru')
    plt.title('Optimal Küme Sayısı Belirleme')
    plt.grid(True, alpha=0.3)
    
    # Optimal küme sayısı
    best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    print(f"En iyi silhouette skoru {silhouette_scores[np.argmax(silhouette_scores)]:.3f} ile {best_n_clusters} kümedir.")
    
    # Final K-means modeli
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Küme merkezlerini geri dönüştür (orijinal ölçekte)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
    
    print("\nKüme Merkezleri:")
    print(cluster_centers_df)
    
    # Her kümenin büyüklüğü
    print("\nKüme Büyüklükleri:")
    print(df['cluster'].value_counts())
    
    # Kümeleri görselleştir
    plt.figure(figsize=(14, 8))
    
    # Scatter plot
    scatter = plt.scatter(df['avg_purchase_value'], df['purchase_frequency'], 
                          c=df['cluster'], cmap='viridis', alpha=0.6, s=50)
    
    # Küme merkezleri
    plt.scatter(cluster_centers_df['avg_purchase_value'], 
                cluster_centers_df['purchase_frequency'], 
                c='red', s=200, alpha=0.8, marker='X')
    
    plt.xlabel('Ortalama Satın Alma Değeri')
    plt.ylabel('Satın Alma Sıklığı')
    plt.title('Müşteri Segmentasyonu')
    plt.colorbar(scatter, label='Küme')
    plt.grid(True, alpha=0.3)
    
    # Kümelerin özelliklerini görselleştirme
    plt.figure(figsize=(16, 12))
    
    for i, feature in enumerate(features):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x='cluster', y=feature, data=df)
        plt.title(f'Kümelere Göre {feature}')
        plt.xlabel('Küme')
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    
    # RFM (Recency, Frequency, Monetary) benzeri analiz
    plt.figure(figsize=(12, 8))
    
    # 3B görselleştirme
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Her küme için farklı renk
    colors = plt.cm.viridis(np.linspace(0, 1, best_n_clusters))
    
    for i in range(best_n_clusters):
        cluster_data = df[df['cluster'] == i]
        ax.scatter(cluster_data['avg_purchase_value'], 
                   cluster_data['purchase_frequency'],
                   cluster_data['customer_value'],
                   color=colors[i],
                   s=40,
                   alpha=0.6,
                   label=f'Küme {i}')
    
    # Küme merkezleri
    ax.scatter(cluster_centers_df['avg_purchase_value'],
               cluster_centers_df['purchase_frequency'],
               cluster_centers_df['customer_value'],
               color='red',
               s=200,
               alpha=0.8,
               marker='X')
    
    ax.set_xlabel('Ortalama Satın Alma Değeri')
    ax.set_ylabel('Satın Alma Sıklığı')
    ax.set_zlabel('Müşteri Değeri')
    ax.set_title('3B Müşteri Segmentasyonu')
    ax.legend()
    
    return df, kmeans, scaler

# ----------------------------------------------------------------------------
# ÖRNEK 3: VERİ GÖRSELLEŞTIRME VE ETKİLEŞIMLİ DASHBOARD
# ----------------------------------------------------------------------------

def create_interactive_dashboard(sales_df, customer_df):
    """Matplotlib ile düzenli ve profesyonel bir dashboard simülasyonu"""

    # Veri hazırlığı
    sales_monthly = sales_df.set_index('date').resample('M').sum()
    sales_monthly.index = sales_monthly.index.strftime('%Y-%m')

    # Dashboard düzeni
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle('Satış ve Müşteri Analiz Dashboardu', fontsize=20, fontweight='bold')
    grid = plt.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    # 1. Panel: Aylık Satış Trendi
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.plot(sales_monthly.index, sales_monthly['sales'], 'o-', color='royalblue', linewidth=2)
    ax1.set_title('Aylık Satış Trendi', fontsize=14)
    ax1.set_xlabel('Ay')
    ax1.set_ylabel('Toplam Satış')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # 2. Panel: Müşteri Segmentleri
    ax2 = fig.add_subplot(grid[0, 2:])
    segment_sizes = customer_df['cluster'].value_counts().sort_index()
    wedges, texts, autotexts = ax2.pie(segment_sizes,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=plt.cm.viridis(np.linspace(0, 1, len(segment_sizes))),
                                       textprops={'fontsize': 10})
    ax2.set_title('Müşteri Segmentleri Dağılımı', fontsize=14)
    ax2.legend([f'Segment {i}' for i in segment_sizes.index],
               loc='center left',
               bbox_to_anchor=(1.0, 0.5))

    # 3. Panel: Satın Alma Değeri vs Sıklık
    ax3 = fig.add_subplot(grid[1, :2])
    scatter = ax3.scatter(customer_df['avg_purchase_value'],
                          customer_df['purchase_frequency'],
                          c=customer_df['cluster'],
                          cmap='viridis',
                          alpha=0.6,
                          s=50)
    ax3.set_title('Müşteri Satın Alma Davranışı', fontsize=14)
    ax3.set_xlabel('Ortalama Satın Alma Değeri')
    ax3.set_ylabel('Satın Alma Sıklığı')
    ax3.grid(True, alpha=0.3)

    # Renk çubuğu
    cbar = fig.colorbar(scatter, ax=ax3)
    cbar.set_label('Segment')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Başlık ile çakışmayı engelle

if __name__ == "__main__":
    # Örnek verileri oluşturun
    print("Satış verileri oluşturuluyor...")
    sales_data = create_sample_sales_data()
    print(f"Oluşturulan satış veri seti: {sales_data.shape[0]} satır, {sales_data.shape[1]} sütun")
    
    # Zaman serisi analizi
    print("\nZaman serisi analizi yapılıyor...")
    result = analyze_time_series(sales_data)
    
    # Satış tahmini
    print("\nSatış tahmini yapılıyor...")
    forecast = forecast_sales(sales_data)
    
    # ML modeli eğitimi
    print("\nMakine öğrenimi modelleri eğitiliyor...")
    rf_model, xgb_model = train_ml_sales_model(sales_data)
    
    # Müşteri verileri oluştur
    print("\nMüşteri verileri oluşturuluyor...")
    customer_data = create_customer_data()
    print(f"Oluşturulan müşteri veri seti: {customer_data.shape[0]} satır, {customer_data.shape[1]} sütun")
    
    # Anomali tespiti
    print("\nMüşteri anomalileri tespit ediliyor...")
    customer_data_with_anomalies = detect_customer_anomalies(customer_data)
    
    # Müşteri segmentasyonu
    print("\nMüşteri segmentasyonu yapılıyor...")
    segmented_customers, kmeans_model, scaler = segment_customers(customer_data)
    
    # Dashboard oluştur
    print("\nDashboard oluşturuluyor...")
    create_interactive_dashboard(sales_data, segmented_customers)
    
    # Grafikleri göster
    plt.tight_layout()
    plt.show()
    
    print("\nAnaliz tamamlandı!")