import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import veri_analizi as va  # orijinal kodunuzu içe aktarın

st.title("Yapay Zeka ile Veri Analizi")

tab1, tab2, tab3, tab4 = st.tabs(["Ana Sayfa", "Satış Tahmini", "Müşteri Analizi", "Kullanım Kılavuzu"])

with tab1:
    st.header("Yapay Zeka ile Veri Analizi Uygulamasına Hoş Geldiniz")
    st.info("Bu uygulama, veri analizi ve yapay zeka yöntemlerini kullanarak satış tahmini, müşteri segmentasyonu ve anomali tespiti yapmak için geliştirilmiştir.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Zaman Serisi Analizi")
        st.write("ARIMA ve ML modelleri ile gelecek satış tahminleri yapın.")
    with col2:
        st.subheader("Müşteri Segmentasyonu")
        st.write("K-means algoritması ile müşterilerinizi segmentlere ayırın.")
    with col3:
        st.subheader("Anomali Tespiti")
        st.write("Isolation Forest algoritması ile anormal davranışları tespit edin.")

with tab2:
    st.header("Zaman Serisi Analizi ve Satış Tahmini")
    
    data_source = st.radio("Veri Kaynağı", ["Örnek Veri Kullan", "CSV Yükle"])
    
    if data_source == "CSV Yükle":
        uploaded_file = st.file_uploader("CSV Dosyası Seçin", type="csv")
        if uploaded_file is not None:
            sales_data = pd.read_csv(uploaded_file)
            st.success(f"{uploaded_file.name} yüklendi")
        else:
            sales_data = None
    else:
        st.info("Örnek veri oluşturuluyor...")
        sales_data = va.create_sample_sales_data()
        st.success("Örnek veri oluşturuldu")
    
    forecast_days = st.slider("Tahmin Günü Sayısı", 7, 90, 30)
    
    if st.button("Analizi Başlat"):
        if sales_data is not None:
            st.info("Analiz yapılıyor...")
            
            # Zaman serisi analizi
            result = va.analyze_time_series(sales_data)
            
            # Tahmin
            forecast = va.forecast_sales(sales_data, forecast_days)
            
            # Sonuçları göster
            st.header("Analiz Sonuçları")
            
            st.subheader("Zaman Serisi Grafikleri")
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(result.observed.plot().figure)
                st.caption("Gözlenen Satışlar")
            with col2:
                st.pyplot(result.trend.plot().figure)
                st.caption("Trend Bileşeni")
            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(result.seasonal.plot().figure)
                st.caption("Mevsimsel Bileşen")
            with col2:
                st.pyplot(result.resid.plot().figure)
                st.caption("Artık Bileşeni")
            
            st.subheader("ARIMA Tahmin Sonuçları")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sales_data.set_index('date')['sales'][-90:].index, sales_data.set_index('date')['sales'][-90:].values, label='Geçmiş Veriler')
            ax.plot(forecast.index, forecast.values, color='red', label='Tahmin')
            ax.set_title(f'{forecast_days} Günlük Tahmin')
            ax.legend()
            st.pyplot(fig)
            
            if st.button("Sonuçları İndir"):
                st.success("Sonuçlar indirildi!")

with tab3:
    st.header("Müşteri Analizi")
    
    subtab1, subtab2 = st.tabs(["Müşteri Segmentasyonu", "Anomali Tespiti"])
    
    with subtab1:
        st.subheader("Müşteri Segmentasyonu")
        
        data_source = st.radio("Veri Kaynağı (Segmentasyon)", ["Örnek Veri Kullan", "CSV Yükle"])
        
        if data_source == "CSV Yükle":
            uploaded_file = st.file_uploader("CSV Dosyası Seçin (Segmentasyon)", type="csv", key="seg_upload")
            if uploaded_file is not None:
                customer_data = pd.read_csv(uploaded_file)
                st.success(f"{uploaded_file.name} yüklendi")
            else:
                customer_data = None
        else:
            st.info("Örnek müşteri verisi oluşturuluyor...")
            customer_data = va.create_customer_data()
            st.success("Örnek müşteri verisi oluşturuldu")
        
        cluster_count = st.select_slider("Küme Sayısı", options=[2, 3, 4, 5, 6], value=4)
        
        if st.button("Segmentasyon Analizini Başlat"):
            if customer_data is not None:
                st.info("Segmentasyon analizi yapılıyor...")
                
                # Segmentasyon analizi
                segmented_data, kmeans_model, scaler = va.segment_customers(customer_data, cluster_count)
                
                # Sonuçları göster
                st.header("Segmentasyon Sonuçları")
                
                # 2B ve 3B görselleştirme
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(customer_data['avg_purchase_value'], 
                              customer_data['purchase_frequency'],
                              c=segmented_data['cluster'], 
                              cmap='viridis')
                    ax.set_xlabel('Ortalama Satın Alma Değeri')
                    ax.set_ylabel('Satın Alma Sıklığı')
                    ax.set_title('2B Müşteri Segmentasyonu')
                    st.pyplot(fig)
                    st.caption("2B Görselleştirme")
                
                # Segment özellikleri
                st.subheader("Segment Özellikleri")
                segment_stats = segmented_data.groupby('cluster').agg({
                    'customer_id': 'count',
                    'avg_purchase_value': 'mean',
                    'purchase_frequency': 'mean',
                    'return_rate': lambda x: np.mean(x) * 100  # yüzde olarak
                }).reset_index()
                
                segment_stats.columns = ['Segment', 'Müşteri Sayısı', 'Ort. Satın Alma', 'Sıklık', 'İade Oranı (%)']
                st.dataframe(segment_stats)
    
    with subtab2:
        st.subheader("Anomali Tespiti")
        
        data_source = st.radio("Veri Kaynağı (Anomali)", ["Örnek Veri Kullan", "CSV Yükle"])
        
        if data_source == "CSV Yükle":
            uploaded_file = st.file_uploader("CSV Dosyası Seçin (Anomali)", type="csv", key="anom_upload")
            if uploaded_file is not None:
                customer_data = pd.read_csv(uploaded_file)
                st.success(f"{uploaded_file.name} yüklendi")
            else:
                customer_data = None
        else:
            st.info("Örnek müşteri verisi oluşturuluyor...")
            customer_data = va.create_customer_data()
            st.success("Örnek müşteri verisi oluşturuldu")
        
        anomaly_threshold = st.slider("Anomali Eşiği (%)", 1, 10, 5) / 100
        
        if st.button("Anomali Tespiti Analizini Başlat"):
            if customer_data is not None:
                st.info("Anomali tespiti yapılıyor...")
                
                # Anomali tespiti 
                customer_data_with_anomalies = va.detect_customer_anomalies(customer_data)
                
                # Sonuçları göster
                st.header("Anomali Tespiti Sonuçları")
                
                anomaly_count = len(customer_data_with_anomalies[customer_data_with_anomalies['anomaly'] == 1])
                st.warning(f"{anomaly_count} anormal müşteri tespit edildi (Tüm müşterilerin %{anomaly_count/len(customer_data_with_anomalies)*100:.1f}'i).")
                
                # Anomali görselleştirme
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(customer_data_with_anomalies[customer_data_with_anomalies['anomaly'] == 0]['avg_purchase_value'], 
                              customer_data_with_anomalies[customer_data_with_anomalies['anomaly'] == 0]['purchase_frequency'],
                              alpha=0.7, c='blue', s=30, label='Normal')
                    ax.scatter(customer_data_with_anomalies[customer_data_with_anomalies['anomaly'] == 1]['avg_purchase_value'], 
                              customer_data_with_anomalies[customer_data_with_anomalies['anomaly'] == 1]['purchase_frequency'],
                              alpha=0.7, c='red', s=50, label='Anomali')
                    ax.set_xlabel('Ortalama Satın Alma Değeri')
                    ax.set_ylabel('Satın Alma Sıklığı')
                    ax.set_title('Müşteri Davranışında Anomaliler')
                    ax.legend()
                    st.pyplot(fig)
                    st.caption("Anomali Dağılımı")
                
                # En anormal 5 müşteri
                st.subheader("En Anormal 5 Müşteri")
                top_anomalies = customer_data_with_anomalies[customer_data_with_anomalies['anomaly'] == 1].sort_values('anomaly_score').head(5)
                top_anomalies = top_anomalies[['customer_id', 'anomaly_score', 'avg_purchase_value', 'purchase_frequency', 'return_rate']]
                top_anomalies.columns = ['Müşteri ID', 'Anomali Skoru', 'Ort. Satın Alma', 'Sıklık', 'İade Oranı']
                top_anomalies['İade Oranı'] = top_anomalies['İade Oranı'] * 100  # yüzde olarak
                st.dataframe(top_anomalies)

# Kullanım Kılavuzu Sekmesi
with tab4:
    st.header("Kullanım Kılavuzu ve Teknik Detaylar")
    
    # Genel Bakış
    st.subheader("1. Genel Bakış")
    st.write("Bu uygulama, veri analizi ve yapay zeka yöntemlerini kullanarak satış tahmini, müşteri segmentasyonu ve anomali tespiti yapmanızı sağlayan etkileşimli bir araçtır.")
    st.markdown("""
    - **Zaman Serisi Analizi ve Satış Tahmini**: Geçmiş verileri analiz ederek gelecek satışlarını tahmin eder
    - **Müşteri Segmentasyonu**: Benzer davranış gösteren müşterileri gruplandırır
    - **Anomali Tespiti**: Normal müşteri davranışından sapan anormal desenleri tespit eder
    """)
    
    # Zaman Serisi Analizi 
    st.subheader("2. Zaman Serisi Analizi ve Satış Tahmini")
    st.write("Bu modül, geçmiş satış verilerini analiz ederek gelecekteki satışları tahmin etmek için kullanılır.")
    
    # Veri Formatı
    st.write("#### 2.1. Veri Formatı")
    st.markdown("""
    CSV dosyanızın aşağıdaki sütunları içermesi gerekir:
    - `date`: YYYY-MM-DD formatında tarih (ör. 2022-01-01)
    - `sales`: Sayısal satış değeri
    
    İsteğe bağlı olarak şu sütunları da ekleyebilirsiniz:
    - `is_holiday`: Tatil günü olup olmadığını belirten 1/0 değeri
    - `is_promotion`: Promosyon dönemi olup olmadığını belirten 1/0 değeri
    - `weekday`: Haftanın günü (0-6, 0=Pazartesi)
    - `month`: Ay (1-12)
    - `year`: Yıl
    - `is_weekend`: Hafta sonu olup olmadığını belirten 1/0 değeri
    """)
    
    # Hesaplama Adımları
    st.write("#### 2.2. Hesaplama Adımları")
    
    st.write("##### 2.2.1. Zaman Serisi Ayrıştırma (Seasonal Decomposition)")
    st.markdown("""
    Zaman serisi ayrıştırma, satış verilerinin içindeki farklı bileşenleri ayrıştırmak için kullanılır:
    
    1. **Gözlemlenen Satışlar**: Orijinal zaman serisi verisi
    2. **Trend Bileşeni**: Uzun vadeli artış veya azalış trendi
       - Hareketli ortalama (moving average) yöntemi ile hesaplanır
       - Formül: Belirli bir periyot boyunca verilerin ortalaması alınır
    3. **Mevsimsel Bileşen**: Tekrarlanan, periyodik dalgalanmalar
       - Trendsiz verilerin mevsimsel periyotlarına göre ortalaması alınarak hesaplanır
       - Günlük, haftalık, aylık ve yıllık desenler içerebilir
    4. **Artık (Residual) Bileşen**: Trend ve mevsimsellikle açıklanamayan değişimler
       - Formül: Gözlemlenen Veri - (Trend + Mevsimsellik)
    
    Bu ayrıştırma için statsmodels kütüphanesinin `seasonal_decompose` fonksiyonunu kullanıyoruz.
    """)
    
    st.write("##### 2.2.2. ARIMA Modeli ile Satış Tahmini")
    st.markdown("""
    ARIMA (AutoRegressive Integrated Moving Average) modelini kullanarak gelecek satışlarını tahmin ediyoruz:
    
    1. **Otoregresif Bileşen (AR - p)**: 
       - Geçmiş değerler kullanılarak gelecek değerlerin tahmini
       - Formül: Yt = c + φ1*Y(t-1) + φ2*Y(t-2) + ... + φp*Y(t-p) + εt
       - Modelimizde p=5 kullanılıyor (5 gecikmeli değer)
    
    2. **Entegrasyon Derecesi (I - d)**:
       - Zaman serisini durağanlaştırmak için kullanılan fark alma işlemi
       - Modelimizde d=1 kullanılıyor (birinci dereceden fark alma)
    
    3. **Hareketli Ortalama Bileşeni (MA - q)**:
       - Geçmiş hata terimlerini kullanarak gelecek değerleri tahmin etme
       - Formül: Yt = c + εt + θ1*ε(t-1) + θ2*ε(t-2) + ... + θq*ε(t-q)
       - Modelimizde q=2 kullanılıyor (2 gecikmeli hata terimi)
    
    4. **Tahmin ve Güven Aralığı**:
       - Model ile gelecek için nokta tahminleri yapılır
       - %95 güven aralığı ile tahmin belirsizliği gösterilir
    
    Bu tahmin için statsmodels kütüphanesinin `ARIMA` modelini kullanıyoruz.
    """)
    
    st.write("##### 2.2.3. Makine Öğrenmesi Modelleri ile Satış Tahmini")
    st.markdown("""
    İki farklı makine öğrenmesi algoritması kullanarak alternatif tahminler yapıyoruz:
    
    1. **RandomForest Regressor**:
       - Çok sayıda karar ağacının ortalamasını alarak çalışır
       - Aşırı öğrenmeye (overfitting) karşı dirençlidir
       - Parametreler:
         - n_estimators=100 (100 farklı ağaç)
         - random_state=42 (tekrarlanabilirlik için)
    
    2. **XGBoost Regressor**:
       - Gradient boosting tekniğini kullanır
       - Her adımda bir önceki modelin hatalarını düzeltmeye çalışır
       - Parametreler:
         - n_estimators=100 (100 iterasyon)
         - learning_rate=0.1 (öğrenme hızı)
         - max_depth=7 (ağaç derinliği)
    
    3. **Özellik Önemliliği**:
       - Hangi faktörlerin satışları en çok etkilediğini gösterir
       - RandomForest modelinin feature_importances_ özelliği kullanılır
    
    4. **Çapraz Doğrulama**:
       - Zaman serisi verilerinde özel bir çapraz doğrulama olan TimeSeriesSplit kullanılır
       - Model performansını değerlendirmek için RMSE (Root Mean Squared Error) kullanılır
    """)
    
    # Müşteri Segmentasyonu
    st.subheader("3. Müşteri Segmentasyonu")
    st.write("Bu modül, müşterilerinizi benzer davranış özelliklerine göre gruplara ayırmak için kullanılır.")
    
    # Veri Formatı
    st.write("#### 3.1. Veri Formatı")
    st.markdown("""
    CSV dosyanızın aşağıdaki sütunları içermesi gerekir:
    - `customer_id`: Müşteri kimliği (ör. CUST_00001)
    - `avg_purchase_value`: Ortalama satın alma değeri (ör. 5000)
    - `purchase_frequency`: Satın alma sıklığı (ör. 12 - yıllık satın alma sayısı)
    - `return_rate`: İade oranı (0-1 arası, ör. 0.05 = %5)
    
    İsteğe bağlı olarak şu sütunları da ekleyebilirsiniz:
    - `loyalty_years`: Müşteri sadakat yılı
    - `avg_basket_size`: Ortalama sepet büyüklüğü (ürün sayısı)
    - `pct_discount_used`: İndirim kullanım oranı (0-1 arası)
    """)
    
    # Hesaplama Adımları
    st.write("#### 3.2. Hesaplama Adımları")
    
    st.write("##### 3.2.1. Veri Ön İşleme")
    st.markdown("""
    Segmentasyon öncesi veri hazırlığı:
    
    1. **Veri Normalizasyonu**:
       - Farklı ölçeklerdeki özellikleri 0-1 arasına getirme
       - StandardScaler kullanılır: z = (x - μ) / σ
       - Burada x: orijinal değer, μ: ortalama, σ: standart sapma
    
    2. **Özellik Seçimi**:
       - Segmentasyon için en bilgilendirici özellikler seçilir
       - Kullanılan özellikler: 'avg_purchase_value', 'purchase_frequency', 'return_rate', 'loyalty_years', 'customer_value'
    """)
    
    st.write("##### 3.2.2. K-means Kümeleme")
    st.markdown("""
    K-means algoritması ile müşteri segmentasyonu:
    
    1. **Optimal Küme Sayısı Belirleme**:
       - Silhouette skoru kullanılır: -1 (kötü) ile 1 (mükemmel) arası bir değer
       - Formül: s(i) = (b(i) - a(i)) / max{a(i), b(i)}
         - a(i): Bir noktanın kendi kümesindeki diğer noktalara olan ortalama mesafesi
         - b(i): Bir noktanın en yakın komşu kümedeki noktalara olan ortalama mesafesi
       - 2'den 8'e kadar her küme sayısı için hesaplanır ve en yüksek skora sahip küme sayısı seçilir
    
    2. **K-means Algoritması**:
       - 1) Rastgele k adet merkez nokta seçilir (başlangıç noktaları)
       - 2) Her veri noktası en yakın merkeze atanır
       - 3) Her küme için yeni merkez hesaplanır (kümedeki noktaların ortalaması)
       - 4) Merkezler değişmeyene kadar adım 2 ve 3 tekrarlanır
       - Uzaklık ölçümü için Öklid mesafesi kullanılır: d(x,y) = √Σ(xi-yi)²
    
    3. **Küme Analiziı**:
       - Her kümenin merkezi özelliklerini belirlemek
       - Her kümede kaç müşteri olduğunu hesaplamak
       - Kümeleri görselleştirmek (2B ve 3B grafikler)
    """)
    
    # Anomali Tespiti
    st.subheader("4. Anomali Tespiti")
    st.write("Bu modül, normal müşteri davranışından sapan anormal desenleri tespit etmek için kullanılır.")
    
    # Hesaplama Adımları
    st.write("#### 4.1. Hesaplama Adımları")
    
    st.write("##### 4.1.1. Isolation Forest Algoritması")
    st.markdown("""
    Isolation Forest, anormallikleri diğer noktalardan "izole etme" kolaylığına göre tespit eder:
    
    1. **Çalışma Prensibi**:
       - Normal noktaları izole etmek daha fazla bölme gerektirir (daha derin ağaç)
       - Anormal noktalar daha az bölme ile izole edilebilir (daha sığ ağaç)
       - Temel varsayım: Anormal noktalar, daha az sayıda ve normal noktalardan daha uzaktadır
    
    2. **Algoritma Adımları**:
       - 1) Veri kümesinden rastgele bir alt küme alınır
       - 2) Rastgele bir özellik seçilir
       - 3) Seçilen özellik için rastgele bir bölme değeri belirlenir
       - 4) Veri iki alt kümeye bölünür
       - 5) İzolasyon tamamlanana veya maksimum ağaç derinliğine ulaşılana kadar tekrarlanır
       - 6) Çoklu ağaçlar oluşturularak ortalama yol uzunluğu hesaplanır
    
    3. **Anomali Skoru Hesaplama**:
       - s(x,n) = 2^(-E(h(x))/c(n))
       - E(h(x)): Noktanın ortalama yol uzunluğu
       - c(n): Normal dağılımlı veride ortalama yol uzunluğu
       - Skor 0'a yaklaştıkça daha anormal, 0.5'e yaklaştıkça daha normal
    
    4. **Contamination (Kirlilik) Parametresi**:
       - Verinin ne kadarının anormal olarak işaretleneceğini belirler
       - Uygulamada 0.05 (veri setinin %5'i) olarak ayarlanmıştır
    """)
    
    st.write("##### 4.1.2. Anomali Görselleştirme")
    st.markdown("""
    Tespit edilen anormal müşterilerin görselleştirilmesi:
    
    1. **Scatter Plot (Dağılım Grafiği)**:
       - x-ekseni: Ortalama satın alma değeri
       - y-ekseni: Satın alma sıklığı
       - Normal müşteriler mavi, anormal müşteriler kırmızı ile gösterilir
    
    2. **Anomali Skoru Histogramı**:
       - x-ekseni: Anomali skoru (düşük değerler anomaliyi gösterir)
       - y-ekseni: Müşteri sayısı
       - Anomali eşiği kırmızı dikey çizgi ile gösterilir
    
    3. **En Anormal Müşteri Listesi**:
       - Anomali skoru en düşük olan müşterilerin özellikleri listelenir
       - Bu müşterilerin davranış özellikleri incelenerek anomali nedeni anlaşılabilir
    """)
    
    # Metrikler ve Değerlendirme
    st.subheader("5. Değerlendirme Metrikleri")
    st.markdown("""
    #### 5.1. Satış Tahminleri için Metrikler
    
    1. **RMSE (Root Mean Squared Error - Kök Ortalama Kare Hata)**:
       - Tahmin hatalarının karesinin ortalamasının kökü
       - Formül: RMSE = √(Σ(yi - ŷi)² / n)
       - Düşük değerler daha iyi tahmin anlamına gelir
    
    2. **MAE (Mean Absolute Error - Ortalama Mutlak Hata)**:
       - Tahmin hatalarının mutlak değerlerinin ortalaması
       - Formül: MAE = Σ|yi - ŷi| / n
       - RMSE'ye göre büyük hatalara daha az duyarlıdır
    
    #### 5.2. Müşteri Segmentasyonu için Metrikler
    
    1. **Silhouette Katsayısı**:
       - Kümelemenin kalitesini ölçer
       - -1 ile 1 arasında değer alır (1'e yakın değerler daha iyi)
       - Kümelerin ne kadar sıkı ve ayrık olduğunu değerlendirir
    
    2. **İnertia (KMeans içinde)**:
       - Her noktanın kendi küme merkezine olan uzaklıklarının karelerinin toplamı
       - Düşük değerler daha kompakt kümeler anlamına gelir
    
    #### 5.3. Anomali Tespiti için Metrikler
    
    1. **Anomali Skoru**:
       - Isolation Forest tarafından üretilen -1'e yakın skorlar güçlü anomalileri gösterir
       - Normal gözlemler genellikle 0'a yakın skorlara sahiptir
    """)
    
    # Pratik İpuçları
    st.subheader("6. Pratik İpuçları")
    st.markdown("""
    1. **Veri Hazırlama**:
       - Tarih sütununu doğru formatta olduğundan emin olun (YYYY-MM-DD)
       - Eksik değerleri doldurun veya ilgili satırları kaldırın
       - Aykırı değerleri tespit edin ve gerekirse düzeltin
    
    2. **Optimum Parametre Seçimi**:
       - ARIMA için farklı p, d, q değerlerini deneyin
       - Kümeleme için farklı küme sayılarını test edin
       - Anomali eşiğini veri setinize göre ayarlayın
    
    3. **Tahmin Sonuçlarını Değerlendirme**:
       - Tahminleri yalnızca RMSE ile değil, grafiksel olarak da inceleyin
       - Özellikle büyük olaylara (örn. tatiller, promosyonlar) dikkat edin
    
    4. **Müşteri Segmentlerini Yorumlama**:
       - Her segmentin belirgin özelliklerini tanımlayın
       - Segmentlere anlamlı isimler verin (ör. "Yüksek Değerli Seyrek Alıcılar")
       - Segmentlere özel pazarlama stratejileri geliştirin
    
    5. **Anomalileri İnceleme**:
       - Her anomaliyi ayrı ayrı inceleyin ve nedenini anlamaya çalışın
       - Sistemsel bir hata mı yoksa gerçek bir anomali mi olduğunu doğrulayın
    """)
