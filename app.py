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

with tab4:
    st.header("Kullanım Kılavuzu")
    
    st.subheader("Genel Bakış")
    st.write("Bu uygulama, Python'da geliştirilmiş veri analizi ve yapay zeka fonksiyonlarını kullanıcı dostu bir arayüz üzerinden erişilebilir hale getirmek için tasarlanmıştır.")
    st.markdown("""
    - Zaman Serisi Analizi ve Satış Tahmini
    - Müşteri Segmentasyonu
    - Anomali Tespiti
    """)
    
    st.subheader("Zaman Serisi Analizi")
    st.write("Bu modül, geçmiş satış verilerini analiz ederek gelecekteki satışları tahmin etmek için kullanılır.")
    st.markdown("""
    **Kullanılan Yöntemler:** ARIMA modeli, RandomForest ve XGBoost algoritmaları.
    
    **CSV Formatı:** Dosyanızda 'date' ve 'sales' sütunları bulunmalıdır.
    """)
    
    st.subheader("Müşteri Segmentasyonu")
    st.write("Müşterilerinizi benzer davranış özelliklerine göre gruplara ayırmak için K-means algoritması kullanılır.")
    st.markdown("""
    **CSV Formatı:** Dosyanızda 'customer_id', 'avg_purchase_value', 'purchase_frequency' ve 'return_rate' sütunları bulunmalıdır.
    """)
    
    st.subheader("Anomali Tespiti")
    st.write("Normal müşteri davranışından sapan anormal desenleri tespit etmek için Isolation Forest algoritması kullanılır.")
    st.markdown("""
    **Anomali Eşiği:** Veri setinin yüzde kaçının anormal olarak işaretleneceğini belirler.
    """)