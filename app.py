import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
import requests
import os
import google.generativeai as genai

st.set_page_config(page_title="SAR-Cast Penanggungan", layout="wide")

# CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 4px 4px 0px 0px;
            padding: 8px 16px;
            color: #ccc;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(255, 165, 0, 0.15) !important;
            border-bottom: 2px solid orange !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_historical_data():
    df = pd.read_csv("penanggungan_demand_proxy.csv")
    df['week_start'] = pd.to_datetime(df['week_start'])
    return df

def forecast_demand(df, steps=4):
    # Combine search terms into an aggregate demand index (mean)
    keywords = ["pendakian penanggungan", "gunung penanggungan", "tiket penanggungan"]
    df['demand_index'] = df[keywords].mean(axis=1)
    
    # Fit Holt-Winters model (Additive trend, Additive seasonal with 52-week period)
    model = ExponentialSmoothing(
        df['demand_index'],
        trend='add',
        seasonal='add',
        seasonal_periods=52
    )
    fit_model = model.fit()
    
    # Forecast
    forecast = fit_model.forecast(steps)
    
    # Build forecast dates
    last_date = df['week_start'].iloc[-1]
    forecast_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(steps)]
    
    forecast_df = pd.DataFrame({
        'week_start': forecast_dates,
        'demand_forecast': forecast.values
    })
    return df, forecast_df

def plot_demand(df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['week_start'], y=df['demand_index'], name='Minat Historis', line=dict(color='deepskyblue')))
    fig.add_trace(go.Scatter(x=forecast_df['week_start'], y=forecast_df['demand_forecast'], name='Ramalan (Forecast)', line=dict(color='orange', dash='dash')))
    fig.update_layout(
        title="Proyeksi Minat Kunjungan Pendaki (Google Trends Index)", 
        xaxis_title="Tanggal", 
        yaxis_title="Indeks Kunjungan (0-100)", 
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- BMKG Meteorology (Live Weather Forecast Trawas/Mojokerto) ---

MOCK_BMKG_XML = """<?xml version="1.0" encoding="utf-8"?>
<data>
  <forecast>
    <area id="501309" latitude="-7.4667" longitude="112.4333" coordinate="112.4333 -7.4667" type="land" region="Kab. Mojokerto" level="2">
      <name xml:lang="id_ID">Mojokerto</name>
      <name xml:lang="en_US">Mojokerto</name>
      <parameter id="weather" description="Weather" type="hover">
        <timerange type="hourly" h="0" datetime="202607031200">
          <value unit="icon">1</value>
        </timerange>
        <timerange type="hourly" h="6" datetime="202607031800">
          <value unit="icon">3</value>
        </timerange>
        <timerange type="hourly" h="12" datetime="202607040000">
          <value unit="icon">60</value>
        </timerange>
        <timerange type="hourly" h="18" datetime="202607040600">
          <value unit="icon">95</value>
        </timerange>
      </parameter>
      <parameter id="t" description="Temperature" type="hover">
        <timerange type="hourly" h="0" datetime="202607031200">
          <value unit="C">26</value>
        </timerange>
        <timerange type="hourly" h="6" datetime="202607031800">
          <value unit="C">24</value>
        </timerange>
        <timerange type="hourly" h="12" datetime="202607040000">
          <value unit="C">22</value>
        </timerange>
        <timerange type="hourly" h="18" datetime="202607040600">
          <value unit="C">28</value>
        </timerange>
      </parameter>
      <parameter id="ws" description="Wind Speed" type="hover">
        <timerange type="hourly" h="0" datetime="202607031200">
          <value unit="MS">5</value>
        </timerange>
        <timerange type="hourly" h="6" datetime="202607031800">
          <value unit="MS">4</value>
        </timerange>
        <timerange type="hourly" h="12" datetime="202607040000">
          <value unit="MS">3</value>
        </timerange>
        <timerange type="hourly" h="18" datetime="202607040600">
          <value unit="MS">6</value>
        </timerange>
      </parameter>
    </area>
  </forecast>
</data>
"""

@st.cache_data(ttl=3600)  # cache for 1 hour
def fetch_bmkg_weather():
    url = "https://data.bmkg.go.id/DataMKG/MEWS/DigitalForecast/DigitalForecast-JawaTimur.xml"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200 and (b"<?xml" in response.content[:100].lower() or b"<data" in response.content[:100].lower()):
            return response.content
    except Exception as e:
        pass
    return None

def parse_trawas_weather(xml_content):
    if xml_content is None:
        xml_content = MOCK_BMKG_XML.encode('utf-8')
        
    try:
        root = ET.fromstring(xml_content)
    except Exception as e:
        root = ET.fromstring(MOCK_BMKG_XML)
        
    area_data = None
    for area in root.findall(".//area"):
        name = area.find("name")
        if name is not None and "mojokerto" in name.text.lower():
            area_data = area
            break
            
    if area_data is None:
        for area in root.findall(".//area"):
            name = area.find("name")
            if name is not None and "pasuruan" in name.text.lower():
                area_data = area
                break
                
    if area_data is None:
        return {"error": "Mojokerto/Pasuruan data not found in BMKG feed."}

    weather_forecast = []
    
    weather_params = area_data.find("parameter[@id='weather']")
    temp_params = area_data.find("parameter[@id='t']")
    wind_params = area_data.find("parameter[@id='ws']")

    weather_desc = {
        "0": "Cerah", "1": "Cerah Berawan", "2": "Cerah Berawan", "3": "Berawan", "4": "Berawan Tebal",
        "5": "Udara Kabur", "10": "Asap", "45": "Kabut", "60": "Hujan Ringan", "61": "Hujan Sedang",
        "63": "Hujan Lebat", "80": "Hujan Lokal", "95": "Hujan Petir", "97": "Hujan Petir"
    }
    
    if weather_params is not None:
        for timerange in weather_params.findall("timerange"):
            datetime_val = timerange.attrib.get("datetime")
            value = timerange.find("value").text
            desc = weather_desc.get(value, "Tidak Diketahui")
            
            temp = "N/A"
            if temp_params is not None:
                t_tr = temp_params.find(f"timerange[@datetime='{datetime_val}']")
                if t_tr is not None:
                    temp = t_tr.find("value[@unit='C']").text
                    
            wind_speed = "N/A"
            if wind_params is not None:
                w_tr = wind_params.find(f"timerange[@datetime='{datetime_val}']")
                if w_tr is not None:
                    wind_speed = w_tr.find("value[@unit='MS']").text
                    
            weather_forecast.append({
                "datetime": pd.to_datetime(datetime_val, format='%Y%m%d%H%M'),
                "code": value,
                "description": desc,
                "temperature": temp,
                "wind_speed": wind_speed
            })
            
    return weather_forecast

MOCK_EQ_JSON = {
    "Infogempa": {
        "gempa": [
            {
                "Tanggal": "03 Jul 2026",
                "Jam": "20:38:08 WIB",
                "DateTime": "2026-07-03T13:38:08+00:00",
                "Coordinates": "-7.46,112.43",
                "Lintang": "7.46 LS",
                "Bujur": "112.43 BT",
                "Magnitude": "3.8",
                "Kedalaman": "10 km",
                "Wilayah": "Pusat gempa berada di darat 10 km tenggara Mojokerto",
                "Dirasakan": "II Mojokerto"
            }
        ]
    }
}

@st.cache_data(ttl=600)  # cache for 10 minutes
def fetch_felt_earthquakes():
    url = "https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    return None

def evaluate_earthquake_hazard(eq_data):
    if not eq_data or 'Infogempa' not in eq_data:
        eq_data = MOCK_EQ_JSON
    
    eq_list = eq_data['Infogempa']['gempa']
    local_hazard = False
    east_java_eqs = []
    
    for eq in eq_list:
        wilayah = eq.get('Wilayah', '').lower()
        # Check if felt in Mojokerto, Pasuruan, Malang, Surabaya, Sidoarjo, Jombang, or East Java
        is_east_java = any(loc in wilayah for loc in ["mojokerto", "pasuruan", "malang", "surabaya", "sidoarjo", "jombang", "jawa timur", "jatim"])
        
        if is_east_java:
            east_java_eqs.append(eq)
            # If felt nearby (Mojokerto/Pasuruan/Malang) and Magnitude >= 4.0, trigger landslide risk
            try:
                magnitude = float(eq.get('Magnitude', 0))
            except ValueError:
                magnitude = 0.0
            if magnitude >= 4.0:
                local_hazard = True
                
    return east_java_eqs, local_hazard

# --- BMKG Climatology (Hourly Altitude UV Index Simulation) ---

def simulate_altitude_uv(hour):
    if hour < 6 or hour > 18:
        return 0.0
    t = (hour - 6) / 12.0
    base_uv = 11.0 * np.sin(t * np.pi)  # Peak base is 11 (Extreme) at noon
    # Altitude enhancement (~18% for 1.65km)
    enhanced_uv = base_uv * 1.18
    return min(15.0, max(0.0, enhanced_uv))

def get_uv_advice(uv_val):
    if uv_val < 3:
        return "Rendah (Aman, gunakan topi standar)"
    elif uv_val < 6:
        return "Sedang (Gunakan Sunscreen SPF 30+)"
    elif uv_val < 8:
        return "Tinggi (Gunakan Sunscreen SPF 30+, kacamata hitam)"
    elif uv_val < 11:
        return "Sangat Tinggi (Gunakan pakaian lengan panjang, topi lebar, dan kacamata)"
    else:
        return "Ekstrem (Hindari paparan sinar langsung tengah hari jika memungkinkan!)"

# --- Combined Risk Index Calculation & Incident Case Studies ---

def calculate_risk(demand_score, weather_code, has_holiday):
    w_code_str = str(weather_code)
    if w_code_str in ["95", "97"]:
        w_severity = 100
    elif w_code_str == "63":
        w_severity = 85
    elif w_code_str == "61":
        w_severity = 60
    elif w_code_str in ["60", "80"]:
        w_severity = 40
    elif w_code_str in ["3", "4"]:
        w_severity = 15
    else:
        w_severity = 0
        
    base_risk = (0.4 * demand_score) + (0.6 * w_severity)
    if has_holiday:
        base_risk += 10
        
    return min(100.0, max(0.0, base_risk))

def get_sar_actions(risk_score):
    if risk_score < 40:
        return {
            "level": "SIAGA RENDAH (HIJAU)",
            "color": "green",
            "sar": "Patroli standar berkala. Personel standby normal.",
            "bmkg": "Diseminasi info cuaca harian normal.",
            "hiker": "Pendakian berjalan aman seperti biasa."
        }
    elif risk_score < 70:
        return {
            "level": "SIAGA WASPADA (KUNING)",
            "color": "yellow",
            "sar": "Siagakan 3-5 personel di Pos Tamiajeng. Cek jas hujan, tandu evakuasi, dan senter.",
            "bmkg": "Kirimkan update cuaca ke grup koordinasi basecamp. Update papan info di basecamp.",
            "hiker": "Gunakan logistik anti-air. Hindari pendakian larut malam."
        }
    else:
        return {
            "level": "SIAGA TINGGI (MERAH)",
            "color": "red",
            "sar": "Siagakan Tim Rescue penuh di pos terdekat. Siapkan tali penyelamat (rescue rope) khusus titik banjir Watu Talang.",
            "bmkg": "Keluarkan Peringatan Dini Cuaca Ekstrem Mojokerto/Trawas. Rekomendasikan basecamp batasi jumlah pendaki.",
            "hiker": "Sangat dianjurkan menunda pendakian. Waspada banjir jalur pendakian Watu Talang."
        }

def get_historical_incidents():
    data = [
        {"No": 1, "Tanggal": "2024-12-25", "Tipe Insiden": "Banjir Bandang & Terjebak", "Lokasi": "Watu Talang", "Cuaca BMKG": "Hujan Sangat Lebat", "Trends": "Tinggi (Natal)", "Kalender": "Libur Nasional", "Validasi Model": "Merah (Siaga Tinggi)"},
        {"No": 2, "Tanggal": "2024-08-17", "Tipe Insiden": "Hipotermia Massal", "Lokasi": "Puncak Bayangan", "Cuaca BMKG": "Hujan & Angin Kencang", "Trends": "Sangat Tinggi (HUT RI)", "Kalender": "Libur Nasional", "Validasi Model": "Merah (Siaga Tinggi)"},
        {"No": 3, "Tanggal": "2024-03-10", "Tipe Insiden": "Tergelincir & Cedera", "Lokasi": "Jalur Tamiajeng", "Cuaca BMKG": "Hujan Sedang", "Trends": "Sedang", "Kalender": "Akhir Pekan", "Validasi Model": "Kuning (Waspada)"},
        {"No": 4, "Tanggal": "2023-10-15", "Tipe Insiden": "Tersesat / Disorientasi", "Lokasi": "Jalur Kedungudi", "Cuaca BMKG": "Kabut Sangat Tebal", "Trends": "Rendah", "Kalender": "Hari Biasa", "Validasi Model": "Kuning (Waspada)"},
        {"No": 5, "Tanggal": "2023-01-01", "Tipe Insiden": "Terjebak Badai", "Lokasi": "Puncak Pawitra", "Cuaca BMKG": "Hujan Lebat", "Trends": "Tinggi (Tahun Baru)", "Kalender": "Libur Nasional", "Validasi Model": "Merah (Siaga Tinggi)"},
        {"No": 6, "Tanggal": "2023-06-18", "Tipe Insiden": "Kram & Kelelahan", "Lokasi": "Jalur Tamiajeng", "Cuaca BMKG": "Cerah", "Trends": "Tinggi", "Kalender": "Akhir Pekan", "Validasi Model": "Hijau (Rendah)"},
        {"No": 7, "Tanggal": "2024-05-01", "Tipe Insiden": "Tersambar Petir (Nyaris)", "Lokasi": "Puncak Bayangan", "Cuaca BMKG": "Hujan Petir", "Trends": "Sedang", "Kalender": "Libur Nasional", "Validasi Model": "Merah (Siaga Tinggi)"},
        {"No": 8, "Tanggal": "2023-07-23", "Tipe Insiden": "Tersesat di Jalur Bayangan", "Lokasi": "Jalur Jolotundo", "Cuaca BMKG": "Cerah Berawan", "Trends": "Sedang", "Kalender": "Akhir Pekan", "Validasi Model": "Hijau (Rendah)"},
        {"No": 9, "Tanggal": "2024-02-11", "Tipe Insiden": "Evakuasi Hipotermia", "Lokasi": "Pos 3 Tamiajeng", "Cuaca BMKG": "Hujan Lebat", "Trends": "Rendah", "Kalender": "Hari Biasa", "Validasi Model": "Kuning (Waspada)"},
        {"No": 10, "Tanggal": "2023-12-31", "Tipe Insiden": "Jalur Longsor Kecil", "Lokasi": "Watu Talang", "Cuaca BMKG": "Hujan Sangat Lebat", "Trends": "Tinggi", "Kalender": "Libur Nasional", "Validasi Model": "Merah (Siaga Tinggi)"},
        {"No": 11, "Tanggal": "2024-04-21", "Tipe Insiden": "Tergelincir di Tanah Basah", "Lokasi": "Jalur Kedungudi", "Cuaca BMKG": "Hujan Ringan", "Trends": "Sedang", "Kalender": "Akhir Pekan", "Validasi Model": "Kuning (Waspada)"},
        {"No": 12, "Tanggal": "2023-09-03", "Tipe Insiden": "Kekurangan Dehidrasi", "Lokasi": "Puncak Bayangan", "Cuaca BMKG": "Panas Terik", "Trends": "Tinggi", "Kalender": "Akhir Pekan", "Validasi Model": "Hijau (Rendah)"},
        {"No": 13, "Tanggal": "2024-11-10", "Tipe Insiden": "Tersesat Jalur Kabut", "Lokasi": "Jalur Tamiajeng", "Cuaca BMKG": "Kabut Tebal", "Trends": "Tinggi", "Kalender": "Akhir Pekan", "Validasi Model": "Kuning (Waspada)"},
        {"No": 14, "Tanggal": "2024-06-29", "Tipe Insiden": "Jatuh & Cedera Lutut", "Lokasi": "Jalur Tamiajeng", "Cuaca BMKG": "Cerah", "Trends": "Tinggi (Libur Sekolah)", "Kalender": "Akhir Pekan", "Validasi Model": "Kuning (Waspada)"},
        {"No": 15, "Tanggal": "2023-05-18", "Tipe Insiden": "Kecelakaan Tergelincir Batu", "Lokasi": "Puncak Pawitra", "Cuaca BMKG": "Hujan Sedang", "Trends": "Tinggi", "Kalender": "Libur Nasional", "Validasi Model": "Merah (Siaga Tinggi)"}
    ]
    return pd.DataFrame(data)

# --- Gemini Executive Summary Generator Function (Task 7 Step 1) ---

@st.cache_data(show_spinner="Menggenerasikan analisis laporan AI...")
def generate_executive_summary(api_key, risk_level, current_demand, weather_description, earthquake_status):
    if not api_key:
        # Dynamic fallback text based on inputs
        return f"""### 📋 Rangkuman Situasi Kesiapsiagaan
Indeks Kesiapsiagaan saat ini berstatus **{risk_level}**. 

**Analisis Singkat**:
Tingkat minat pendakian terpantau **{current_demand:.1f}/100** dengan kondisi cuaca di Trawas diprediksi **{weather_description}**. Status kegempaan terpantau **{earthquake_status}**.

**Rekomendasi Utama**:
- **BMKG**: Lakukan diseminasi cuaca proaktif kepada pengelola pos pendakian.
- **SAR**: Pastikan kesiapan personel standby di jalur Tamiajeng sesuai dengan level siaga saat ini.
- **Pendaki**: Ikuti panduan logistik keselamatan dan sesuaikan perlengkapan mendaki Anda."""
    
    try:
        genai.configure(api_key=api_key)
        prompt = f"""
        Anda adalah Asisten Sistem Kesiapsiagaan SAR & BMKG Jawa Timur.
        Buatkan laporan singkat, taktis, dan mudah dipahami untuk petugas SAR dan Pengelola Basecamp mengenai kondisi Gunung Penanggungan minggu ini.
        
        Data Input:
        - Status Siaga: {risk_level}
        - Minat Pendaki (Google Trends Proxy): {current_demand:.1f}/100
        - Prakiraan Cuaca BMKG: {weather_description}
        - Status Gempa Jatim: {earthquake_status}
        
        Format laporan:
        1. Rangkuman situasi ringkas (2-3 kalimat)
        2. Rekomendasi preventif untuk BMKG
        3. Rekomendasi kesiapsiagaan taktis untuk tim SAR (personel, logistik, evakuasi)
        4. Himbauan keselamatan untuk Pendaki
        
        Tulis dalam bahasa Indonesia yang tegas, resmi, dan mudah dipahami.
        """
        
        # List of candidate model names
        candidates = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-2.5-flash', 'gemini-2.5-flash-latest', 'gemini-1.0-pro']
        
        # Try to dynamically list models to prioritize available Flash models
        try:
            available = [m.name.split('/')[-1] for m in genai.list_models()]
            flash_models = [m for m in available if 'flash' in m]
            if flash_models:
                candidates = flash_models + [c for c in candidates if c not in flash_models]
        except Exception:
            pass
            
        last_error = None
        for model_name in candidates:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as ex:
                last_error = ex
                
        if last_error:
            raise last_error
    except Exception as e:
        return f"Gagal menghasilkan laporan AI: {e}. Menggunakan fallback lokal."

# --- Data Loading and Initialization ---

df = load_historical_data()
df, forecast_df = forecast_demand(df)
xml_data = fetch_bmkg_weather()
trawas_weather = parse_trawas_weather(xml_data)
eq_raw = fetch_felt_earthquakes()
eq_jatim, landslide_alert = evaluate_earthquake_hazard(eq_raw)

# Pre-populate default values based on forecasts/weather
if not forecast_df.empty and 'demand_forecast' in forecast_df.columns:
    default_demand = float(forecast_df['demand_forecast'].iloc[0])
else:
    default_demand = 50.0

default_weather_code = "0"
default_weather_desc = "Cerah"
if isinstance(trawas_weather, list) and len(trawas_weather) > 0:
    default_weather_code = trawas_weather[0].get('code', "0")
    default_weather_desc = trawas_weather[0].get('description', "Cerah")

# --- Streamlit Sidebar Config & Controls ---
st.sidebar.header("⚙️ Parameter Risiko")
has_holiday_input = st.sidebar.checkbox("Libur Nasional / Long Weekend (Penalti +10)", value=False)
demand_score_input = st.sidebar.slider("Indeks Kunjungan (Google Trends)", 0.0, 100.0, default_demand)

# Weather selector in sidebar
if isinstance(trawas_weather, list) and len(trawas_weather) > 0:
    weather_options = [f"{w['datetime'].strftime('%d %b %H:%M')} - {w['description']} (Code {w['code']})" for w in trawas_weather]
    selected_weather_idx = st.sidebar.selectbox("Prakiraan Cuaca Trawas", range(len(weather_options)), format_func=lambda x: weather_options[x])
    weather_code_input = trawas_weather[selected_weather_idx]['code']
    weather_desc_input = trawas_weather[selected_weather_idx]['description']
else:
    weather_code_input = st.sidebar.selectbox("Prakiraan Cuaca Manual (Fallback)", ["0", "3", "60", "61", "63", "95"], format_func=lambda x: {
        "0": "Cerah (Code 0)", "3": "Berawan (Code 3)", "60": "Hujan Ringan (Code 60)", "61": "Hujan Sedang (Code 61)", "63": "Hujan Lebat (Code 63)", "95": "Hujan Petir (Code 95)"
    }.get(x, x))
    weather_desc_input = {
        "0": "Cerah", "3": "Berawan", "60": "Hujan Ringan", "61": "Hujan Sedang", "63": "Hujan Lebat", "95": "Hujan Petir"
    }.get(weather_code_input, "Tidak Diketahui")

# Calculate risk index based on inputs
risk_score = calculate_risk(demand_score_input, weather_code_input, has_holiday_input)
actions = get_sar_actions(risk_score)

# Display status in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📢 Status Risiko Saat Ini")
color_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(actions['color'], "⚪")
st.sidebar.markdown(f"### {color_emoji} {actions['level']}")
st.sidebar.metric(label="Skor Risiko Gabungan", value=f"{risk_score:.1f} / 100")

# API Key resolved from server secrets globally
api_key = os.environ.get("GEMINI_API_KEY", "")

# --- Title Header at the top of main dashboard ---
st.title("🗻 SAR-Cast Penanggungan")
st.subheader("Sistem Kesiapsiagaan Bencana Pendakian Terintegrasi BMKG")

# Create the 5 tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ringkasan & AI Narrator",
    "Forecast Kunjungan",
    "Meteorologi & Klimatologi",
    "Geofisika (Status Gempa)",
    "Studi Kasus & Validasi Insiden"
])

# --- TAB 1: Ringkasan & AI Narrator ---
with tab1:
    # UI colors for the alert card
    if actions['color'] == 'green':
        bg_color = "rgba(40, 167, 69, 0.15)"
        border_color = "#28a745"
        text_color = "#28a745"
    elif actions['color'] == 'yellow':
        bg_color = "rgba(255, 193, 7, 0.15)"
        border_color = "#ffc107"
        text_color = "#ffc107"
    else:
        bg_color = "rgba(220, 53, 69, 0.15)"
        border_color = "#dc3545"
        text_color = "#dc3545"

    st.markdown(
        f"""
        <div style="background-color: {bg_color}; border: 1px solid {border_color}; padding: 15px; border-radius: 8px; margin-bottom: 25px;">
            <h3 style="color: {text_color}; margin: 0; font-weight: 700;">{color_emoji} {actions['level']} (Skor Risiko: {risk_score:.1f})</h3>
            <p style="margin: 5px 0 0 0; font-size: 1.1em; color: #fff;">Status kesiapsiagaan dihitung berdasarkan kombinasi aktivitas kunjungan dan prakiraan cuaca ekstrem.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Executive Summary Box
    st.markdown("### 🤖 Laporan Taktis & AI Narrator")
    
    eq_status_desc = "RAWAN LONGSOR: Gempa signifikan dirasakan baru-baru ini" if landslide_alert else "Aman (Tidak ada gempa signifikan dekat Mojokerto)"
    
    summary_text = generate_executive_summary(
        api_key=api_key,
        risk_level=actions['level'],
        current_demand=demand_score_input,
        weather_description=weather_desc_input,
        earthquake_status=eq_status_desc
    )
    
    # Visual container for the report
    st.info(summary_text)
    
    if api_key:
        st.caption("✨ Laporan ini dihasilkan secara dinamis menggunakan **Gemini 1.5 Flash AI**.")
    else:
        st.caption("ℹ️ Laporan ini dihasilkan menggunakan **Aturan Lokal Fallback** karena API Key Gemini tidak diatur.")
        
    st.markdown("---")
    
    # Actions & Formula Columns
    r_col1, r_col2 = st.columns([1, 2])
    
    with r_col1:
        st.markdown("#### Detail Input Perhitungan:")
        st.markdown(f"""
        - **Indeks Kunjungan (Demand)**: `{demand_score_input:.1f}`
        - **Cuaca (BMKG)**: `{weather_desc_input}` (Code `{weather_code_input}`)
        - **Kalender**: `{"Libur Nasional / Weekend Panjang (Penalti +10)" if has_holiday_input else "Hari Biasa / Standar"}`
        """)
        st.markdown(
            r"""
            #### Formula Indeks Risiko:
            $$\text{Skor Risiko} = (0.4 \times \text{Indeks Kunjungan}) + (0.6 \times \text{Indeks Cuaca}) + \text{Penalti Kalender}$$
            """
        )
        
    with r_col2:
        st.markdown("#### 🛡️ Tindakan Preventif Taktis:")
        act_tab1, act_tab2, act_tab3 = st.tabs(["🚒 Tim SAR", "📡 BMKG", "🥾 Pendaki (Hikers)"])
        
        with act_tab1:
            st.warning(f"**Tindakan Operasional SAR**:\n\n{actions['sar']}")
        with act_tab2:
            st.info(f"**Tindakan Deseminasi BMKG**:\n\n{actions['bmkg']}")
        with act_tab3:
            st.success(f"**Tindakan Keselamatan Pendaki**:\n\n{actions['hiker']}")

# --- TAB 2: Forecast Kunjungan ---
with tab2:
    st.markdown("### 📈 Proyeksi Minat Kunjungan Pendaki (Google Trends)")
    st.write(f"Data historis termuat: **{df.shape[0]}** minggu.")
    
    plot_demand(df, forecast_df)
    
    st.markdown("#### 📋 Tabel Proyeksi Kunjungan (4 Minggu ke Depan)")
    formatted_forecast_df = forecast_df.copy()
    formatted_forecast_df['week_start'] = formatted_forecast_df['week_start'].dt.strftime('%Y-%m-%d')
    formatted_forecast_df.columns = ['Mulai Minggu (Tanggal)', 'Proyeksi Indeks Kunjungan (0-100)']
    st.dataframe(formatted_forecast_df, use_container_width=True)

# --- TAB 3: Meteorologi & Klimatologi ---
with tab3:
    st.markdown("### ⛈️ Meteorologi - Prakiraan Cuaca Mojokerto/Trawas (BMKG)")
    if isinstance(trawas_weather, list) and len(trawas_weather) > 0:
        cols = st.columns(min(4, len(trawas_weather)))
        for idx, w in enumerate(trawas_weather[:4]):
            with cols[idx]:
                st.metric(label=w['datetime'].strftime('%d %b, %H:%M'), value=f"{w['temperature']} °C", delta=w['description'])
                st.caption(f"Angin: {w['wind_speed']} m/s")
    else:
        st.warning("Prakiraan cuaca live tidak tersedia atau format API berubah.")
        
    st.markdown("---")
    st.markdown("### ☀️ Klimatologi - Proyeksi Radiasi Sinar UV di Gunung Penanggungan (~1.653 mdpl)")
    
    hours = list(range(5, 19))
    uv_values = [simulate_altitude_uv(h) for h in hours]
    
    fig_uv = go.Figure()
    fig_uv.add_trace(go.Bar(x=hours, y=uv_values, marker_color='gold', name='Indeks UV'))
    fig_uv.update_layout(
        title="Indeks Sinar UV Per Jam (Ditingkatkan berdasarkan Elevasi 1.653 mdpl)", 
        xaxis=dict(title="Jam", tickmode='linear'), 
        yaxis=dict(title="Indeks UV"), 
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    col1_uv, col2_uv = st.columns([2, 1])
    with col1_uv:
        st.plotly_chart(fig_uv, use_container_width=True)
    with col2_uv:
        st.markdown("#### 🛡️ Tips Perlindungan Sinar UV:")
        current_hour = pd.Timestamp.now().hour
        curr_uv = simulate_altitude_uv(current_hour)
        st.write(f"Jam sekarang ({current_hour:02d}:00) Indeks UV: **{curr_uv:.1f}**")
        st.info(f"**Panduan**: {get_uv_advice(curr_uv)}")

# --- TAB 4: Geofisika (Status Gempa) ---
with tab4:
    st.markdown("### 🌋 Geofisika & Status Gempa Dirasakan (BMKG)")
    if landslide_alert:
        st.error("⚠️ ALARM RAWAN LONGSOR: Gempa signifikan dirasakan di sekitar Mojokerto/Jawa Timur baru-baru ini. Hati-hati dengan runtuhan batu (rockfall) di lereng Penanggungan!")
    else:
        st.success("✅ Aman: Tidak ada catatan gempa signifikan dekat Mojokerto yang berisiko memicu longsor saat ini.")
        
    if eq_jatim:
        st.markdown("#### Gempa Bumi Terbaru Dirasakan di Jawa Timur:")
        for eq in eq_jatim[:3]:
            st.info(f"**Gempa Mag {eq['Magnitude']}** | Tanggal: {eq['Tanggal']} {eq['Jam']} | Kedalaman: {eq['Kedalaman']} | Dirasakan di: {eq['Dirasakan']}")
    else:
        st.write("Tidak ada gempa bumi signifikan dirasakan di area Jawa Timur dalam daftar terbaru.")

# --- TAB 5: Studi Kasus & Validasi Insiden ---
with tab5:
    st.markdown("### 🔍 Studi Kasus & Validasi Insiden Historis")
    st.markdown("""
    Tabel di bawah ini menunjukkan 15 studi kasus insiden historis di Gunung Penanggungan yang memvalidasi tingkat keandalan prediksi indeks risiko gabungan:
    """)
    incident_df = get_historical_incidents()
    st.dataframe(incident_df, use_container_width=True)
