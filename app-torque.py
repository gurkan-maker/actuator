from scipy.interpolate import CubicSpline
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import math
import base64
import tempfile
import os
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
import requests
from PIL import Image
import traceback

# ========================
# CONSTANTS & UNIT CONVERSION
# ========================
BAR_TO_PSI = 14.5038
MM_TO_INCH = 0.0393701
N_TO_LBF = 0.224809
NM_TO_LBIN = 8.85075
KW_TO_HP = 1.34102

# Valve types
VALVE_TYPES = {
    "Ball": {"torque_type": "Rotary", "thrust_type": "None"},
    "Butterfly": {"torque_type": "Rotary", "thrust_type": "None"},
    "Globe": {"torque_type": "None", "thrust_type": "Linear"},
    "Gate": {"torque_type": "None", "thrust_type": "Linear"},
    "Plug": {"torque_type": "Rotary", "thrust_type": "None"},
    "Diaphragm": {"torque_type": "None", "thrust_type": "Linear"}
}

# Safety factors
SAFETY_FACTORS = {
    "Standard": 1.25,
    "High": 1.5,
    "Critical": 2.0
}

# Seal friction coefficients
SEAL_FRICTION = {
    "PTFE": 0.05,
    "Graphite": 0.1,
    "Metal": 0.15,
    "Elastomer": 0.08
}

# ========================
# VALVE DATABASE
# ========================
class Valve:
    def __init__(self, size_inch, valve_type, pressure_class, seat_material, stem_dia_mm, 
                 max_pressure_bar, min_pressure_bar, max_temp_c, min_temp_c):
        self.size = size_inch
        self.type = valve_type
        self.pressure_class = pressure_class
        self.seat_material = seat_material
        self.stem_dia_mm = stem_dia_mm
        self.max_pressure = max_pressure_bar
        self.min_pressure = min_pressure_bar
        self.max_temp = max_temp_c
        self.min_temp = min_temp_c
        
    def get_seal_friction(self):
        return SEAL_FRICTION.get(self.seat_material, 0.1)
    
    def get_area(self):
        # Calculate valve area in m²
        radius_m = (self.size * 0.0254) / 2
        return math.pi * radius_m**2

VALVE_DATABASE = [
    Valve(2, "Ball", 600, "PTFE", 20, 100, 0, 200, -20),
    Valve(4, "Ball", 600, "Metal", 30, 100, 0, 250, -20),
    Valve(6, "Butterfly", 300, "Elastomer", 40, 40, 0, 150, -10),
    Valve(8, "Butterfly", 150, "PTFE", 50, 25, 0, 180, -10),
    Valve(3, "Globe", 900, "Graphite", 25, 150, 0, 350, -50),
    Valve(6, "Gate", 600, "Graphite", 35, 100, 0, 400, -30),
    Valve(2, "Plug", 600, "PTFE", 22, 100, 0, 200, -20),
    Valve(10, "Diaphragm", 150, "Elastomer", 60, 16, 0, 120, -10)
]

# ========================
# ACTUATOR DATABASE
# ========================
class Actuator:
    def __init__(self, model, manufacturer, torque_nm, thrust_n, max_pressure_bar, 
                 min_temp_c, max_temp_c, power_kw, supply_type, weight_kg, price_usd):
        self.model = model
        self.manufacturer = manufacturer
        self.torque = torque_nm
        self.thrust = thrust_n
        self.max_pressure = max_pressure_bar
        self.min_temp = min_temp_c
        self.max_temp = max_temp_c
        self.power = power_kw
        self.supply = supply_type
        self.weight = weight_kg
        self.price = price_usd
        
    def get_torque_lbin(self):
        return self.torque * NM_TO_LBIN
    
    def get_thrust_lbf(self):
        return self.thrust * N_TO_LBF

ACTUATOR_DATABASE = [
    Actuator("SR-100", "Rotork", 1000, 0, 100, -30, 120, 0.5, "Pneumatic", 25, 3500),
    Actuator("SR-500", "Rotork", 5000, 0, 100, -30, 120, 1.0, "Pneumatic", 45, 5500),
    Actuator("IQT-300", "Rotork", 3000, 0, 100, -40, 150, 0.75, "Electric", 40, 6500),
    Actuator("SMC-200", "Emerson", 2000, 0, 100, -20, 100, 0.6, "Pneumatic", 30, 4200),
    Actuator("SMC-800", "Emerson", 8000, 0, 100, -20, 100, 1.5, "Pneumatic", 65, 7800),
    Actuator("F10", "Flowserve", 0, 15000, 150, -50, 200, 1.2, "Hydraulic", 85, 9500),
    Actuator("F25", "Flowserve", 0, 25000, 200, -50, 250, 2.0, "Hydraulic", 120, 12500),
    Actuator("E-200", "AUMA", 2000, 0, 100, -30, 120, 0.8, "Electric", 38, 5800),
    Actuator("E-1000", "AUMA", 10000, 0, 100, -30, 120, 2.5, "Electric", 95, 11200),
    Actuator("H-150", "Honeywell", 0, 20000, 180, -40, 180, 1.8, "Hydraulic", 100, 10500)
]

# ========================
# TORQUE/THRUST CALCULATION MODULE
# ========================
def calculate_ball_valve_torque(valve, pressure_bar, temperature_c):
    """Calculate torque for ball valves based on size, pressure, and temperature"""
    # Base torque calculation (Nm)
    base_torque = valve.size * pressure_bar * 0.8
    
    # Temperature adjustment
    temp_factor = 1.0
    if temperature_c > 100:
        temp_factor = 1.0 + (temperature_c - 100) * 0.005
    
    # Seat material factor
    seat_factor = 1.0
    if valve.seat_material == "Metal":
        seat_factor = 1.3
    elif valve.seat_material == "Graphite":
        seat_factor = 1.1
    
    return base_torque * temp_factor * seat_factor

def calculate_butterfly_valve_torque(valve, pressure_bar, temperature_c):
    """Calculate torque for butterfly valves"""
    # Base torque calculation (Nm)
    base_torque = valve.size**1.5 * pressure_bar * 0.5
    
    # Temperature adjustment
    temp_factor = 1.0
    if temperature_c > 80:
        temp_factor = 1.0 + (temperature_c - 80) * 0.007
    
    # Seat material factor
    seat_factor = 1.0
    if valve.seat_material == "Elastomer":
        seat_factor = 0.9
    elif valve.seat_material == "PTFE":
        seat_factor = 1.0
    
    return base_torque * temp_factor * seat_factor

def calculate_globe_valve_thrust(valve, pressure_bar, temperature_c):
    """Calculate thrust for globe valves"""
    # Differential pressure force
    area = valve.get_area()
    dp_force = area * pressure_bar * 100000  # N
    
    # Packing friction force
    stem_area = math.pi * (valve.stem_dia_mm / 1000 / 2)**2
    packing_force = stem_area * pressure_bar * 100000 * valve.get_seal_friction()
    
    # Seat load force (empirical)
    seat_force = valve.size * 1000
    
    # Temperature factor
    temp_factor = 1.0
    if temperature_c > 150:
        temp_factor = 1.0 + (temperature_c - 150) * 0.002
    
    return (dp_force + packing_force + seat_force) * temp_factor

def calculate_gate_valve_thrust(valve, pressure_bar, temperature_c):
    """Calculate thrust for gate valves"""
    # Differential pressure force
    area = valve.get_area()
    dp_force = area * pressure_bar * 100000  # N
    
    # Packing friction force
    stem_area = math.pi * (valve.stem_dia_mm / 1000 / 2)**2
    packing_force = stem_area * pressure_bar * 100000 * valve.get_seal_friction()
    
    # Wedge effect factor
    wedge_factor = 1.5 if valve.type == "Gate" else 1.0
    
    # Temperature factor
    temp_factor = 1.0
    if temperature_c > 200:
        temp_factor = 1.0 + (temperature_c - 200) * 0.0015
    
    return (dp_force * wedge_factor + packing_force) * temp_factor

def calculate_valve_torque_thrust(valve, pressure_bar, temperature_c):
    """Calculate torque or thrust based on valve type"""
    if valve.type == "Ball" or valve.type == "Plug":
        return calculate_ball_valve_torque(valve, pressure_bar, temperature_c), "Torque (Nm)"
    elif valve.type == "Butterfly":
        return calculate_butterfly_valve_torque(valve, pressure_bar, temperature_c), "Torque (Nm)"
    elif valve.type == "Globe":
        return calculate_globe_valve_thrust(valve, pressure_bar, temperature_c), "Thrust (N)"
    elif valve.type == "Gate":
        return calculate_gate_valve_thrust(valve, pressure_bar, temperature_c), "Thrust (N)"
    elif valve.type == "Diaphragm":
        # Diaphragm valves use thrust but with different calculation
        area = valve.get_area()
        return area * pressure_bar * 100000 * 1.2, "Thrust (N)"
    else:
        return 0, "Unknown"

# ========================
# ACTUATOR SELECTION LOGIC
# ========================
def find_suitable_actuators(required_value, value_type, valve, safety_factor, supply_type=None):
    """Find actuators that meet the torque/thrust requirements"""
    suitable_actuators = []
    
    for actuator in ACTUATOR_DATABASE:
        # Skip actuators that don't match the required motion type
        if value_type == "Torque (Nm)" and actuator.torque == 0:
            continue
        if value_type == "Thrust (N)" and actuator.thrust == 0:
            continue
            
        # Check if actuator matches supply type if specified
        if supply_type and supply_type != "Any" and actuator.supply != supply_type:
            continue
            
        # Check pressure rating
        if actuator.max_pressure < valve.max_pressure:
            continue
            
        # Check temperature range
        if actuator.min_temp > valve.min_temp or actuator.max_temp < valve.max_temp:
            continue
            
        # Check torque/thrust capability with safety factor
        if value_type == "Torque (Nm)":
            capability = actuator.torque
        else:
            capability = actuator.thrust
            
        if capability >= required_value * safety_factor:
            # Calculate margin
            margin = (capability / (required_value * safety_factor) - 1) * 100
            suitable_actuators.append({
                "actuator": actuator,
                "capability": capability,
                "margin": margin
            })
    
    # Sort by capability (ascending) to get the most economical first
    suitable_actuators.sort(key=lambda x: x["capability"])
    
    return suitable_actuators

# ========================
# ENHANCED PDF REPORT GENERATION
# ========================
class EnhancedPDFReport(FPDF):
    def __init__(self, logo_bytes=None, logo_type=None):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.logo_bytes = logo_bytes
        self.logo_type = logo_type
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        self.set_title("Actuator Sizing Report")
        self.set_author("VASTAŞ Actuator Sizing Software")
        self.alias_nb_pages()
        self.set_compression(True)
        
        # Add Unicode support
        self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        self.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
        self.add_font('DejaVu', 'I', 'DejaVuSans-Oblique.ttf', uni=True)
        self.add_font('DejaVu', 'BI', 'DejaVuSans-BoldOblique.ttf', uni=True)
    
    def header(self):
        if self.page_no() == 1:
            return
            
        # Draw top border
        self.set_draw_color(0, 51, 102)
        self.set_line_width(0.5)
        self.line(10, 15, 200, 15)
        
        # Logo
        if self.logo_bytes and self.logo_type:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{self.logo_type.lower()}") as tmpfile:
                    tmpfile.write(self.logo_bytes)
                    tmpfile_path = tmpfile.name
                self.image(tmpfile_path, x=15, y=8, w=20)
                os.unlink(tmpfile_path)
            except Exception as e:
                pass
        
        # Title
        self.set_font('DejaVu', 'B', 10)
        self.set_text_color(0, 51, 102)
        self.set_y(10)
        self.cell(0, 10, 'Actuator Sizing Report', 0, 0, 'C')
        
        # Page number
        self.set_font('DejaVu', 'I', 8)
        self.set_text_color(100)
        self.set_y(10)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'R')
        
        # Line break
        self.ln(15)
        
    def footer(self):
        if self.page_no() == 1:
            return
            
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.set_text_color(100)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'L')
        self.cell(0, 10, 'Confidential - VASTAŞ Valve Technologies', 0, 0, 'R')
    
    def cover_page(self, title, subtitle, project_info=None):
        self.add_page()
        
        # Background rectangle
        self.set_fill_color(0, 51, 102)
        self.rect(0, 0, 210, 297, 'F')
        
        # Main content area
        self.set_fill_color(255, 255, 255)
        self.rect(15, 15, 180, 267, 'F')
        
        # Logo at top
        if self.logo_bytes and self.logo_type:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{self.logo_type.lower()}") as tmpfile:
                    tmpfile.write(self.logo_bytes)
                    tmpfile_path = tmpfile.name
                self.image(tmpfile_path, x=80, y=40, w=50)
                os.unlink(tmpfile_path)
            except Exception as e:
                pass
        
        # Title
        self.set_y(120)
        self.set_font('DejaVu', 'B', 24)
        self.set_text_color(0, 51, 102)
        self.cell(0, 15, title, 0, 1, 'C')
        
        # Subtitle
        self.set_font('DejaVu', 'I', 18)
        self.set_text_color(70, 70, 70)
        self.cell(0, 10, subtitle, 0, 1, 'C')
        
        # Project info
        if project_info:
            self.set_font('DejaVu', '', 14)
            self.set_text_color(0, 0, 0)
            self.ln(20)
            self.cell(0, 10, project_info, 0, 1, 'C')
        
        # Company info
        self.set_y(220)
        self.set_font('DejaVu', 'B', 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, 'VASTAŞ Valve Technologies', 0, 1, 'C')
        
        # Date
        self.set_font('DejaVu', 'I', 12)
        self.set_text_color(70, 70, 70)
        self.cell(0, 10, datetime.now().strftime("%B %d, %Y"), 0, 1, 'C')
        
        # Confidential notice
        self.set_y(270)
        self.set_font('DejaVu', 'I', 10)
        self.set_text_color(150, 0, 0)
        self.cell(0, 5, 'CONFIDENTIAL - For internal use only', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 14)
        self.set_text_color(0, 51, 102)
        self.set_fill_color(230, 240, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(5)
    
    def chapter_body(self, body, font_size=12):
        self.set_font('DejaVu', '', font_size)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, body)
        self.ln()
    
    def add_table(self, headers, data, col_widths=None, header_color=(0, 51, 102), 
                  row_colors=[(255, 255, 255), (240, 248, 255)]):
        if col_widths is None:
            col_widths = [self.w / len(headers)] * len(headers)
        
        # Table header
        self.set_font('DejaVu', 'B', 10)
        self.set_text_color(255, 255, 255)
        self.set_fill_color(*header_color)
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C', 1)
        self.ln()
        
        # Table data
        self.set_font('DejaVu', '', 10)
        self.set_text_color(0, 0, 0)
        
        for row_idx, row in enumerate(data):
            fill_color = row_colors[row_idx % len(row_colors)]
            self.set_fill_color(*fill_color)
            
            for i, item in enumerate(row):
                self.cell(col_widths[i], 6, str(item), 1, 0, 'C', 1)
            self.ln()
    
    def add_key_value_table(self, data, col_widths=[70, 130], font_size=10):
        self.set_font('DejaVu', 'B', font_size)
        self.set_text_color(0, 51, 102)
        self.set_fill_color(240, 248, 255)
        
        for key, value in data:
            self.cell(col_widths[0], 7, key, 1, 0, 'L', 1)
            self.set_font('DejaVu', '', font_size)
            self.set_text_color(0, 0, 0)
            self.set_fill_color(255, 255, 255)
            self.multi_cell(col_widths[1], 7, str(value), 1, 'L', 1)
            self.set_font('DejaVu', 'B', font_size)
            self.set_text_color(0, 51, 102)
            self.set_fill_color(240, 248, 255)

# ========================
# VISUALIZATION FUNCTIONS
# ========================
def plot_actuator_comparison(actuators):
    if not actuators:
        return None
        
    df = pd.DataFrame([{
        "Model": a["actuator"].model,
        "Manufacturer": a["actuator"].manufacturer,
        "Capability": a["capability"],
        "Margin": a["margin"],
        "Power": a["actuator"].power,
        "Weight": a["actuator"].weight,
        "Price": a["actuator"].price
    } for a in actuators])
    
    fig = px.bar(df, x="Model", y="Capability", 
                 color="Manufacturer",
                 title="Actuator Capability Comparison",
                 labels={"Capability": "Torque/Thrust Capability"},
                 hover_data=["Margin", "Power", "Weight", "Price"])
    
    fig.update_layout(barmode='group', height=500)
    return fig

def plot_torque_thrust_vs_pressure(valve, temperature_c, max_pressure):
    pressures = np.linspace(0, max_pressure, 20)
    values = []
    labels = []
    
    for pressure in pressures:
        value, value_type = calculate_valve_torque_thrust(valve, pressure, temperature_c)
        values.append(value)
        labels.append(value_type)
    
    value_type = labels[0]  # All will be the same
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pressures, 
        y=values, 
        mode='lines+markers',
        name=value_type,
        line=dict(width=3)
    ))
    
    fig.update_layout(
        title=f"{valve.type} Valve {value_type.split(' ')[0]} vs Pressure",
        xaxis_title="Pressure (bar)",
        yaxis_title=value_type,
        height=500,
        template='plotly_white'
    )
    
    return fig

# ========================
# STREAMLIT APPLICATION
# ========================
def main():
    st.set_page_config(
        page_title="Valve Torque/Thrust Calculator",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 1rem;
        }
        .result-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .success-card {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
        }
        .warning-card {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
        }
        .danger-card {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            padding: 10px 0;
        }
        .stMetric {
            font-size: 20px !important;
        }
        .stNumberInput, .stTextInput, .stSelectbox {
            font-size: 18px;
        }
        .stMarkdown {
            font-size: 18px;
        }
        .actuator-table {
            width: 100%;
            border-collapse: collapse;
        }
        .actuator-table th {
            background-color: #2c3e50;
            color: white;
            padding: 10px;
            text-align: center;
        }
        .actuator-table td {
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
        }
        .status-green {
            background-color: #d4edda;
        }
        .status-yellow {
            background-color: #fff3cd;
        }
        .status-red {
            background-color: #f8d7da;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'valve' not in st.session_state:
        st.session_state.valve = None
    if 'logo_bytes' not in st.session_state:
        st.session_state.logo_bytes = None
    if 'logo_type' not in st.session_state:
        st.session_state.logo_type = None
    
    col1, col2 = st.columns([1, 4])
    with col1:
        default_logo = "logo.png"
        if os.path.exists(default_logo):
            st.image(default_logo, width=100)
        else:
            st.image("https://via.placeholder.com/100x100?text=LOGO", width=100)
    with col2:
        st.title("Valve Torque/Thrust Calculator")
        st.markdown("**Actuator Sizing and Selection Tool**")
    
    with st.sidebar:
        st.header("VASTAŞ Logo")
        logo_upload = st.file_uploader("Upload VASTAŞ logo", type=["png", "jpg", "jpeg"], key="logo_uploader")
        if logo_upload is not None:
            st.session_state.logo_bytes = logo_upload.getvalue()
            st.session_state.logo_type = "PNG"
            st.success("Logo uploaded successfully!")
        if st.session_state.logo_bytes:
            st.image(Image.open(BytesIO(st.session_state.logo_bytes)), use_container_width=True)
        elif os.path.exists("logo.png"):
            st.image(Image.open("logo.png"), use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x100?text=VASTAŞ+Logo", use_container_width=True)
        
        st.header("Valve Selection")
        valve_options = {f"{valve.size}\" {valve.type} (Class {valve.pressure_class})": valve for valve in VALVE_DATABASE}
        selected_valve_name = st.selectbox("Select Valve", list(valve_options.keys()))
        selected_valve = valve_options[selected_valve_name]
        
        st.header("Operating Conditions")
        pressure = st.number_input("Operating Pressure (bar)", min_value=0.0, max_value=500.0, value=10.0, step=1.0)
        temperature = st.number_input("Operating Temperature (°C)", min_value=-50.0, max_value=500.0, value=20.0, step=1.0)
        safety_factor = st.selectbox("Safety Factor", list(SAFETY_FACTORS.keys()), index=0)
        supply_type = st.selectbox("Actuator Supply Type", ["Any", "Pneumatic", "Electric", "Hydraulic"])
        
        st.header("Actions")
        calculate_btn = st.button("Calculate Torque/Thrust", type="primary", use_container_width=True)
        export_btn = st.button("Export PDF Report", use_container_width=True)
        
        st.header("Valve Details")
        st.markdown(f"**Type:** {selected_valve.type}")
        st.markdown(f"**Size:** {selected_valve.size}\"")
        st.markdown(f"**Pressure Class:** {selected_valve.pressure_class}")
        st.markdown(f"**Seat Material:** {selected_valve.seat_material}")
        st.markdown(f"**Stem Diameter:** {selected_valve.stem_dia_mm} mm")
        st.markdown(f"**Max Pressure:** {selected_valve.max_pressure} bar")
        st.markdown(f"**Temp Range:** {selected_valve.min_temp}°C to {selected_valve.max_temp}°C")
    
    # Main content area
    tab1, tab2 = st.tabs(["Calculation", "Actuator Selection"])
    
    with tab1:
        st.subheader("Torque/Thrust Calculation")
        
        if calculate_btn:
            try:
                # Calculate torque/thrust
                required_value, value_type = calculate_valve_torque_thrust(selected_valve, pressure, temperature)
                sf_value = SAFETY_FACTORS[safety_factor]
                required_with_sf = required_value * sf_value
                
                st.session_state.results = {
                    "required_value": required_value,
                    "value_type": value_type,
                    "safety_factor": sf_value,
                    "required_with_sf": required_with_sf,
                    "valve": selected_valve,
                    "pressure": pressure,
                    "temperature": temperature
                }
            except Exception as e:
                st.error(f"Calculation error: {str(e)}")
                st.error(traceback.format_exc())
        
        if st.session_state.results:
            results = st.session_state.results
            value_type = results["value_type"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Calculated Requirement", f"{results['required_value']:.1f}", value_type)
            with col2:
                st.metric("Safety Factor", f"{results['safety_factor']:.2f}")
            with col3:
                st.metric("Required with Safety Factor", f"{results['required_with_sf']:.1f}", value_type)
            
            st.subheader("Visualization")
            fig = plot_torque_thrust_vs_pressure(
                results["valve"], 
                results["temperature"],
                min(100, results["valve"].max_pressure)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Calculation Details")
            with st.expander("View calculation parameters"):
                st.markdown(f"**Valve Type:** {results['valve'].type}")
                st.markdown(f"**Valve Size:** {results['valve'].size}\"")
                st.markdown(f"**Seat Material:** {results['valve'].seat_material}")
                st.markdown(f"**Stem Diameter:** {results['valve'].stem_dia_mm} mm")
                st.markdown(f"**Operating Pressure:** {results['pressure']} bar")
                st.markdown(f"**Operating Temperature:** {results['temperature']} °C")
                st.markdown(f"**Safety Factor:** {results['safety_factor']} ({safety_factor})")
                
                if "Torque" in value_type:
                    st.markdown("""
                    **Torque Calculation Method:**
                    - For ball valves: Torque = Size × Pressure × Material Factor × Temperature Factor
                    - For butterfly valves: Torque = Size¹·⁵ × Pressure × Material Factor × Temperature Factor
                    """)
                else:
                    st.markdown("""
                    **Thrust Calculation Method:**
                    - Differential pressure force: Area × Pressure
                    - Packing friction force: Stem Area × Pressure × Friction Coefficient
                    - Seat load: Empirical value based on valve size
                    """)
    
    with tab2:
        st.subheader("Actuator Selection")
        
        if st.session_state.results:
            results = st.session_state.results
            value_type = results["value_type"]
            
            # Find suitable actuators
            suitable_actuators = find_suitable_actuators(
                results["required_value"],
                results["value_type"],
                results["valve"],
                results["safety_factor"],
                supply_type
            )
            
            if suitable_actuators:
                st.success(f"Found {len(suitable_actuators)} suitable actuators")
                
                # Show actuator comparison chart
                st.plotly_chart(plot_actuator_comparison(suitable_actuators), use_container_width=True)
                
                # Show actuator table
                st.subheader("Recommended Actuators")
                actuator_data = []
                for actuator in suitable_actuators:
                    act = actuator["actuator"]
                    capability = actuator["capability"]
                    margin = actuator["margin"]
                    
                    # Determine status
                    if margin > 50:
                        status = "Over-sized"
                        status_class = "status-yellow"
                    elif margin > 20:
                        status = "Well-sized"
                        status_class = "status-green"
                    else:
                        status = "Minimal margin"
                        status_class = "status-red"
                    
                    actuator_data.append([
                        f"{act.manufacturer} {act.model}",
                        act.supply,
                        f"{capability:.1f}",
                        f"{act.power:.1f} kW",
                        f"{act.weight} kg",
                        f"${act.price}",
                        f"{margin:.1f}%",
                        status
                    ])
                
                # Create HTML table with styling
                html_table = """
                <table class="actuator-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Type</th>
                            <th>Capability</th>
                            <th>Power</th>
                            <th>Weight</th>
                            <th>Price</th>
                            <th>Margin</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for row in actuator_data:
                    html_table += "<tr>"
                    for i, item in enumerate(row):
                        if i == 7:  # Status column
                            if "Over-sized" in item:
                                html_table += f'<td class="status-yellow">{item}</td>'
                            elif "Well-sized" in item:
                                html_table += f'<td class="status-green">{item}</td>'
                            else:
                                html_table += f'<td class="status-red">{item}</td>'
                        else:
                            html_table += f"<td>{item}</td>"
                    html_table += "</tr>"
                
                html_table += "</tbody></table>"
                
                st.markdown(html_table, unsafe_allow_html=True)
            else:
                st.warning("No suitable actuators found. Consider increasing the safety factor or selecting a different valve.")
        else:
            st.info("Perform a calculation first to see actuator recommendations")
    
    if export_btn and st.session_state.results:
        try:
            # Generate PDF report
            pdf = EnhancedPDFReport(logo_bytes=st.session_state.logo_bytes, 
                                  logo_type=st.session_state.logo_type)
            
            # Cover page
            pdf.cover_page(
                title="VALVE ACTUATOR SIZING REPORT",
                subtitle="Torque/Thrust Calculation",
                project_info=f"Prepared by VASTAŞ Engineering Department"
            )
            
            # Valve details
            pdf.add_page()
            pdf.chapter_title('Valve Details')
            valve = st.session_state.results["valve"]
            valve_details = [
                ("Valve Type:", valve.type),
                ("Size:", f"{valve.size}\""),
                ("Pressure Class:", str(valve.pressure_class)),
                ("Seat Material:", valve.seat_material),
                ("Stem Diameter:", f"{valve.stem_dia_mm} mm"),
                ("Max Pressure:", f"{valve.max_pressure} bar"),
                ("Temperature Range:", f"{valve.min_temp}°C to {valve.max_temp}°C")
            ]
            pdf.add_key_value_table(valve_details)
            
            # Operating conditions
            pdf.chapter_title('Operating Conditions')
            op_conditions = [
                ("Operating Pressure:", f"{st.session_state.results['pressure']} bar"),
                ("Operating Temperature:", f"{st.session_state.results['temperature']} °C"),
                ("Safety Factor:", f"{st.session_state.results['safety_factor']} ({safety_factor})")
            ]
            pdf.add_key_value_table(op_conditions)
            
            # Calculation results
            pdf.chapter_title('Calculation Results')
            value_type = st.session_state.results["value_type"]
            calc_results = [
                ("Required Value:", f"{st.session_state.results['required_value']:.1f} {value_type}"),
                ("With Safety Factor:", f"{st.session_state.results['required_with_sf']:.1f} {value_type}")
            ]
            pdf.add_key_value_table(calc_results)
            
            # Actuator recommendations
            suitable_actuators = find_suitable_actuators(
                st.session_state.results["required_value"],
                st.session_state.results["value_type"],
                st.session_state.results["valve"],
                st.session_state.results["safety_factor"],
                supply_type
            )
            
            if suitable_actuators:
                pdf.add_page()
                pdf.chapter_title('Recommended Actuators')
                actuator_data = []
                for actuator in suitable_actuators:
                    act = actuator["actuator"]
                    actuator_data.append([
                        f"{act.manufacturer} {act.model}",
                        act.supply,
                        f"{actuator['capability']:.1f}",
                        f"{actuator['margin']:.1f}%",
                        f"{act.power:.1f} kW",
                        f"{act.weight} kg",
                        f"${act.price}"
                    ])
                pdf.add_table(
                    ["Model", "Type", "Capability", "Margin", "Power", "Weight", "Price"],
                    actuator_data,
                    col_widths=[40, 25, 25, 20, 20, 20, 20]
                )
            
            # Generate PDF in memory
            pdf_bytes_io = BytesIO()
            pdf.output(pdf_bytes_io)
            pdf_bytes_io.seek(0)
            
            # Download button
            st.sidebar.download_button(
                label="Download PDF Report",
                data=pdf_bytes_io,
                file_name=f"actuator_sizing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            st.sidebar.success("PDF report generated successfully!")
        except Exception as e:
            st.sidebar.error(f"PDF generation failed: {str(e)}")
            st.sidebar.text(traceback.format_exc())

if __name__ == "__main__":
    main()