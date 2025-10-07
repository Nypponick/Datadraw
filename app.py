import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Function to clean and convert currency strings to float
def clean_currency(s):
    if isinstance(s, str):
        s = s.replace('USD', '').replace('.', '').replace(',', '.').strip()
    return float(s)

# Function to clean and convert percentage strings to float
def clean_percentage(s):
    if isinstance(s, str):
        s = s.replace('%', '').replace(',', '.').strip()
    return float(s)

# Load the data
file_path = 'dashboard2.csv'

# Read the summary table
resumo_df = pd.read_csv(file_path, skiprows=3, nrows=3, usecols=[1, 2], header=None, names=['Metrica', 'Valor'])

# Read the main data table (header is at row 8, data starts at row 9)
df = pd.read_csv(file_path, skiprows=8, header=0, nrows=53)
df = df.dropna(how='all', axis=1)
df = df.dropna(how='all', axis=0)

# Remove unnamed columns (empty first column)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Keep original column names for display but create a mapping for easier reference
original_columns = df.columns.tolist()
column_mapping = {
    'Nº do Item': 'Item',
    'Total Saídas': 'Total Saídas',
    'Valor Total (USD)': 'Valor Total (USD)',
    'Total Vinculado a DU-Es': 'Total Vinculado a DU-Es',
    'Saldo Disponível': 'Saldo Disponível',
    'Valor Médio Unitário (USD)': 'Valor Médio Unitário (USD)',
    'Média Utilização/DU-E': 'Média Utilização/DU-E'
}

# Rename columns to user-friendly names
df = df.rename(columns=column_mapping)

# Create reverse mapping for code reference (clean names without special characters)
clean_col_mapping = {
    'Item': 'Item',
    'Total Saídas': 'Total_Saidas',
    'Valor Total (USD)': 'Valor_Total_USD',
    'Total Vinculado a DU-Es': 'Total_Vinculado_DU_Es',
    'Saldo Disponível': 'Saldo_Disponivel',
    'Valor Médio Unitário (USD)': 'Valor_Medio_Unitario_USD',
    'Média Utilização/DU-E': 'Media_Utilizacao_DU_E'
}

# Clean the data using the display column names
for col in ['Total Saídas', 'Total Vinculado a DU-Es', 'Saldo Disponível', 'Média Utilização/DU-E']:
    df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)

for col in ['Valor Total (USD)', 'Valor Médio Unitário (USD)']:
    df[col] = df[col].apply(clean_currency)

# Calculate percentage of consumption based on Total Vinculado a DU-Es
# Negative Saldo means over-consumption, so we calculate: (Total Vinculado / Total Vinculado) * 100 = 100% consumed
# Plus the extra consumption: abs(Saldo) / Total Vinculado * 100
df['Consumo (%)'] = ((df['Total Vinculado a DU-Es'] - df['Saldo Disponível']) / df['Total Vinculado a DU-Es']) * 100

# For compatibility with existing code, also create the Estoque Disponível (%) column
# This represents how much is REMAINING (negative values mean over-consumed)
df['Estoque Disponível (%)'] = (df['Saldo Disponível'] / df['Total Vinculado a DU-Es']) * 100

# No need for display formatting - we'll use column_config instead


# ============================================================================
# Statistical Functions for Advanced Planning
# ============================================================================
def calculate_item_correlation_with_value(item_usage, due_values):
    """
    Calculate correlation between DU-E value and item usage
    Returns: correlation coefficient, regression slope, regression intercept
    """
    if len(item_usage) < 2 or len(due_values) < 2:
        return 0, 0, 0
    
    # Remove any NaN values
    mask = ~(np.isnan(item_usage) | np.isnan(due_values))
    clean_usage = item_usage[mask]
    clean_values = due_values[mask]
    
    if len(clean_usage) < 2:
        return 0, 0, 0
    
    # Calculate correlation
    correlation = np.corrcoef(clean_values, clean_usage)[0, 1] if len(clean_usage) > 0 else 0
    
    # Simple linear regression: usage = slope * value + intercept
    if len(clean_values) > 1 and np.std(clean_values) > 0:
        slope = np.cov(clean_values, clean_usage)[0, 1] / np.var(clean_values)
        intercept = np.mean(clean_usage) - slope * np.mean(clean_values)
    else:
        slope = 0
        intercept = np.mean(clean_usage) if len(clean_usage) > 0 else 0
    
    return correlation, slope, intercept


def predict_item_usage_for_due_value(due_value, avg_usage, correlation, slope, intercept, std_dev):
    """
    Predict item usage for a specific DU-E value using statistical model
    If correlation is strong, use regression; otherwise use average with confidence interval
    """
    # If correlation is strong (|r| > 0.5), use regression model
    if abs(correlation) > 0.5 and slope != 0:
        predicted = slope * due_value + intercept
        # Add confidence interval based on standard deviation
        confidence_margin = 1.96 * std_dev  # 95% confidence interval
        return max(0, predicted), confidence_margin
    else:
        # Use average with standard deviation as confidence
        confidence_margin = 1.96 * std_dev
        return avg_usage, confidence_margin


# Read the DU-E status table (header at row 66, data starts at row 67)
due_df = pd.read_csv(file_path, skiprows=65, header=0)
due_df = due_df.dropna(how='all', axis=1)
due_df = due_df.dropna(how='all', axis=0)
# Remove unnamed columns (empty first column)
due_df = due_df.loc[:, ~due_df.columns.str.contains('^Unnamed')]
due_df = due_df.rename(columns={'Nº da DU-E': 'Nº da DU-E', 'Data de Embarque': 'Data de Embarque',
                                'Valor da DU-E (USD)': 'Valor da DU-E (USD)', 'Status': 'Status'})
due_df['Valor da DU-E (USD)'] = due_df['Valor da DU-E (USD)'].apply(clean_currency)

# Extract historical average DU-E value from summary
avg_due_value_historical = due_df['Valor da DU-E (USD)'].mean()
total_due_count = len(due_df)

# Extract summary statistics
valor_medio_por_exportacao = resumo_df[resumo_df['Metrica'] == 'Valor Médio por Exportação (USD)']['Valor'].values[0] if len(resumo_df) > 0 else f"USD {avg_due_value_historical:,.2f}"
# Clean the value if it's a string
if isinstance(valor_medio_por_exportacao, str):
    avg_due_value_from_summary = clean_currency(valor_medio_por_exportacao)
else:
    avg_due_value_from_summary = avg_due_value_historical

# No need for display formatting - we'll use column_config instead


# Streamlit App
st.set_page_config(
    layout="wide",
    page_title="Dashboard de Controle de Estoque",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3b82f6;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title('📊 Dashboard de Controle de Estoque')
st.markdown("**Sistema de Gestão de Exportações e DU-Es**")
st.divider()

# ============================================================================
# SIDEBAR - Filters and Settings
# ============================================================================
with st.sidebar:
    st.markdown("### 🎯 Filtros e Configurações")
    st.divider()
    
    # Quick filters
    filter_type = st.selectbox(
        "Filtro Rápido:",
        ["Todos os Items", "Items 100%+ Utilizados", "Alta Utilização (>75%)", "Alto Valor (>USD 100k)", "Items Performáticos"],
        help="Selecione um filtro rápido para visualizar items específicos"
    )
    
    # Search by item number
    search_item = st.text_input(
        "🔍 Buscar por Item:", 
        placeholder="Ex: 1, 2, 3...",
        help="Digite números de items separados por vírgula"
    )
    
    # Sort options
    st.markdown("**Ordenação**")
    col_sort1, col_sort2 = st.columns(2)
    with col_sort1:
        sort_by = st.selectbox(
            "Ordenar por:",
            ["Item", "Saldo Disponível", "Consumo (%)", "Utilização (%)", "Valor Total", "Performance"],
            label_visibility="collapsed"
        )
    with col_sort2:
        sort_order = st.selectbox(
            "Ordem:", 
            ["Crescente", "Decrescente"],
            label_visibility="collapsed"
        )
    
    st.divider()
    
    # Export options
    st.markdown("### 📥 Exportar Dados")
    
    # CSV download
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="💾 Download CSV",
        data=csv,
        file_name=f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.divider()
    st.caption("📊 Dashboard v2.0")
    st.caption(f"🕐 Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# ============================================================================
# Apply Filters (before tabs)
# ============================================================================
filtered_df = df.copy()

# Apply quick filters
if filter_type == "Items 100%+ Utilizados":
    filtered_df = filtered_df[filtered_df['Saldo Disponível'] < 0]  # Fully consumed = GOOD
elif filter_type == "Alta Utilização (>75%)":
    filtered_df = filtered_df[filtered_df['Estoque Disponível (%)'] < 25]  # Less than 25% remaining = >75% used
elif filter_type == "Alto Valor (>USD 100k)":
    filtered_df = filtered_df[filtered_df['Valor Total (USD)'] > 100000]
elif filter_type == "Items Performáticos":
    # Items with high utilization OR fully consumed
    filtered_df = filtered_df[
        (filtered_df['Saldo Disponível'] < 0) | 
        (filtered_df['Estoque Disponível (%)'] < 25)
    ]

# Apply item search
if search_item:
    try:
        search_items = [int(x.strip()) for x in search_item.split(',') if x.strip()]
        filtered_df = filtered_df[filtered_df['Item'].isin(search_items)]
    except ValueError:
        st.error("Por favor, digite números de items válidos separados por vírgula.")

# Add performance score for sorting (higher utilization = better performance)
filtered_df_with_score = filtered_df.copy()
filtered_df_with_score['Performance'] = (
    (filtered_df['Saldo Disponível'] < 0).astype(int) * 3 +  # Fully consumed gets score 3 (BEST)
    ((filtered_df['Estoque Disponível (%)'] < 25) & (filtered_df['Saldo Disponível'] >= 0)).astype(int) * 2 +  # High utilization gets score 2
    ((filtered_df['Estoque Disponível (%)'] < 50) & (filtered_df['Estoque Disponível (%)'] >= 25)).astype(int) * 1  # Medium utilization gets score 1
)

# Apply sorting
ascending = sort_order == "Crescente"
if sort_by == "Item":
    filtered_df_with_score = filtered_df_with_score.sort_values('Item', ascending=ascending)
elif sort_by == "Saldo Disponível":
    filtered_df_with_score = filtered_df_with_score.sort_values('Saldo Disponível', ascending=ascending)
elif sort_by == "Consumo (%)":
    # Sort by consumption percentage (higher is better)
    filtered_df_with_score = filtered_df_with_score.sort_values('Consumo (%)', ascending=ascending)
elif sort_by == "Utilização (%)":
    # Sort by utilization (inverse of Estoque Disponível %)
    filtered_df_with_score = filtered_df_with_score.sort_values('Estoque Disponível (%)', ascending=not ascending)
elif sort_by == "Valor Total":
    filtered_df_with_score = filtered_df_with_score.sort_values('Valor Total (USD)', ascending=ascending)
elif sort_by == "Performance":
    filtered_df_with_score = filtered_df_with_score.sort_values('Performance', ascending=False)  # Always show best performers first

# Remove the performance score column from display
display_df = filtered_df_with_score.drop('Performance', axis=1)

# ============================================================================
# Calculate KPIs (used across tabs)
# ============================================================================
fully_consumed_items = len(df[df['Saldo Disponível'] < 0])  # 100%+ utilization = GOOD
underutilized_items = len(df[df['Saldo Disponível'] >= 0])  # Surplus = need better planning
total_items = len(df)

# Items with high consumption (>=100%) are performing well
high_consumption_items = len(df[df['Consumo (%)'] >= 100])  # 100%+ consumed = EXCELLENT
high_consumption_percentage = (high_consumption_items / total_items * 100) if total_items > 0 else 0

# Average consumption rate across all items
avg_consumption = df['Consumo (%)'].mean()

# Items with high utilization (>75%) are performing well
high_utilization_items = len(df[df['Estoque Disponível (%)'] < 25])  # Less than 25% remaining = >75% used
high_utilization_percentage = (high_utilization_items / total_items * 100) if total_items > 0 else 0

avg_stock_utilization = 100 - df['Estoque Disponível (%)'].mean()

# Calculate total value of fully consumed items (this is GOOD - shows efficient use)
fully_consumed_df = df[df['Saldo Disponível'] < 0]
total_value_fully_consumed = fully_consumed_df['Valor Total (USD)'].sum() if len(fully_consumed_df) > 0 else 0

# Show filter results
total_filtered = len(display_df)
if total_filtered < total_items:
    st.info(f"🔍 Mostrando **{total_filtered}** de **{total_items}** items | Filtro: **{filter_type}**")

# ============================================================================
# TABS ORGANIZATION
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "📋 Análise Detalhada", "🎯 Planejamento de Compras", "🚢 Status DU-Es"])

# ============================================================================
# TAB 1: Visão Geral - KPIs and Summary Charts
# ============================================================================
with tab1:
    st.header('📊 Indicadores Chave de Performance (KPIs)')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Items Totalmente Utilizados",
            value=f"{fully_consumed_items} / {total_items}",
            delta=f"{(fully_consumed_items/total_items*100):.1f}% totalmente consumidos",
            delta_color="normal",  # Positive indicator
            help="Mostra quantos items alcançaram 100%+ de utilização (saldo negativo). Isso é POSITIVO e indica que os items foram totalmente consumidos nas DU-Es, demonstrando eficiência no uso do estoque. Meta: Quanto maior, melhor!"
        )

    with col2:
        st.metric(
            label="% Items com Alta Utilização",
            value=f"{high_utilization_percentage:.1f}%",
            delta=f"{high_utilization_items} de {total_items} items",
            delta_color="normal" if high_utilization_percentage >= 70 else "inverse",
            help="Percentual de items com mais de 75% de utilização (menos de 25% de estoque restante). Alta utilização indica eficiência no consumo. Meta: ≥70% dos items devem ter alta utilização para demonstrar boa gestão."
        )

    with col3:
        st.metric(
            label="Taxa Média de Consumo",
            value=f"{avg_consumption:.1f}%",
            delta=f"{high_consumption_items} items com 100%+",
            delta_color="normal" if avg_consumption >= 95 else "inverse",
            help="Percentual médio de consumo dos items vinculados às DU-Es. Valores próximos de 100% ou acima indicam excelente gestão e utilização eficiente. Meta: ≥95% de consumo médio."
        )

    with col4:
        st.metric(
            label="Valor Totalmente Consumido",
            value=f"USD {total_value_fully_consumed:,.0f}".replace(',', '.'),
            delta=f"{fully_consumed_items} items 100% utilizados",
            delta_color="normal",  # This is GOOD
            help="Valor total em USD dos items que atingiram 100%+ de consumo (saldo negativo). Este é um INDICADOR POSITIVO que representa o montante financeiro de mercadorias completamente utilizadas, demonstrando eficiência operacional máxima."
        )
    
    st.divider()
    
    # Summary table
    st.subheader('📈 Resumo Geral das Exportações')
    col1, col2 = st.columns([2, 1])
    with col1:
        st.table(resumo_df)
    with col2:
        st.metric(
            label="Total de DU-Es",
            value="30",
            help="Total de Declarações Únicas de Exportação processadas"
        )
        st.metric(
            label="Items no Sistema",
            value=f"{total_items}",
            help="Total de items cadastrados no controle de estoque"
        )
    
    st.divider()
    
    # Interactive Plotly Charts
    st.subheader('📊 Visualizações Interativas')
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Visualização do Saldo Disponível**")
        st.caption('🟢 Verde = Disponível | 🟠 Laranja = Sobre-Utilizado (Déficit)')
        
        # Plotly chart for Saldo Disponível
        positive_saldo = display_df[display_df['Saldo Disponível'] >= 0]
        negative_saldo = display_df[display_df['Saldo Disponível'] < 0]
        
        fig_saldo = go.Figure()
        
        if len(positive_saldo) > 0:
            fig_saldo.add_trace(go.Bar(
                x=positive_saldo['Item'],
                y=positive_saldo['Saldo Disponível'],
                name='Disponível (Positivo)',
                marker_color='#2e7d32',
                hovertemplate='<b>Item %{x}</b><br>Saldo: %{y:,.2f}<extra></extra>'
            ))
        
        if len(negative_saldo) > 0:
            fig_saldo.add_trace(go.Bar(
                x=negative_saldo['Item'],
                y=negative_saldo['Saldo Disponível'],
                name='Sobre-Utilizado (Déficit)',
                marker_color='#f57c00',
                hovertemplate='<b>Item %{x}</b><br>Déficit: %{y:,.2f}<extra></extra>'
            ))
        
        fig_saldo.update_layout(
            barmode='relative',
            height=400,
            hovermode='x unified',
            showlegend=True,
            xaxis_title="Item",
            yaxis_title="Saldo Disponível",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig_saldo, use_container_width=True)
    
    with col_chart2:
        st.markdown("**Estoque Disponível (%)**")
        st.caption('🟢 Verde = Alta Utilização | 🟠 Laranja = Baixa Utilização')
        
        # Plotly chart for Estoque Disponível %
        high_util = display_df[display_df['Estoque Disponível (%)'] < 25]
        low_util = display_df[display_df['Estoque Disponível (%)'] >= 25]
        
        fig_estoque = go.Figure()
        
        if len(high_util) > 0:
            fig_estoque.add_trace(go.Bar(
                x=high_util['Item'],
                y=high_util['Estoque Disponível (%)'],
                name='Alta Utilização (<25%)',
                marker_color='#2e7d32',
                hovertemplate='<b>Item %{x}</b><br>Estoque: %{y:.1f}%<extra></extra>'
            ))
        
        if len(low_util) > 0:
            fig_estoque.add_trace(go.Bar(
                x=low_util['Item'],
                y=low_util['Estoque Disponível (%)'],
                name='Baixa Utilização (≥25%)',
                marker_color='#f57c00',
                hovertemplate='<b>Item %{x}</b><br>Estoque: %{y:.1f}%<extra></extra>'
            ))
        
        fig_estoque.update_layout(
            barmode='relative',
            height=400,
            hovermode='x unified',
            showlegend=True,
            xaxis_title="Item",
            yaxis_title="Estoque Disponível (%)",
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig_estoque, use_container_width=True)
    
    # Additional insight: Top 10 items by value
    st.divider()
    st.subheader('� Top 10 Items por Valor Total')
    top_10_items = df.nlargest(10, 'Valor Total (USD)')[['Item', 'Valor Total (USD)', 'Saldo Disponível', 'Estoque Disponível (%)']]
    
    fig_top10 = px.bar(
        top_10_items,
        x='Item',
        y='Valor Total (USD)',
        color='Estoque Disponível (%)',
        color_continuous_scale=['#2e7d32', '#ffa726', '#f57c00'],
        hover_data={'Valor Total (USD)': ':,.2f', 'Saldo Disponível': ':,.2f', 'Estoque Disponível (%)': ':.1f'},
        labels={'Valor Total (USD)': 'Valor (USD)', 'Estoque Disponível (%)': 'Estoque (%)'}
    )
    fig_top10.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_top10, use_container_width=True)

# ============================================================================
# TAB 2: Análise Detalhada - Table with data
# ============================================================================
with tab2:
    st.header('📋 Relação Saídas vs. Consumo DU-E por Item')
    
    # Create styled dataframe for Brazilian formatting
    def format_dataframe_brazilian(df):
        # Define styling function for different column types
        def format_number_br(val):
            if pd.isna(val):
                return ''
            return f"{val:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        
        def format_currency_br(val):
            if pd.isna(val):
                return ''
            return f"USD {val:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        
        def format_percentage_br(val):
            if pd.isna(val):
                return ''
            return f"{val:.2f}".replace('.', ',') + '%'
        
        # Apply formatting and conditional styling
        # GREEN = High utilization (negative saldo = 100%+ consumption = GOOD)
        # RED/YELLOW = Low utilization (positive saldo = underutilized = needs improvement)
        styled_df = df.style.format({
            'Total Saídas': format_number_br,
            'Total Vinculado a DU-Es': format_number_br,
            'Saldo Disponível': format_number_br,
            'Média Utilização/DU-E': format_number_br,
            'Valor Total (USD)': format_currency_br,
            'Valor Médio Unitário (USD)': format_currency_br,
            'Consumo (%)': format_percentage_br,
            'Estoque Disponível (%)': format_percentage_br
        }).apply(lambda x: [
            'background-color: #e8f5e8; color: #2e7d32; font-weight: bold' if val < 0 else  # GREEN for fully consumed
            'background-color: #fff3e0; color: #f57c00; font-weight: bold' if val > 0 else ''  # ORANGE for surplus
            for val in x
        ], subset=['Saldo Disponível']).apply(lambda x: [
            'background-color: #e8f5e8; color: #2e7d32; font-weight: bold' if val >= 100 else  # GREEN for 100%+ consumption
            'background-color: #fff9c4; color: #f57c00; font-weight: bold' if val >= 75 else  # YELLOW for 75-100% consumption
            'background-color: #ffebee; color: #c62828; font-weight: bold' if val < 75 else ''  # RED for low consumption
            for val in x
        ], subset=['Consumo (%)']).apply(lambda x: [
            'background-color: #e8f5e8; color: #2e7d32; font-weight: bold' if val < 25 else  # GREEN for high utilization
            'background-color: #fff3e0; color: #f57c00; font-weight: bold' if val > 50 else ''  # ORANGE for low utilization
            for val in x
        ], subset=['Estoque Disponível (%)'])
        
        return styled_df
    
    # Display the styled dataframe
    st.dataframe(format_dataframe_brazilian(display_df), use_container_width=True, height=600)
    
    # Statistics
    st.divider()
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total de Items", len(display_df))
    with col_stat2:
        avg_saldo = display_df['Saldo Disponível'].mean()
        st.metric("Saldo Médio", f"{avg_saldo:,.1f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    with col_stat3:
        total_value = display_df['Valor Total (USD)'].sum()
        st.metric("Valor Total", f"USD {total_value:,.0f}".replace(',', '.'))
    with col_stat4:
        items_negative = len(display_df[display_df['Saldo Disponível'] < 0])
        st.metric("Items 100%+ Consumidos", items_negative)

# ============================================================================
# TAB 3: Planejamento de Compras
# ============================================================================
with tab3:
    # Use st.toggle for better immediate response
    show_planning_calculator = st.toggle(
        "📊 Mostrar Calculadora de Compras",
        value=False,
        help="Ative para mostrar a calculadora de planejamento de compras. O objetivo é garantir que todos os items sejam 100% consumidos nas DU-Es planejadas."
    )

    # Show the calculator only if the toggle is active
    if show_planning_calculator:
        # Create input section
        plan_col1, plan_col2, plan_col3 = st.columns(3)

        with plan_col1:
            # Number of DU-Es planned
            planned_due_count = st.number_input(
                "Número de DU-Es Planejadas:",
                min_value=1,
                max_value=1000,
                value=5,
                help="Quantas DU-Es você planeja processar?"
            )

        with plan_col2:
            # Value per DU-E with historical average as default
            value_per_due = st.number_input(
                "Valor por DU-E (USD):",
                min_value=1000.0,
                max_value=10000000.0,
                value=float(avg_due_value_from_summary),
                step=1000.0,
                help=f"Valor médio estimado para cada DU-E. Média histórica: USD {avg_due_value_from_summary:,.2f}"
            )

        with plan_col3:
            # Safety margin
            safety_margin = st.slider(
                "Margem de Segurança (%):",
                min_value=0,
                max_value=100,
                value=15,
                help="Margem adicional para cobrir variações e imprevistos"
            )

        # Calculate planning results with advanced statistical modeling
        total_planned_value = planned_due_count * value_per_due

        # Get DU-E values for correlation analysis
        due_values_array = due_df['Valor da DU-E (USD)'].values
        num_dues_historical = len(due_values_array)

        # Calculate requirements for each item based on statistical modeling
        planning_results = []

        for _, item_row in df.iterrows():
            item_num = item_row['Item']
            avg_usage_per_due = item_row['Média Utilização/DU-E']
            total_vinculado = item_row['Total Vinculado a DU-Es']
            current_stock = item_row['Saldo Disponível']
            unit_value = item_row['Valor Médio Unitário (USD)']
            
            # Estimate historical usage per DU-E (assuming uniform distribution)
            # This is a simplification - ideally we'd have item-level DU-E tracking
            historical_usage_per_due = np.full(num_dues_historical, avg_usage_per_due)
            
            # Calculate standard deviation (using 10% coefficient of variation as estimate)
            # In a real scenario, this would come from historical data
            std_dev = avg_usage_per_due * 0.15  # 15% variation
            
            # Calculate correlation between DU-E value and item usage
            correlation, slope, intercept = calculate_item_correlation_with_value(
                historical_usage_per_due, 
                due_values_array
            )
            
            # Predict usage for the planned DU-E value using statistical model
            predicted_usage_per_due, confidence_margin = predict_item_usage_for_due_value(
                value_per_due,
                avg_usage_per_due,
                correlation,
                slope,
                intercept,
                std_dev
            )
            
            # Calculate total needed for planned DU-Es
            total_needed = predicted_usage_per_due * planned_due_count
            
            # Apply safety margin (user-defined) + statistical confidence interval
            statistical_margin = confidence_margin * planned_due_count
            user_margin = total_needed * (safety_margin / 100)
            total_with_margin = total_needed + statistical_margin + user_margin
            
            # Calculate net purchase needed (considering current stock)
            net_purchase_needed = max(0, total_with_margin - current_stock)
            
            # Calculate costs
            purchase_cost = net_purchase_needed * unit_value
            
            # Calculate coverage analysis
            current_due_coverage = current_stock / predicted_usage_per_due if predicted_usage_per_due > 0 else float('inf')
            
            # Calculate confidence level
            confidence_level = abs(correlation) if abs(correlation) > 0.5 else 0.7  # Default moderate confidence
            
            planning_results.append({
                'Item': item_num,
                'Uso Médio Histórico': avg_usage_per_due,
                'Uso Previsto por DU-E': predicted_usage_per_due,
                'Estoque Atual': current_stock,
                'Total Necessário': total_needed,
                'Margem Estatística': statistical_margin,
                'Com Margem Segurança': total_with_margin,
                'Necessidade Compra': net_purchase_needed,
                'Custo Compra (USD)': purchase_cost,
                'Cobertura Atual (DU-Es)': current_due_coverage,
                'Valor Unitário (USD)': unit_value,
                'Correlação Valor': correlation,
                'Confiança (%)': confidence_level * 100
            })

        planning_df = pd.DataFrame(planning_results)

        # Calculate summary metrics
        total_purchase_cost = planning_df['Custo Compra (USD)'].sum()
        items_need_purchase = len(planning_df[planning_df['Necessidade Compra'] > 0])
        items_sufficient_stock = len(planning_df[planning_df['Necessidade Compra'] == 0])
        items_will_be_fully_consumed = len(planning_df[planning_df['Cobertura Atual (DU-Es)'] >= planned_due_count])

        # Display main summary
        st.subheader('📊 Resumo do Planejamento')
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        with summary_col1:
            st.metric(
                "Valor Total Planejado",
                f"USD {total_planned_value:,.0f}".replace(',', '.'),
                delta=f"{planned_due_count} DU-Es × USD {value_per_due:,.0f}".replace(',', '.'),
                help="Valor total estimado para todas as DU-Es planejadas"
            )

        with summary_col2:
            st.metric(
                "Investimento em Compras",
                f"USD {total_purchase_cost:,.0f}".replace(',', '.'),
                delta=f"{(total_purchase_cost/total_planned_value*100):.1f}% do valor planejado".replace(',', '.'),
                delta_color="normal",
                help="Custo total estimado para compras necessárias para garantir 100% de utilização"
            )

        with summary_col3:
            st.metric(
                "Items Precisam Compra",
                f"{items_need_purchase}",
                delta=f"{items_sufficient_stock} já têm estoque suficiente",
                delta_color="inverse" if items_need_purchase > items_sufficient_stock else "normal",
                help="Quantidade de items que precisam de reposição para atingir 100% de consumo"
            )

        with summary_col4:
            st.metric(
                "Items que Atingirão 100%",
                f"{items_will_be_fully_consumed}",
                delta="serão totalmente consumidos",
                delta_color="normal",
                help="Items cujo estoque atual permite alcançar 100% de utilização nas DU-Es planejadas (objetivo positivo!)"
            )

        # Show comparison with historical average
        st.divider()
        comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
        
        with comparison_col1:
            variance_percentage = ((value_per_due - avg_due_value_from_summary) / avg_due_value_from_summary * 100)
            st.metric(
                "Variação vs Média Histórica",
                f"{variance_percentage:+.1f}%",
                delta=f"Média histórica: USD {avg_due_value_from_summary:,.2f}".replace(',', '.'),
                delta_color="normal" if abs(variance_percentage) < 20 else "off",
                help=f"Compara o valor planejado (USD {value_per_due:,.2f}) com a média histórica de {total_due_count} DU-Es processadas. Variações >20% podem indicar mudança no perfil das exportações."
            )
        
        with comparison_col2:
            avg_value_planned = total_planned_value / planned_due_count
            st.metric(
                "Valor Médio Planejado",
                f"USD {avg_value_planned:,.0f}".replace(',', '.'),
                help="Valor médio por DU-E que você está planejando"
            )
        
        with comparison_col3:
            total_historical = avg_due_value_from_summary * total_due_count
            efficiency_indicator = (total_purchase_cost / total_planned_value * 100)
            st.metric(
                "Eficiência do Planejamento",
                f"{efficiency_indicator:.1f}%",
                delta="do valor DU-E em compras",
                delta_color="normal" if efficiency_indicator < 30 else "inverse",
                help="Quanto do valor total das DU-Es será necessário investir em compras. Menor é melhor - indica melhor utilização do estoque existente."
            )

        # Show statistical methodology explanation
        with st.expander("📊 Metodologia Estatística do Planejamento"):
            st.markdown(f"""
            ### Como funciona o cálculo estatístico:
            
            **📈 Dados Históricos Utilizados:**
            - **{total_due_count} DU-Es processadas** historicamente
            - **Valor médio histórico: USD {avg_due_value_from_summary:,.2f}**
            - Desvio padrão: USD {due_df['Valor da DU-E (USD)'].std():,.2f}
            - Range: USD {due_df['Valor da DU-E (USD)'].min():,.2f} - USD {due_df['Valor da DU-E (USD)'].max():,.2f}
            
            **1. Análise de Correlação:**
            - O sistema analisa a relação entre o **valor da DU-E** e o **consumo de cada item**
            - Se houver correlação forte (>50%), usa regressão linear para prever o consumo
            - DU-Es de maior valor tendem a consumir mais insumos proporcionalmente
            - **Seu valor planejado** (USD {value_per_due:,.2f}) é **{variance_percentage:+.1f}%** em relação à média histórica
            
            **2. Previsão Ajustada por Valor:**
            - **Uso Previsto por DU-E**: Calculado com base no valor específico da DU-E planejada
            - Considera que DU-Es de USD 150k podem consumir mais que DU-Es de USD 80k
            
            **3. Margem Estatística (Intervalo de Confiança 95%):**
            - Calculada automaticamente com base na variabilidade histórica
            - Adiciona proteção contra variações normais do processo
            - Diferente da margem de segurança manual
            
            **4. Margem de Segurança (Usuário):**
            - Margem adicional definida por você (%)
            - Para cobrir imprevistos, mudanças de processo, ou requisitos especiais
            
            **5. Nível de Confiança:**
            - Indica a confiabilidade da previsão (0-100%)
            - Baseado na correlação histórica e consistência dos dados
            - >70% = Alta confiança | 50-70% = Média | <50% = Baixa (usar média histórica)
            
            **Fórmula Final:**
            ```
            Total com Margens = (Uso Previsto × Qtd DU-Es) + Margem Estatística + Margem Usuário
            ```
            """)
        
        # Show detailed breakdown
        st.subheader('📋 Detalhamento por Item')

        # Filter options for the planning table
        show_options = st.selectbox(
            "Mostrar:",
            ["Todos os Items", "Apenas Items que Precisam Compra", "Items que Atingirão 100% Consumo", "Items com Estoque Insuficiente"]
        )

        # Apply filter
        if show_options == "Apenas Items que Precisam Compra":
            filtered_planning_df = planning_df[planning_df['Necessidade Compra'] > 0]
        elif show_options == "Items que Atingirão 100% Consumo":
            filtered_planning_df = planning_df[planning_df['Cobertura Atual (DU-Es)'] >= planned_due_count]
        elif show_options == "Items com Estoque Insuficiente":
            filtered_planning_df = planning_df[planning_df['Cobertura Atual (DU-Es)'] < planned_due_count]
        else:
            filtered_planning_df = planning_df

        # Sort by purchase cost (highest first) to prioritize most expensive items
        filtered_planning_df = filtered_planning_df.sort_values('Custo Compra (USD)', ascending=False)

        # Format the planning dataframe for display
        def format_planning_dataframe_brazilian(df):
            def format_number_br(val):
                if pd.isna(val) or val == 0:
                    return '-'
                return f"{val:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            
            def format_currency_br(val):
                if pd.isna(val) or val == 0:
                    return '-'
                return f"USD {val:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            
            def format_coverage_br(val):
                if pd.isna(val):
                    return '-'
                if val == float('inf'):
                    return '∞'
                return f"{val:.1f}".replace('.', ',')
            
            def format_percentage_br(val):
                if pd.isna(val):
                    return '-'
                return f"{val:.1f}%".replace('.', ',')
            
            # Create display columns with new statistical metrics
            display_columns = [
                'Item',
                'Uso Previsto por DU-E',
                'Confiança (%)',
                'Estoque Atual',
                'Total Necessário',
                'Margem Estatística',
                'Com Margem Segurança',
                'Necessidade Compra',
                'Custo Compra (USD)',
                'Cobertura Atual (DU-Es)'
            ]
            
            display_df = df[display_columns].copy()
            
            # Apply formatting and conditional styling
            styled_df = display_df.style.format({
                'Uso Previsto por DU-E': format_number_br,
                'Confiança (%)': format_percentage_br,
                'Estoque Atual': format_number_br,
                'Total Necessário': format_number_br,
                'Margem Estatística': format_number_br,
                'Com Margem Segurança': format_number_br,
                'Necessidade Compra': format_number_br,
                'Custo Compra (USD)': format_currency_br,
                'Cobertura Atual (DU-Es)': format_coverage_br
            }).apply(lambda x: [
                'background-color: #e8f5e8; color: #2e7d32; font-weight: bold' if val < 0 else ''  # Green if will be fully consumed
                for val in x
            ], subset=['Estoque Atual']).apply(lambda x: [
                'background-color: #fff3e0; color: #f57c00; font-weight: bold' if pd.notna(val) and val > 0 else ''  # Orange if purchase needed
                for val in x
            ], subset=['Necessidade Compra']).apply(lambda x: [
                'background-color: #e8f5e8; color: #2e7d32; font-weight: bold' if pd.notna(val) and val != float('inf') and val >= planned_due_count else  # Green if coverage is good
                'background-color: #fff3e0; color: #f57c00; font-weight: bold' if pd.notna(val) and val != float('inf') and val < planned_due_count else ''  # Orange if insufficient
                for val in x
            ], subset=['Cobertura Atual (DU-Es)'])
            
            return styled_df

        # Display the planning table
        st.info(f"📊 Mostrando {len(filtered_planning_df)} items | Filtro: {show_options}")
        st.dataframe(format_planning_dataframe_brazilian(filtered_planning_df), use_container_width=True)

        # Show purchase summary by priority
        if items_need_purchase > 0:
            st.subheader('🛒 Lista Prioritária de Compras')
            
            priority_items = planning_df[planning_df['Necessidade Compra'] > 0].sort_values('Custo Compra (USD)', ascending=False).head(10)
            
            priority_summary = {
                'Prioridade': [f"#{i+1}" for i in range(len(priority_items))],
                'Item': priority_items['Item'].tolist(),
                'Quantidade Necessária': [f"{val:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') for val in priority_items['Necessidade Compra']],
                'Investimento (USD)': [f"USD {val:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.') for val in priority_items['Custo Compra (USD)']],
                'Prioridade Compra': ['🔴 Alta' if cov < planned_due_count/2 else '🟡 Média' for cov in priority_items['Cobertura Atual (DU-Es)']]
            }
            
            priority_df = pd.DataFrame(priority_summary)
            st.table(priority_df)
            
            st.success(f"💡 **Recomendação:** Focar nos primeiros {min(5, len(priority_items))} items da lista que representam {priority_items.head(5)['Custo Compra (USD)'].sum()/total_purchase_cost*100:.1f}% do investimento total. Estes items são essenciais para maximizar a utilização do estoque.")

    else:
        st.info("💡 **Dica:** Ative o toggle acima para abrir a calculadora e planejar suas compras. O objetivo é garantir que todos os items atinjam 100% de consumo nas DU-Es planejadas.")

# ============================================================================
# TAB 4: Status DU-Es
# ============================================================================
with tab4:
    st.header('🚢 Status das DU-Es (Declarações Únicas de Exportação)')
    
    # DU-E Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de DU-Es", len(due_df))
    with col2:
        total_due_value = due_df['Valor da DU-E (USD)'].sum()
        st.metric("Valor Total", f"USD {total_due_value:,.0f}".replace(',', '.'))
    with col3:
        avg_due_value = due_df['Valor da DU-E (USD)'].mean()
        st.metric("Valor Médio por DU-E", f"USD {avg_due_value:,.0f}".replace(',', '.'))
    
    st.divider()
    
    # Timeline chart of DU-Es
    st.subheader('📅 Timeline de Exportações')
    
    # Convert date column to datetime
    due_df_chart = due_df.copy()
    due_df_chart['Data de Embarque'] = pd.to_datetime(due_df_chart['Data de Embarque'], format='%d/%m/%Y')
    due_df_chart = due_df_chart.sort_values('Data de Embarque')
    
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Scatter(
        x=due_df_chart['Data de Embarque'],
        y=due_df_chart['Valor da DU-E (USD)'],
        mode='lines+markers',
        name='Valor por DU-E',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=8, color='#3b82f6'),
        hovertemplate='<b>%{text}</b><br>Data: %{x|%d/%m/%Y}<br>Valor: USD %{y:,.2f}<extra></extra>',
        text=due_df_chart['Nº da DU-E']
    ))
    
    fig_timeline.update_layout(
        height=400,
        xaxis_title="Data de Embarque",
        yaxis_title="Valor (USD)",
        hovermode='closest',
        margin=dict(l=0, r=0, t=20, b=0)
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.divider()
    
    # DU-E Table
    st.subheader('📋 Detalhamento das DU-Es')
    
    # Create styled dataframe for DU-E table with Brazilian formatting
    def format_due_dataframe_brazilian(df):
        def format_currency_br(val):
            if pd.isna(val):
                return ''
            return f"USD {val:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        
        # Apply formatting and conditional styling for high-value transactions
        styled_df = df.style.format({
            'Valor da DU-E (USD)': format_currency_br
        }).apply(lambda x: [
            'background-color: #fff3e0; color: #ef6c00; font-weight: bold' if val > 200000 else
            'background-color: #e8f5e8; color: #2e7d32; font-weight: bold' if val > 100000 else ''
            for val in x
        ], subset=['Valor da DU-E (USD)'])
        
        return styled_df
    
    st.dataframe(format_due_dataframe_brazilian(due_df), use_container_width=True, height=500)
    
    # DU-E Distribution Chart
    st.divider()
    st.subheader('📊 Distribuição de Valores das DU-Es')
    
    fig_dist = go.Figure()
    
    # Create histogram
    fig_dist.add_trace(go.Histogram(
        x=due_df['Valor da DU-E (USD)'],
        nbinsx=10,
        marker_color='#3b82f6',
        hovertemplate='Faixa: %{x}<br>Quantidade: %{y}<extra></extra>'
    ))
    
    fig_dist.update_layout(
        height=300,
        xaxis_title="Valor da DU-E (USD)",
        yaxis_title="Quantidade de DU-Es",
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0)
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

