import streamlit as st
import pandas as pd
import numpy as np

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
file_path = 'd:/Usuários/User/Área de Trabalho/dados drawback_Kilbra/Testes_Fechamento - Dashboard.csv'

# Read the summary table
resumo_df = pd.read_csv(file_path, skiprows=3, nrows=3, usecols=[1, 2], header=None, names=['Metrica', 'Valor'])

# Read the main data table
df = pd.read_csv(file_path, skiprows=9, header=0, nrows=55)
df = df.dropna(how='all', axis=1)
df = df.dropna(how='all', axis=0)

# Keep original column names for display but create a mapping for easier reference
original_columns = df.columns.tolist()
column_mapping = {
    'Nº do Item': 'Item',
    'Total Saídas (Quantidade)': 'Total Saídas',
    'Valor Total (USD)': 'Valor Total (USD)',
    'Total Vinculado a DU-Es (Quantidade)': 'Total Vinculado a DU-Es',
    'Saldo Disponível': 'Saldo Disponível',
    'Valor Médio Unitário (USD)': 'Valor Médio Unitário (USD)',
    'Média Utilização/DU-E': 'Média Utilização/DU-E',
    'Estoque disponível em %': 'Estoque Disponível (%)',
    'Coluna 1': 'Coluna 1'
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
    'Média Utilização/DU-E': 'Media_Utilizacao_DU_E',
    'Estoque Disponível (%)': 'Estoque_Disponivel_Percent',
    'Coluna 1': 'Coluna_1'
}

# Clean the data using the display column names
for col in ['Total Saídas', 'Total Vinculado a DU-Es', 'Saldo Disponível', 'Média Utilização/DU-E']:
    df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)

for col in ['Valor Total (USD)', 'Valor Médio Unitário (USD)']:
    df[col] = df[col].apply(clean_currency)

df['Estoque Disponível (%)'] = df['Estoque Disponível (%)'].apply(clean_percentage)

# No need for display formatting - we'll use column_config instead


# Read the DU-E status table
due_df = pd.read_csv(file_path, skiprows=66, header=0)
due_df = due_df.dropna(how='all', axis=1)
due_df = due_df.dropna(how='all', axis=0)
due_df = due_df.rename(columns={'Nº da DU-E': 'Nº da DU-E', 'Data de Embarque': 'Data de Embarque',
                                'Valor da DU-E (USD)': 'Valor da DU-E (USD)', 'Status': 'Status'})
due_df['Valor da DU-E (USD)'] = due_df['Valor da DU-E (USD)'].apply(clean_currency)

# No need for display formatting - we'll use column_config instead


# Streamlit App
st.set_page_config(layout="wide")

st.title('Dashboard de Controle de Estoque')

# Calculate KPIs
deficit_items = len(df[df['Saldo Disponível'] < 0])
surplus_items = len(df[df['Saldo Disponível'] >= 0])
total_items = len(df)

healthy_stock_items = len(df[df['Estoque Disponível (%)'] >= 25])
healthy_stock_percentage = (healthy_stock_items / total_items * 100) if total_items > 0 else 0

avg_stock_utilization = 100 - df['Estoque Disponível (%)'].mean()

deficit_df = df[df['Saldo Disponível'] < 0]
total_value_at_risk = deficit_df['Valor Total (USD)'].sum() if len(deficit_df) > 0 else 0

# Display KPIs in a nice layout
st.header('📊 Indicadores Chave de Performance (KPIs)')
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Items em Déficit vs. Superávit",
        value=f"{deficit_items} / {surplus_items}",
        delta=f"{deficit_items} em déficit",
        delta_color="inverse",
        help="Mostra quantos items têm saldo negativo (déficit) versus saldo positivo (superávit). Items em déficit indicam que mais mercadoria foi utilizada nas DU-Es do que o disponível em estoque, representando um risco operacional."
    )

with col2:
    st.metric(
        label="% Items com Estoque Saudável",
        value=f"{healthy_stock_percentage:.1f}%",
        delta=f"{healthy_stock_items} de {total_items} items",
        delta_color="normal" if healthy_stock_percentage >= 70 else "inverse",
        help="Percentual de items com estoque disponível igual ou superior a 25%. Um estoque saudável garante margem de segurança para futuras exportações. Valores abaixo de 70% indicam necessidade de atenção ao planejamento de estoque."
    )

with col3:
    st.metric(
        label="Taxa Média de Utilização",
        value=f"{avg_stock_utilization:.1f}%",
        delta="Média de utilização do estoque",
        delta_color="normal" if avg_stock_utilization <= 80 else "inverse",
        help="Indica o percentual médio de utilização do estoque disponível (100% - estoque disponível médio). Valores altos (>80%) sugerem eficiência na utilização, mas também podem indicar risco de desabastecimento. Valores muito baixos podem indicar excesso de estoque."
    )

with col4:
    st.metric(
        label="Valor em Risco (Déficit)",
        value=f"USD {total_value_at_risk:,.0f}".replace(',', '.'),
        delta=f"{deficit_items} items em déficit",
        delta_color="inverse" if total_value_at_risk > 0 else "normal",
        help="Valor total em USD dos items que estão em situação de déficit (saldo negativo). Este valor representa o montante financeiro potencialmente em risco devido à utilização excessiva do estoque em relação ao disponível. Valores altos requerem atenção imediata para reposição."
    )

st.header('Resumo Geral das Exportações')
st.table(resumo_df)

st.header('Relação Saídas vs. Consumo DU-E por Item')

# 🔍 Smart Filtering & Search
st.subheader('🔍 Filtros e Busca')

# Create filter columns
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    # Quick filters
    filter_type = st.selectbox(
        "Filtro Rápido:",
        ["Todos os Items", "Items em Déficit", "Estoque Baixo (<25%)", "Alto Valor (>USD 100k)", "Críticos"]
    )

with filter_col2:
    # Search by item number
    search_item = st.text_input("Buscar por Item:", placeholder="Ex: 1, 2, 3...")

with filter_col3:
    # Sort options
    sort_by = st.selectbox(
        "Ordenar por:",
        ["Item", "Saldo Disponível", "Estoque (%)", "Valor Total", "Criticidade"]
    )

with filter_col4:
    # Sort order
    sort_order = st.selectbox("Ordem:", ["Crescente", "Decrescente"])

# Apply filters
filtered_df = df.copy()

# Apply quick filters
if filter_type == "Items em Déficit":
    filtered_df = filtered_df[filtered_df['Saldo Disponível'] < 0]
elif filter_type == "Estoque Baixo (<25%)":
    filtered_df = filtered_df[filtered_df['Estoque Disponível (%)'] < 25]
elif filter_type == "Alto Valor (>USD 100k)":
    filtered_df = filtered_df[filtered_df['Valor Total (USD)'] > 100000]
elif filter_type == "Críticos":
    filtered_df = filtered_df[
        (filtered_df['Saldo Disponível'] < 0) | 
        (filtered_df['Estoque Disponível (%)'] < 10)
    ]

# Apply item search
if search_item:
    try:
        search_items = [int(x.strip()) for x in search_item.split(',') if x.strip()]
        filtered_df = filtered_df[filtered_df['Item'].isin(search_items)]
    except ValueError:
        st.error("Por favor, digite números de items válidos separados por vírgula.")

# Add criticality score for sorting
filtered_df_with_score = filtered_df.copy()
filtered_df_with_score['Criticidade'] = (
    (filtered_df['Saldo Disponível'] < 0).astype(int) * 3 +  # Deficit items get score 3
    (filtered_df['Estoque Disponível (%)'] < 10).astype(int) * 2 +  # Very low stock gets score 2
    (filtered_df['Estoque Disponível (%)'] < 25).astype(int) * 1     # Low stock gets score 1
)

# Apply sorting
ascending = sort_order == "Crescente"
if sort_by == "Item":
    filtered_df_with_score = filtered_df_with_score.sort_values('Item', ascending=ascending)
elif sort_by == "Saldo Disponível":
    filtered_df_with_score = filtered_df_with_score.sort_values('Saldo Disponível', ascending=ascending)
elif sort_by == "Estoque (%)":
    filtered_df_with_score = filtered_df_with_score.sort_values('Estoque Disponível (%)', ascending=ascending)
elif sort_by == "Valor Total":
    filtered_df_with_score = filtered_df_with_score.sort_values('Valor Total (USD)', ascending=ascending)
elif sort_by == "Criticidade":
    filtered_df_with_score = filtered_df_with_score.sort_values('Criticidade', ascending=False)  # Always show most critical first

# Remove the criticality score column from display
display_df = filtered_df_with_score.drop('Criticidade', axis=1)

# Show filter results
total_filtered = len(display_df)
st.info(f"📊 Mostrando {total_filtered} de {len(df)} items | Filtro: {filter_type}")

# Create styled dataframe for Brazilian formatting while keeping data sortable
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
    styled_df = df.style.format({
        'Total Saídas': format_number_br,
        'Total Vinculado a DU-Es': format_number_br,
        'Saldo Disponível': format_number_br,
        'Média Utilização/DU-E': format_number_br,
        'Valor Total (USD)': format_currency_br,
        'Valor Médio Unitário (USD)': format_currency_br,
        'Estoque Disponível (%)': format_percentage_br
    }).apply(lambda x: [
        'background-color: #ffebee; color: #c62828; font-weight: bold' if val < 0 else ''
        for val in x
    ], subset=['Saldo Disponível']).apply(lambda x: [
        'background-color: #ffebee; color: #c62828; font-weight: bold' if val < 0 else 
        'background-color: #e8f5e8; color: #2e7d32; font-weight: bold' if val > 50 else ''
        for val in x
    ], subset=['Estoque Disponível (%)'])
    
    return styled_df

# Display the styled dataframe (sorting will work on original numeric values)
st.dataframe(format_dataframe_brazilian(display_df), use_container_width=True)

st.header('Visualização do Saldo Disponível')
# Create data with color coding for positive/negative values (use filtered data)
saldo_chart_data = display_df.set_index('Item')['Saldo Disponível'].to_frame()
saldo_chart_data['Positivo'] = saldo_chart_data['Saldo Disponível'].where(saldo_chart_data['Saldo Disponível'] >= 0, 0)
saldo_chart_data['Negativo'] = saldo_chart_data['Saldo Disponível'].where(saldo_chart_data['Saldo Disponível'] < 0, 0)
saldo_chart_data = saldo_chart_data[['Positivo', 'Negativo']]
st.bar_chart(saldo_chart_data, color=['#2e7d32', '#c62828'])

st.header('Visualização do Estoque Disponível (%)')
# Create data with color coding for negative percentages (use filtered data)
estoque_chart_data = display_df.set_index('Item')['Estoque Disponível (%)'].to_frame()
estoque_chart_data['Positivo'] = estoque_chart_data['Estoque Disponível (%)'].where(estoque_chart_data['Estoque Disponível (%)'] >= 0, 0)
estoque_chart_data['Negativo'] = estoque_chart_data['Estoque Disponível (%)'].where(estoque_chart_data['Estoque Disponível (%)'] < 0, 0)
estoque_chart_data = estoque_chart_data[['Positivo', 'Negativo']]
st.bar_chart(estoque_chart_data, color=['#2e7d32', '#c62828'])


st.header('Status das DU-Es')

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

st.dataframe(format_due_dataframe_brazilian(due_df), use_container_width=True)
