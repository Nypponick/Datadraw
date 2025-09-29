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

# Calculate Estoque Disponível (%) based on the Saldo Disponível and Total Saídas
df['Estoque Disponível (%)'] = (df['Saldo Disponível'] / df['Total Saídas']) * 100

# No need for display formatting - we'll use column_config instead


# Read the DU-E status table (header at row 66, data starts at row 67)
due_df = pd.read_csv(file_path, skiprows=65, header=0)
due_df = due_df.dropna(how='all', axis=1)
due_df = due_df.dropna(how='all', axis=0)
# Remove unnamed columns (empty first column)
due_df = due_df.loc[:, ~due_df.columns.str.contains('^Unnamed')]
due_df = due_df.rename(columns={'Nº da DU-E': 'Nº da DU-E', 'Data de Embarque': 'Data de Embarque',
                                'Valor da DU-E (USD)': 'Valor da DU-E (USD)', 'Status': 'Status'})
due_df['Valor da DU-E (USD)'] = due_df['Valor da DU-E (USD)'].apply(clean_currency)

# No need for display formatting - we'll use column_config instead


# Streamlit App
st.set_page_config(layout="wide")

st.title('Dashboard de Controle de Estoque')

# Calculate KPIs
fully_consumed_items = len(df[df['Saldo Disponível'] < 0])  # 100%+ utilization = GOOD
underutilized_items = len(df[df['Saldo Disponível'] >= 0])  # Surplus = need better planning
total_items = len(df)

# Items with high utilization (>75%) are performing well
high_utilization_items = len(df[df['Estoque Disponível (%)'] < 25])  # Less than 25% remaining = >75% used
high_utilization_percentage = (high_utilization_items / total_items * 100) if total_items > 0 else 0

avg_stock_utilization = 100 - df['Estoque Disponível (%)'].mean()

# Calculate total value of fully consumed items (this is GOOD - shows efficient use)
fully_consumed_df = df[df['Saldo Disponível'] < 0]
total_value_fully_consumed = fully_consumed_df['Valor Total (USD)'].sum() if len(fully_consumed_df) > 0 else 0

# Display KPIs in a nice layout
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
        label="Taxa Média de Utilização",
        value=f"{avg_stock_utilization:.1f}%",
        delta="Média de utilização do estoque",
        delta_color="normal" if avg_stock_utilization >= 75 else "inverse",
        help="Indica o percentual médio de utilização do estoque disponível. Valores altos (>75%) demonstram eficiência na utilização e consumo adequado. Valores baixos indicam subutilização e necessidade de ajustar o planejamento de compras."
    )

with col4:
    st.metric(
        label="Valor Totalmente Consumido",
        value=f"USD {total_value_fully_consumed:,.0f}".replace(',', '.'),
        delta=f"{fully_consumed_items} items 100% utilizados",
        delta_color="normal",  # This is GOOD
        help="Valor total em USD dos items que atingiram 100%+ de consumo (saldo negativo). Este é um INDICADOR POSITIVO que representa o montante financeiro de mercadorias completamente utilizadas, demonstrando eficiência operacional máxima."
    )

# Add Purchase Planning Calculator with toggle
st.header('🎯 Planejamento de Compras para DU-Es')

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
        # Value per DU-E
        value_per_due = st.number_input(
            "Valor por DU-E (USD):",
            min_value=1000.0,
            max_value=10000000.0,
            value=120000.0,
            step=1000.0,
            help="Valor médio estimado para cada DU-E"
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

    # Calculate planning results
    total_planned_value = planned_due_count * value_per_due

    # Calculate requirements for each item based on historical usage patterns
    planning_results = []

    for _, item_row in df.iterrows():
        item_num = item_row['Item']
        avg_usage_per_due = item_row['Média Utilização/DU-E']
        current_stock = item_row['Saldo Disponível']
        unit_value = item_row['Valor Médio Unitário (USD)']
        
        # Calculate total needed for planned DU-Es
        total_needed = avg_usage_per_due * planned_due_count
        
        # Apply safety margin
        total_with_margin = total_needed * (1 + safety_margin / 100)
        
        # Calculate net purchase needed (considering current stock)
        net_purchase_needed = max(0, total_with_margin - current_stock)
        
        # Calculate costs
        purchase_cost = net_purchase_needed * unit_value
        
        # Calculate coverage analysis
        current_due_coverage = current_stock / avg_usage_per_due if avg_usage_per_due > 0 else float('inf')
        
        planning_results.append({
            'Item': item_num,
            'Uso Médio por DU-E': avg_usage_per_due,
            'Estoque Atual': current_stock,
            'Total Necessário': total_needed,
            'Com Margem Segurança': total_with_margin,
            'Necessidade Compra': net_purchase_needed,
            'Custo Compra (USD)': purchase_cost,
            'Cobertura Atual (DU-Es)': current_due_coverage,
            'Valor Unitário (USD)': unit_value
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
        
        # Create display columns
        display_columns = [
            'Item',
            'Uso Médio por DU-E',
            'Estoque Atual',
            'Total Necessário',
            'Com Margem Segurança',
            'Necessidade Compra',
            'Custo Compra (USD)',
            'Cobertura Atual (DU-Es)'
        ]
        
        display_df = df[display_columns].copy()
        
        # Apply formatting and conditional styling
        styled_df = display_df.style.format({
            'Uso Médio por DU-E': format_number_br,
            'Estoque Atual': format_number_br,
            'Total Necessário': format_number_br,
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

st.divider()

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
        ["Todos os Items", "Items 100%+ Utilizados", "Alta Utilização (>75%)", "Alto Valor (>USD 100k)", "Items Performáticos"]
    )

with filter_col2:
    # Search by item number
    search_item = st.text_input("Buscar por Item:", placeholder="Ex: 1, 2, 3...")

with filter_col3:
    # Sort options
    sort_by = st.selectbox(
        "Ordenar por:",
        ["Item", "Saldo Disponível", "Utilização (%)", "Valor Total", "Performance"]
    )

with filter_col4:
    # Sort order
    sort_order = st.selectbox("Ordem:", ["Crescente", "Decrescente"])

# Apply filters
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
elif sort_by == "Utilização (%)":
    # Sort by utilization (inverse of Estoque Disponível %)
    filtered_df_with_score = filtered_df_with_score.sort_values('Estoque Disponível (%)', ascending=not ascending)
elif sort_by == "Valor Total":
    filtered_df_with_score = filtered_df_with_score.sort_values('Valor Total (USD)', ascending=ascending)
elif sort_by == "Performance":
    filtered_df_with_score = filtered_df_with_score.sort_values('Performance', ascending=False)  # Always show best performers first

# Remove the performance score column from display
display_df = filtered_df_with_score.drop('Performance', axis=1)

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
    # GREEN = High utilization (negative saldo = 100%+ consumption = GOOD)
    # RED/YELLOW = Low utilization (positive saldo = underutilized = needs improvement)
    styled_df = df.style.format({
        'Total Saídas': format_number_br,
        'Total Vinculado a DU-Es': format_number_br,
        'Saldo Disponível': format_number_br,
        'Média Utilização/DU-E': format_number_br,
        'Valor Total (USD)': format_currency_br,
        'Valor Médio Unitário (USD)': format_currency_br,
        'Estoque Disponível (%)': format_percentage_br
    }).apply(lambda x: [
        'background-color: #e8f5e8; color: #2e7d32; font-weight: bold' if val < 0 else  # GREEN for fully consumed
        'background-color: #fff3e0; color: #f57c00; font-weight: bold' if val > 0 else ''  # ORANGE for surplus
        for val in x
    ], subset=['Saldo Disponível']).apply(lambda x: [
        'background-color: #e8f5e8; color: #2e7d32; font-weight: bold' if val < 25 else  # GREEN for high utilization
        'background-color: #fff3e0; color: #f57c00; font-weight: bold' if val > 50 else ''  # ORANGE for low utilization
        for val in x
    ], subset=['Estoque Disponível (%)'])
    
    return styled_df

# Display the styled dataframe (sorting will work on original numeric values)
st.dataframe(format_dataframe_brazilian(display_df), use_container_width=True)

st.header('Visualização do Saldo Disponível')
st.caption('🟢 Verde = Estoque Disponível (Dentro dos Limites) | 🟠 Laranja = Sobre-Utilizado (Déficit - RISCO)')
# Create data with color coding for positive/negative values (use filtered data)
saldo_chart_data = display_df.set_index('Item')['Saldo Disponível'].to_frame()
saldo_chart_data['Disponível (Positivo)'] = saldo_chart_data['Saldo Disponível'].where(saldo_chart_data['Saldo Disponível'] >= 0, 0)
saldo_chart_data['Sobre-Utilizado (Déficit)'] = saldo_chart_data['Saldo Disponível'].where(saldo_chart_data['Saldo Disponível'] < 0, 0)
saldo_chart_data = saldo_chart_data[['Disponível (Positivo)', 'Sobre-Utilizado (Déficit)']]
st.bar_chart(saldo_chart_data, color=['#2e7d32', '#f57c00'])

st.header('Visualização do Estoque Disponível (%)')
st.caption('🟢 Verde = Alta Utilização (<25% restante - BOM) | 🟠 Laranja = Baixa Utilização (>25% restante - Melhorar)')
# Create data with color coding based on utilization level (use filtered data)
estoque_chart_data = display_df.set_index('Item')['Estoque Disponível (%)'].to_frame()
estoque_chart_data['Alta Utilização (<25%)'] = estoque_chart_data['Estoque Disponível (%)'].where(estoque_chart_data['Estoque Disponível (%)'] < 25, 0)
estoque_chart_data['Baixa Utilização (≥25%)'] = estoque_chart_data['Estoque Disponível (%)'].where(estoque_chart_data['Estoque Disponível (%)'] >= 25, 0)
estoque_chart_data = estoque_chart_data[['Alta Utilização (<25%)', 'Baixa Utilização (≥25%)']]
st.bar_chart(estoque_chart_data, color=['#2e7d32', '#f57c00'])


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
