# 📊 Dashboard de Controle de Estoque - Drawback Kilbra

Dashboard interativo para controle de estoque e análise de exportações com sistema de drawback.

## ✨ Características

- **📊 KPIs Visuais**: Indicadores chave de performance com tooltips explicativos
- **🔍 Filtros Inteligentes**: Busca e filtros por déficit, estoque baixo, alto valor
- **🎨 Formatação Brasileira**: Números formatados no padrão brasileiro
- **📈 Gráficos Interativos**: Visualizações com cores condicionais
- **🚨 Alertas Visuais**: Highlighting automático de valores negativos e críticos

## 🚀 Como Usar

1. Acesse o dashboard através do link de deploy
2. Use os filtros na seção "🔍 Filtros e Busca" para encontrar items específicos
3. Passe o mouse sobre os KPIs para ver explicações detalhadas
4. Analise os gráficos para identificar tendências e problemas

## 📋 KPIs Disponíveis

- **Items em Déficit vs. Superávit**: Contagem de items com saldo negativo/positivo
- **% Items com Estoque Saudável**: Percentual de items com estoque >= 25%
- **Taxa Média de Utilização**: Eficiência média de uso do estoque
- **Valor em Risco**: Valor total USD dos items em déficit

## 🔧 Tecnologias

- **Streamlit**: Framework para aplicações web
- **Pandas**: Manipulação e análise de dados
- **Python**: Linguagem de programação

## 📊 Estrutura dos Dados

O dashboard processa dados de:
- Resumo geral das exportações
- Relação saídas vs. consumo DU-E por item
- Status das DU-Es processadas

---
*Desenvolvido para controle eficiente de estoque e análise de drawback*