"""
Generador de Dashboards PNG para S&OP
=====================================
Crea visualizaciones de ejemplo del sistema S&OP integrado.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Agregar src al path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from sop_optimizer import SOPOptimizer
from config_loader import load_config

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def create_sample_forecast():
    """Crea forecast de ejemplo para demostrar S&OP"""
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=26, freq='W')
    
    # Crear forecast para 3 SKUs
    skus = ['DEMAND', 'PRODUCT_A', 'PRODUCT_B']
    base_demand = {'DEMAND': 200, 'PRODUCT_A': 150, 'PRODUCT_B': 180}
    
    records = []
    np.random.seed(42)
    
    for sku in skus:
        base = base_demand[sku]
        for i, date in enumerate(dates):
            # Tendencia + estacionalidad + ruido
            trend = base * (1 + 0.002 * i)
            seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * i / 26)
            noise = np.random.normal(1, 0.1)
            forecast = max(0, trend * seasonal * noise)
            
            records.append({
                'SKU': sku,
                'Date': date,
                'Forecast': round(forecast, 2)
            })
    
    return pd.DataFrame(records)

def generate_dashboards():
    """Genera todos los dashboards PNG"""
    print("\n" + "="*70)
    print("GENERANDO DASHBOARDS S&OP")
    print("="*70)
    
    # Cargar configuracion
    config = load_config("config.yaml")
    
    # Crear forecast de ejemplo
    forecast_df = create_sample_forecast()
    print(f"[OK] Forecast de ejemplo creado: {len(forecast_df)} registros")
    
    # Inicializar optimizador S&OP
    sop = SOPOptimizer(config, service_level=0.95)
    
    # Cargar datos S&OP
    sop.load_inventory_policies('data/inventory_policies.xlsx')
    sop.load_supply_plan('data/supply_plan.xlsx')
    
    # Simular inventario
    print("[*] Ejecutando simulacion S&OP...")
    results = sop.simulate_inventory(forecast_df)
    
    # Crear directorios
    output_dir = Path('outputs/sop_dashboards')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar dashboards
    print("[*] Generando dashboards...")
    
    # 1. Dashboard resumen
    create_summary_dashboard(sop, output_dir)
    
    # 2. Dashboard por SKU
    for sku in results['SKU'].unique():
        create_sku_dashboard(sop, sku, output_dir)
    
    # 3. Dashboard de metricas
    create_metrics_dashboard(sop, output_dir)
    
    print(f"\n[OK] Dashboards generados en: {output_dir}")
    print("="*70)

def create_summary_dashboard(sop, output_dir):
    """Dashboard resumen principal"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Nivel de servicio por SKU
    ax1 = axes[0, 0]
    colors = ['green' if x >= 95 else 'orange' if x >= 90 else 'red' 
             for x in sop.metrics_df['Service_Level_Pct']]
    bars = ax1.barh(sop.metrics_df['SKU'], sop.metrics_df['Service_Level_Pct'], 
                    color=colors, alpha=0.8)
    ax1.axvline(x=95, color='green', linestyle='--', linewidth=2, label='Meta 95%')
    ax1.set_xlabel('Nivel de Servicio (%)', fontsize=12)
    ax1.set_ylabel('SKU', fontsize=12)
    ax1.set_title('Nivel de Servicio por SKU', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Agregar valores en barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', fontweight='bold')
    
    # Stockouts por SKU
    ax2 = axes[0, 1]
    bars2 = ax2.bar(sop.metrics_df['SKU'], sop.metrics_df['Stockout_Weeks'], 
                    color='#E74C3C', alpha=0.8, edgecolor='darkred', linewidth=1)
    ax2.set_xlabel('SKU', fontsize=12)
    ax2.set_ylabel('Semanas con Stockout', fontsize=12)
    ax2.set_title('Stockouts por SKU', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Valores en barras
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Inventario promedio
    ax3 = axes[1, 0]
    bars3 = ax3.bar(sop.metrics_df['SKU'], sop.metrics_df['Avg_Inventory'], 
                    color='#3498DB', alpha=0.8, edgecolor='darkblue', linewidth=1)
    ax3.set_xlabel('SKU', fontsize=12)
    ax3.set_ylabel('Inventario Promedio (unidades)', fontsize=12)
    ax3.set_title('Inventario Promedio por SKU', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Valores en barras
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 10,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Cobertura promedio
    ax4 = axes[1, 1]
    bars4 = ax4.bar(sop.metrics_df['SKU'], sop.metrics_df['Avg_Coverage_Days'], 
                    color='#2ECC71', alpha=0.8, edgecolor='darkgreen', linewidth=1)
    ax4.axhline(y=7, color='orange', linestyle='--', linewidth=2, label='1 semana')
    ax4.axhline(y=14, color='green', linestyle='--', linewidth=2, label='2 semanas')
    ax4.set_xlabel('SKU', fontsize=12)
    ax4.set_ylabel('Cobertura Promedio (días)', fontsize=12)
    ax4.set_title('Cobertura de Inventario por SKU', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Valores en barras
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{height:.1f}d', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('📊 Dashboard S&OP - Resumen Ejecutivo', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'SOP_Dashboard_Resumen.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[OK] Dashboard resumen generado")

def create_sku_dashboard(sop, sku, output_dir):
    """Dashboard detallado por SKU"""
    sku_data = sop.results[sop.results['SKU'] == sku].copy()
    sku_metrics = sop.metrics_df[sop.metrics_df['SKU'] == sku].iloc[0]
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    # 1. Inventario proyectado con políticas
    ax1 = axes[0]
    ax1.plot(sku_data['Week'], sku_data['Inventory_Projected'], 
            label='Inventario Proyectado', linewidth=3, color='#2E86AB', marker='o', markersize=4)
    
    # Líneas de política
    ax1.axhline(y=sku_data['ROP'].iloc[0], color='#F39C12', linestyle='--', 
               linewidth=2, label=f'ROP = {sku_data["ROP"].iloc[0]:.0f}')
    ax1.axhline(y=sku_data['Safety_Stock_Final'].iloc[0], color='#E74C3C', linestyle='--', 
               linewidth=2, label=f'Safety Stock = {sku_data["Safety_Stock_Final"].iloc[0]:.0f}')
    
    # Zona crítica
    ax1.fill_between(sku_data['Week'], 0, sku_data['Safety_Stock_Final'].iloc[0], 
                    alpha=0.3, color='red', label='Zona Crítica')
    
    # Stockouts
    stockouts = sku_data[sku_data['Stockout_Flag'] == 1]
    if len(stockouts) > 0:
        ax1.scatter(stockouts['Week'], [0] * len(stockouts), 
                   color='red', s=150, marker='X', label='Stockout', zorder=10)
    
    ax1.set_ylabel('Unidades', fontsize=12)
    ax1.set_title(f'SKU: {sku} - Proyección de Inventario', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Demanda vs Suministro
    ax2 = axes[1]
    
    # Barras de demanda
    bars1 = ax2.bar(sku_data['Week'], sku_data['Weekly_Forecast'], 
                    label='Demanda (Forecast)', alpha=0.7, color='#9B59B6', width=5)
    
    # Línea de suministro optimizado
    ax2.plot(sku_data['Week'], sku_data['Supply_Optimized'], 
            label='Suministro Optimizado', linewidth=3, color='#27AE60', 
            marker='s', markersize=6)
    
    # Órdenes de reabastecimiento
    replenishments = sku_data[sku_data['Replenishment_Order'] > 0]
    if len(replenishments) > 0:
        ax2.scatter(replenishments['Week'], replenishments['Replenishment_Order'], 
                   color='#E67E22', s=120, marker='^', 
                   label=f'Órdenes ({len(replenishments)})', zorder=10, edgecolor='black')
    
    ax2.set_ylabel('Unidades', fontsize=12)
    ax2.set_title(f'SKU: {sku} - Demanda vs Suministro', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Métricas de cobertura
    ax3 = axes[2]
    coverage_clean = sku_data['Days_of_Coverage'].replace([np.inf, -np.inf], 30).clip(0, 30)
    
    bars3 = ax3.bar(sku_data['Week'], coverage_clean, 
                    color=['green' if x >= 14 else 'orange' if x >= 7 else 'red' 
                          for x in coverage_clean], alpha=0.8, width=5)
    
    ax3.axhline(y=7, color='orange', linestyle='--', linewidth=2, label='1 semana')
    ax3.axhline(y=14, color='green', linestyle='--', linewidth=2, label='2 semanas')
    ax3.set_xlabel('Fecha', fontsize=12)
    ax3.set_ylabel('Días de Cobertura', fontsize=12)
    ax3.set_title(f'SKU: {sku} - Cobertura de Inventario', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 25)
    
    # Agregar caja con métricas
    metrics_text = (
        f"📊 MÉTRICAS CLAVE\n"
        f"Nivel de Servicio: {sku_metrics['Service_Level_Pct']:.1f}%\n"
        f"Stockouts: {int(sku_metrics['Stockout_Weeks'])} semanas\n"
        f"Inventario Promedio: {sku_metrics['Avg_Inventory']:.0f} unidades\n"
        f"Cobertura Promedio: {sku_metrics['Avg_Coverage_Days']:.1f} días\n"
        f"Órdenes Generadas: {int(sku_metrics['Num_Orders'])}"
    )
    
    fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure, 
            fontsize=11, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'SOP_Dashboard_{sku}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Dashboard {sku} generado")

def create_metrics_dashboard(sop, output_dir):
    """Dashboard de métricas comparativas"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Comparación de servicio
    ax1 = axes[0, 0]
    colors = ['#27AE60' if x >= 95 else '#F39C12' if x >= 90 else '#E74C3C' 
             for x in sop.metrics_df['Service_Level_Pct']]
    
    wedges, texts, autotexts = ax1.pie(sop.metrics_df['Service_Level_Pct'], 
                                      labels=sop.metrics_df['SKU'],
                                      colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Distribución Nivel de Servicio', fontsize=12, fontweight='bold')
    
    # 2. Inventario vs Órdenes
    ax2 = axes[0, 1]
    scatter = ax2.scatter(sop.metrics_df['Avg_Inventory'], sop.metrics_df['Num_Orders'],
                         c=sop.metrics_df['Service_Level_Pct'], cmap='RdYlGn',
                         s=200, alpha=0.8, edgecolor='black')
    
    for i, sku in enumerate(sop.metrics_df['SKU']):
        ax2.annotate(sku, (sop.metrics_df.iloc[i]['Avg_Inventory'], 
                          sop.metrics_df.iloc[i]['Num_Orders']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax2.set_xlabel('Inventario Promedio', fontsize=11)
    ax2.set_ylabel('Número de Órdenes', fontsize=11)
    ax2.set_title('Inventario vs Órdenes de Reabastecimiento', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Nivel de Servicio (%)', fontsize=10)
    
    # 3. Ranking de performance
    ax3 = axes[0, 2]
    performance_score = (sop.metrics_df['Service_Level_Pct'] * 0.6 + 
                        (100 - sop.metrics_df['Stockout_Weeks'] * 10) * 0.4)
    
    sorted_data = sop.metrics_df.copy()
    sorted_data['Performance_Score'] = performance_score
    sorted_data = sorted_data.sort_values('Performance_Score', ascending=True)
    
    bars = ax3.barh(sorted_data['SKU'], sorted_data['Performance_Score'],
                    color=['#27AE60' if x >= 90 else '#F39C12' if x >= 80 else '#E74C3C' 
                          for x in sorted_data['Performance_Score']])
    
    ax3.set_xlabel('Score de Performance', fontsize=11)
    ax3.set_title('Ranking de Performance S&OP', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Valores en barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}', ha='left', va='center', fontweight='bold')
    
    # 4. Tendencia de stockouts
    ax4 = axes[1, 0]
    ax4.bar(sop.metrics_df['SKU'], sop.metrics_df['Stockout_Weeks'],
            color='#E74C3C', alpha=0.8, edgecolor='darkred')
    ax4.set_ylabel('Semanas con Stockout', fontsize=11)
    ax4.set_title('Análisis de Stockouts', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Eficiencia de inventario
    ax5 = axes[1, 1]
    efficiency = sop.metrics_df['Service_Level_Pct'] / sop.metrics_df['Avg_Inventory'] * 100
    
    bars5 = ax5.bar(sop.metrics_df['SKU'], efficiency,
                    color='#3498DB', alpha=0.8, edgecolor='darkblue')
    ax5.set_ylabel('Eficiencia (Servicio/Inventario)', fontsize=11)
    ax5.set_title('Eficiencia de Inventario', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Resumen ejecutivo
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calcular métricas globales
    avg_service = sop.metrics_df['Service_Level_Pct'].mean()
    total_stockouts = sop.metrics_df['Stockout_Weeks'].sum()
    avg_inventory = sop.metrics_df['Avg_Inventory'].mean()
    total_orders = sop.metrics_df['Num_Orders'].sum()
    
    summary_text = f"""
📈 RESUMEN EJECUTIVO S&OP

🎯 Nivel de Servicio Promedio:
   {avg_service:.1f}%

⚠️ Total Stockouts:
   {int(total_stockouts)} semanas

📦 Inventario Promedio Global:
   {avg_inventory:.0f} unidades

🔄 Total Órdenes Generadas:
   {int(total_orders)} órdenes

✅ SKUs con Servicio ≥95%:
   {len(sop.metrics_df[sop.metrics_df['Service_Level_Pct'] >= 95])}/{len(sop.metrics_df)}

🎯 Performance General:
   {'EXCELENTE' if avg_service >= 95 else 'BUENA' if avg_service >= 90 else 'MEJORABLE'}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('📊 Dashboard S&OP - Análisis de Métricas Avanzadas', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'SOP_Dashboard_Metricas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[OK] Dashboard de métricas generado")

if __name__ == "__main__":
    generate_dashboards()
