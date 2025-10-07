"""
Script helper para generar archivos de ejemplo de datos S&OP.
Crea inventory_policies.xlsx y supply_plan.xlsx con datos de prueba.

Uso: python create_sop_data_templates.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

def create_inventory_policies():
    """
    Crea archivo con politicas de inventario de ejemplo.
    """
    print("[*] Creando inventory_policies.xlsx...")
    
    # Ejemplo con 3 SKUs
    data = {
        'SKU': ['DEMAND', 'PRODUCT_A', 'PRODUCT_B'],
        'Opening_Inventory': [1000, 500, 800],
        'Safety_Stock': [200, 100, 150],
        'Max_Stock': [3000, 1500, 2000],
        'Lead_Time_Days': [14, 21, 14],
        'MOQ': [400, 200, 300],
        'Shelf_Life_Days': [180, 365, 180]
    }
    
    df = pd.DataFrame(data)
    
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / 'inventory_policies.xlsx'
    df.to_excel(output_path, index=False)
    
    print(f"[OK] Archivo creado: {output_path}")
    print(f"     {len(df)} SKUs configurados")
    
    return df

def create_supply_plan():
    """
    Crea archivo con plan de suministro de ejemplo.
    """
    print("\n[*] Creando supply_plan.xlsx...")
    
    skus = ['DEMAND', 'PRODUCT_A', 'PRODUCT_B']
    start_date = datetime(2024, 1, 1)
    
    # Crear entregas semanales para 26 semanas
    weeks = pd.date_range(start=start_date, periods=26, freq='W-MON')
    
    records = []
    base_supply = {'DEMAND': 800, 'PRODUCT_A': 400, 'PRODUCT_B': 600}
    
    for sku in skus:
        base = base_supply[sku]
        for week in weeks:
            # Entregas con variabilidad
            supply = base * np.random.uniform(0.8, 1.2)
            
            # 10% de probabilidad de no recibir entrega
            if np.random.random() < 0.10:
                supply = 0
            
            records.append({
                'SKU': sku,
                'Date': week,
                'Supply': round(supply, 2)
            })
    
    df = pd.DataFrame(records)
    
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / 'supply_plan.xlsx'
    df.to_excel(output_path, index=False)
    
    print(f"[OK] Archivo creado: {output_path}")
    print(f"     {len(df)} registros de suministro")
    
    return df

def main():
    print("\n" + "="*70)
    print("GENERADOR DE DATOS S&OP - TEMPLATES DE EJEMPLO")
    print("="*70 + "\n")
    
    # Crear archivos
    inv_df = create_inventory_policies()
    supply_df = create_supply_plan()
    
    print("\n" + "="*70)
    print("[OK] ARCHIVOS GENERADOS EN: data/")
    print("="*70)
    print("\nPara usar estos archivos:")
    print("1. Edita config.yaml")
    print("2. Cambia 'sop: enabled: true'")
    print("3. Configura las rutas:")
    print("   inventory_policies_path: 'data/inventory_policies.xlsx'")
    print("   supply_plan_path: 'data/supply_plan.xlsx'")
    print("4. Ejecuta: python main.py")
    print("\n")

if __name__ == "__main__":
    main()
