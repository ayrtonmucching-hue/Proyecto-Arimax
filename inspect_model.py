#!/usr/bin/env python3
"""
Script para inspeccionar el modelo ARIMAX guardado
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

# Agregar src al path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from models import ARIMAXForecaster

def inspect_model():
    """Inspeccionar el modelo guardado."""
    
    model_path = "models/arimax_model.pkl"
    
    if not Path(model_path).exists():
        print(f"[ERROR] Modelo no encontrado en: {model_path}")
        return
    
    try:
        # Cargar datos del modelo
        print("="*60)
        print("INSPECCION DEL MODELO ARIMAX")
        print("="*60)
        
        model_data = joblib.load(model_path)
        
        print(f"Archivo: {model_path}")
        print(f"Tamaño: {Path(model_path).stat().st_size / 1024:.2f} KB")
        print()
        
        # Información básica
        print("INFORMACION BASICA:")
        print("-" * 30)
        print(f"Order (p,d,q): {model_data['order']}")
        print(f"Seasonal order (P,D,Q,s): {model_data['seasonal_order']}")
        print()
        
        # Información del modelo ajustado
        fitted_model = model_data['fitted_model']
        print("ESTADISTICAS DEL MODELO:")
        print("-" * 30)
        print(f"AIC: {fitted_model.aic:.4f}")
        print(f"BIC: {fitted_model.bic:.4f}")
        print(f"Log-likelihood: {fitted_model.llf:.4f}")
        print(f"Observaciones: {fitted_model.nobs}")
        print()
        
        # Parámetros del modelo
        print("PARAMETROS DEL MODELO:")
        print("-" * 30)
        params = fitted_model.params
        for i, (param_name, value) in enumerate(params.items()):
            print(f"{param_name}: {value:.6f}")
        print()
        
        # Variables exógenas si existen
        if hasattr(fitted_model.data, 'exog') and fitted_model.data.exog is not None:
            print("VARIABLES EXOGENAS:")
            print("-" * 30)
            exog_shape = fitted_model.data.exog.shape
            print(f"Número de variables: {exog_shape[1]}")
            print(f"Períodos de entrenamiento: {exog_shape[0]}")
            
            # Mostrar estadísticas de las variables exógenas
            exog_df = pd.DataFrame(fitted_model.data.exog)
            print(f"\nEstadísticas de variables exógenas:")
            print(exog_df.describe())
        
        # Resumen del modelo
        print("\n" + "="*60)
        print("RESUMEN COMPLETO DEL MODELO:")
        print("="*60)
        print(fitted_model.summary())
        
    except Exception as e:
        print(f"[ERROR] Error al cargar el modelo: {e}")

if __name__ == "__main__":
    inspect_model()
