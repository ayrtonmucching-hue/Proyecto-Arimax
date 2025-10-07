#!/usr/bin/env python3
"""
Supply Chain Forecasting - Main Execution Script

Este script ejecuta el flujo completo de forecasting de demanda:
1. Cargando configuracion
2. Creando directorios de salida
3. Cargando datos
4. Dividiendo datos en entrenamiento y prueba
5. Entrenando modelo ARIMAX/SARIMAX
6. Generando pronosticos
7. Evaluando modelo
8. Generando visualizaciones
9. Ejecutando backtesting
10. Guardando modelo
11. Guardando metricas
"""

import sys
import os
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from config_loader import load_config
from data_loader import DataLoader
from models import ARIMAXForecaster
from evaluation import ModelEvaluator, BacktestEngine
from visualization import ForecastVisualizer
from sop_optimizer import SOPOptimizer

# Suprimir warnings no críticos
warnings.filterwarnings('ignore')


def check_requirements():
    """Verificar que las dependencias estén instaladas."""
    try:
        import pandas as pd
        import numpy as np
        import statsmodels
        import sklearn
        import matplotlib
        import yaml
        import joblib
        print("[OK] Todas las dependencias están instaladas")
        return True
    except ImportError as e:
        print(f"[ERROR] Falta instalar dependencias: {e}")
        print("Ejecuta: pip install -r requirements.txt")
        return False


def create_output_directories(config):
    """Crear directorios de salida si no existen."""
    output_dirs = [
        config['output']['models_dir'],
        config['output']['plots_dir'],
        config['output']['forecasts_dir'],
        config['output']['metrics_dir']
    ]

    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Directorio creado: {dir_path}")


def main():
    """Función principal que ejecuta todo el flujo de trabajo."""
    print("="*70)
    print("SUPPLY CHAIN DEMAND FORECASTING")
    print("="*70)

    # 1. Verificar dependencias
    if not check_requirements():
        return

    try:
        # 2. Cargar configuración
        print("\nCargando configuracion...")
        config = load_config("config.yaml")
        print("[OK] Configuracion cargada exitosamente")

        # 3. Crear directorios de salida
        print("\nCreando directorios de salida...")
        create_output_directories(config)

        # 4. Cargar y preprocesar datos
        print("\nCargando datos...")
        data_loader = DataLoader(config)
        df = data_loader.load_excel()

        target, exog = data_loader.preprocess()

        # 5. Dividir datos en entrenamiento y prueba
        print("\nDividiendo datos en entrenamiento y prueba...")
        train_target, test_target, train_exog, test_exog = data_loader.train_test_split(
            target, exog, test_size=config['forecasting']['test_size']
        )

        # 6. Entrenar modelo
        print("\nEntrenando modelo ARIMAX/SARIMAX...")
        forecaster = ARIMAXForecaster(config)
        forecaster.train(train_target, train_exog)

        # 7. Generar pronósticos
        print("\nGenerando pronosticos...")
        horizon = config['forecasting']['horizon']

        # ✅ CORREGIDO: pasamos horizon sin "steps="
        forecast, conf_int = forecaster.forecast(
            horizon,
            exog=None,  # No tenemos datos exogenos futuros para el pronostico
            alpha=config['forecasting']['alpha']
        )

        # 8. Evaluar modelo en datos de prueba
        print("\nEvaluando modelo...")
        evaluator = ModelEvaluator(config)

        # Crear índice de fechas para el pronóstico
        last_date = test_target.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq='MS'
        )
        forecast.index = forecast_dates

        # Evaluar en datos de prueba históricos
        test_predictions = forecaster.predict(
            start=len(train_target),
            end=len(target) - 1,
            exog=test_exog
        )

        test_metrics = evaluator.evaluate(test_target, test_predictions)
        evaluator.print_metrics(test_metrics)

        # 9. Generar visualizaciones
        print("\nGenerando visualizaciones...")
        visualizer = ForecastVisualizer(config)

        fig = visualizer.plot_forecast(
            train=train_target,
            test=test_target,
            forecast=forecast,
            conf_int=conf_int,
            title="Pronostico de Demanda - ARIMAX",
            save_path=f"{config['output']['plots_dir']}/forecast_plot.png"
        )

        if hasattr(forecaster.fitted_model, 'resid'):
            residuals = pd.Series(forecaster.fitted_model.resid, index=train_target.index)
            visualizer.plot_residuals(
                residuals,
                title="Analisis de Residuos",
                save_path=f"{config['output']['plots_dir']}/residuals_analysis.png"
            )

        if len(target) >= 24:
            visualizer.plot_components(
                target,
                title="Descomposicion de la Serie Temporal",
                save_path=f"{config['output']['plots_dir']}/decomposition.png"
            )

        # 10. Backtesting
        print("\nEjecutando backtesting...")
        backtest_engine = BacktestEngine(config)
        backtest_metrics, backtest_predictions = backtest_engine.backtest(
            target, exog, forecaster, horizon=1
        )

        # 11. Guardar modelo
        print("\nGuardando modelo...")
        model_path = f"{config['output']['models_dir']}/arimax_model.pkl"
        forecaster.save_model(model_path)

        # 12. Guardar métricas
        print("\nGuardando metricas...")
        metrics_df = pd.DataFrame([test_metrics])
        metrics_path = f"{config['output']['metrics_dir']}/evaluation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        # Guardar pronósticos
        forecast_df = pd.DataFrame({
            'fecha': forecast.index,
            'pronostico': forecast.values,
            'limite_inferior': conf_int['lower'].values,
            'limite_superior': conf_int['upper'].values
        })
        forecast_path = f"{config['output']['forecasts_dir']}/forecasts.csv"
        forecast_df.to_csv(forecast_path, index=False)

        # 13. S&OP Optimization (opcional)
        if config.get('sop', {}).get('enabled', False):
            print("\n" + "="*70)
            print("S&OP - OPTIMIZACION DE INVENTARIO")
            print("="*70)
            
            try:
                sop = SOPOptimizer(config, service_level=config['sop']['service_level'])
                
                # Cargar politicas (opcional)
                if config['sop']['inventory_policies_path']:
                    sop.load_inventory_policies(config['sop']['inventory_policies_path'])
                
                if config['sop']['supply_plan_path']:
                    sop.load_supply_plan(config['sop']['supply_plan_path'])
                
                # Preparar forecast para S&OP (agregar SKU si no existe)
                sop_forecast = forecast_df.copy()
                sop_forecast.rename(columns={'fecha': 'Date', 'pronostico': 'Forecast'}, inplace=True)
                
                if 'SKU' not in sop_forecast.columns:
                    # Usar nombre del target como SKU
                    sop_forecast['SKU'] = config['data']['target_column'].upper()
                
                # Simular inventario
                sop.simulate_inventory(sop_forecast)
                
                # Exportar resultados
                sop.export_results(config['sop']['output_dir'])
                
                # Generar graficos
                sop.generate_charts(config['sop']['charts_dir'])
                
                # Mostrar resumen
                sop.print_summary()
                
            except Exception as e:
                print(f"[WARNING] Error en optimizacion S&OP: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "="*70)
        print("EJECUCION COMPLETADA EXITOSAMENTE")
        print("="*70)
        print("Archivos generados en:")
        print(f"   - Modelo: {model_path}")
        print(f"   - Pronosticos: {forecast_path}")
        print(f"   - Metricas: {metrics_path}")
        print(f"   - Graficos: {config['output']['plots_dir']}/")
        print(f"   - Resultados backtest: {config['output']['metrics_dir']}/backtest_*.csv")

        print("\nResumen de metricas principales:")
        for metric, value in test_metrics.items():
            if metric != 'N' and not pd.isna(value):
                if metric == 'MAPE':
                    print(f"   - {metric}: {value:.2f}%")
                elif metric == 'R2':
                    print(f"   - {metric}: {value:.4f}")
                else:
                    print(f"   - {metric}: {value:.4f}")

    except Exception as e:
        print(f"\n[ERROR] Error durante la ejecucion: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
