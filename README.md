# Supply Chain Demand Forecasting

Sistema de pronóstico de demanda para cadenas de suministro utilizando modelos ARIMAX/SARIMAX con variables exógenas.

## 📋 Características

- **Modelos avanzados**: ARIMAX/SARIMAX con soporte para variables exógenas
- **Auto-tuning**: Optimización automática de parámetros con `pmdarima`
- **Evaluación completa**: Métricas MAE, RMSE, MAPE, R² y backtesting
- **Visualizaciones**: Gráficos de pronósticos, residuos y descomposición
- **Configuración flexible**: Archivo YAML para personalización completa
- **🆕 S&OP Integrado**: Optimización de inventario basada en forecasts ARIMAX
  - Cálculo automático de ROP (Reorder Point) y Safety Stock dinámico
  - Simulación de inventario semana a semana
  - Generación automática de órdenes de reabastecimiento
  - Análisis de nivel de servicio y cobertura de inventario

## 🚀 Instalación y Configuración

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar datos

Edita el archivo `config.yaml`:

```yaml
data:
  # Ruta al archivo Excel con tus datos
  excel_path: "C:\\ruta\\a\\tu\\archivo.xlsx"
  sheet_name: "Sheet1"
  date_column: "date"
  target_column: "demand"
  exogenous_columns:
    - "price"
    - "promo_flag"
    - "lead_time_days"
    - "stock_on_hand"
  frequency: "MS"  # Mensual
  timezone: "America/Lima"
```

### 3. Preparar datos

Tu archivo Excel debe tener las siguientes columnas mínimas:
- **date**: Fechas (formato YYYY-MM-DD)
- **demand**: Valores de demanda históricos
- Variables exógenas adicionales según configuración

## 🎯 Ejecución

### Opción 1: Script principal (Recomendado)

```bash
python main.py
```

Este comando ejecutará el flujo completo:
1. ✅ Carga configuración
2. ✅ Carga y preprocesa datos
3. ✅ Entrena modelo con auto-tuning
4. ✅ Genera pronósticos
5. ✅ Evalúa modelo
6. ✅ Crea visualizaciones
7. ✅ Guarda modelo entrenado
8. ✅ Genera métricas y pronósticos

### Opción 2: Jupyter Notebook

Abre `Arimax.IPYNB` para análisis interactivo.

## 📊 Resultados

Después de la ejecución, encontrarás:

```
outputs/
├── plots/
│   ├── forecast_plot.png      # Pronóstico principal
│   ├── residuals_analysis.png # Análisis de residuos
│   └── decomposition.png      # Descomposición temporal
├── forecasts/
│   └── forecasts.csv          # Pronósticos futuros
├── metrics/
│   ├── evaluation_metrics.csv # Métricas de evaluación
│   └── backtest_*.csv         # Resultados de backtesting
└── models/
    └── arimax_model.pkl       # Modelo entrenado
```

## ⚙️ Configuración Avanzada

### Parámetros del modelo

```yaml
model:
  # Usa auto_arima para optimización automática
  auto_arima:
    max_p: 5        # Máximo orden AR
    max_q: 5        # Máximo orden MA
    m: 12          # Período estacional (12 para datos mensuales)
    seasonal: true # Incluir componente estacional
    stepwise: true # Búsqueda paso a paso
    n_jobs: -1     # Usar todos los núcleos del CPU
```

### Pronósticos

```yaml
forecasting:
  horizon: 12      # Número de períodos a pronosticar
  test_size: 0.2   # Proporción de datos para prueba
  alpha: 0.05     # Nivel de significancia (95% confianza)
```

### Evaluación

```yaml
evaluation:
  metrics: ["MAE", "RMSE", "MAPE", "R2"]
  backtesting:
    n_splits: 5
    min_train_size: 24
```

## 🔧 Solución de problemas

### Error: "Excel file not found"
- Verifica que la ruta en `config.yaml` sea correcta
- Usa barras invertidas dobles: `C:\\ruta\\archivo.xlsx`

### Error: "Missing required columns"
- Asegúrate de que tu Excel tenga las columnas especificadas
- Los nombres de columnas deben coincidir exactamente

### Modelo no converge
- Reduce la complejidad del modelo en `config.yaml`
- Usa parámetros más conservadores en `auto_arima`

### Memoria insuficiente
- Reduce `max_p`, `max_q` en configuración
- Usa `n_jobs: 1` en lugar de `-1`

## 📈 Interpretación de resultados

### Métricas principales:
- **MAE** (Mean Absolute Error): Error promedio en unidades originales
- **RMSE** (Root Mean Square Error): Error cuadrático medio
- **MAPE** (Mean Absolute Percentage Error): Error porcentual promedio
- **R²**: Coeficiente de determinación (1.0 = perfecto)

### Visualizaciones:
- **Pronóstico**: Datos históricos vs pronósticos futuros
- **Residuos**: Análisis de errores del modelo
- **Descomposición**: Tendencia, estacionalidad y componente irregular

## 🆕 Optimización S&OP (Opcional)

El sistema ahora incluye **optimización de inventario** integrada que usa los forecasts ARIMAX para generar planes S&OP.

### Activar S&OP

Edita `config.yaml`:

```yaml
sop:
  enabled: true  # Activar optimización S&OP
  inventory_policies_path: "data/inventory_policies.xlsx"  # Opcional
  supply_plan_path: "data/supply_plan.xlsx"  # Opcional
  service_level: 0.95  # 95% nivel de servicio
  output_dir: "outputs/sop"
```

### Generar archivos de ejemplo

```bash
python create_sop_data_templates.py
```

Esto crea:
- `data/inventory_policies.xlsx` - Políticas de inventario (Safety Stock, ROP, MOQ, etc.)
- `data/supply_plan.xlsx` - Plan de suministro programado

### Resultados S&OP

Después de ejecutar `python main.py` con S&OP activado:

```
outputs/sop/
├── SOP_Inventory_Plan.xlsx    # Plan completo con 3 hojas
│   ├── Inventory_Projection   # Proyección semanal
│   ├── Performance_Metrics    # Métricas por SKU
│   └── Inventory_Policies     # ROP, Safety Stock calculados
└── charts/
    ├── SOP_Summary_Metrics.png
    └── SOP_SKU_*.png
```

### Fórmulas S&OP

**Safety Stock Dinámico:**
```
SS = Z × σ_Demanda × √(LeadTime_semanas)
```
Donde Z = 1.64 para 95% de nivel de servicio

**Punto de Reorden (ROP):**
```
ROP = (Demanda_Promedio_Semanal × LeadTime_semanas) + SafetyStock
```

**Reabastecimiento Automático:**
```
SI Inventario_Proyectado < ROP ENTONCES:
    Orden = MAX(MOQ, Max_Stock - Inventario)
```

## 🔄 Uso programático

```python
from src.models import ARIMAXForecaster
from src.data_loader import DataLoader
from src.sop_optimizer import SOPOptimizer

# Cargar datos
config = {"data": {"excel_path": "datos.xlsx", ...}}
data_loader = DataLoader(config)
target, exog = data_loader.load_excel(), data_loader.preprocess()

# Entrenar modelo
forecaster = ARIMAXForecaster(config)
forecaster.train(target[:-12], exog[:-12])

# Generar pronósticos
forecast, conf_int = forecaster.forecast(12, exog=None)

# Optimización S&OP (opcional)
sop = SOPOptimizer(config, service_level=0.95)
sop_forecast = pd.DataFrame({
    'SKU': ['DEMAND'],
    'Date': forecast.index,
    'Forecast': forecast.values
})
results = sop.simulate_inventory(sop_forecast)
sop.export_results('outputs/sop')
```

## 📞 Soporte

Para problemas o preguntas:
1. Revisa la sección de solución de problemas
2. Verifica que todas las dependencias estén instaladas
3. Asegúrate de que los datos estén en el formato correcto

---

**¡Éxito con tus pronósticos de demanda! 🚀**
