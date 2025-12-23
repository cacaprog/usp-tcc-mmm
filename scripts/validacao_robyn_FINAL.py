"""
VALIDAÇÃO ROBYN - VERSÃO FINAL CORRIGIDA
=========================================

Corrige o erro de KeyError com a coluna 'ds'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr

# ==============================================================================
# FUNÇÃO CORRIGIDA DE EXTRAÇÃO
# ==============================================================================

def extract_robyn_decompositions_corrigida(pareto_result, solution_id):
    """Extrai decomposições do x_decomp_vec_collect."""
    try:
        decomp_df = pareto_result.x_decomp_vec_collect
        
        if not isinstance(decomp_df, pd.DataFrame):
            raise ValueError(f"x_decomp_vec_collect não é DataFrame: {type(decomp_df)}")
        
        decomp_filtered = decomp_df[decomp_df['sol_id'] == solution_id].copy()
        
        if len(decomp_filtered) == 0:
            if 'top_sol' in decomp_df.columns:
                decomp_filtered = decomp_df[decomp_df['top_sol'] == solution_id].copy()
            
            if len(decomp_filtered) == 0:
                soluções_disponiveis = decomp_df['sol_id'].unique()[:10].tolist()
                raise ValueError(f"Solução {solution_id} não encontrada. Disponíveis: {soluções_disponiveis}")
        
        print(f"✅ Decomposições extraídas: {decomp_filtered.shape}")
        print(f"   Período: {decomp_filtered['ds'].min()} a {decomp_filtered['ds'].max()}")
        
        return decomp_filtered
        
    except Exception as e:
        print(f"❌ Erro ao extrair decomposições: {str(e)}")
        return None


def prepare_validation_data_corrigida(decomp_df, df_ground_truth):
    """Prepara dados para validação."""
    # Garantir formato de data
    decomp_df['ds'] = pd.to_datetime(decomp_df['ds'])
    
    # Preparar ground truth - verificar nome da coluna de data
    df_gt = df_ground_truth.copy()
    
    if 'date' in df_gt.columns and 'ds' not in df_gt.columns:
        df_gt['ds'] = pd.to_datetime(df_gt['date'])
    elif 'ds' in df_gt.columns:
        df_gt['ds'] = pd.to_datetime(df_gt['ds'])
    else:
        raise ValueError("Ground truth deve ter coluna 'date' ou 'ds'")
    
    # Merge
    validation_df = decomp_df.merge(
        df_gt[['ds', 'revenue', 'true_base_revenue', 
               'true_google_contribution', 'true_meta_contribution', 
               'true_tv_contribution']],
        on='ds',
        how='inner'
    )
    
    # Renomear colunas
    rename_map = {
        'dep_var': 'pred_revenue',
        'google_spend': 'pred_google_contribution',
        'meta_spend': 'pred_meta_contribution', 
        'tv_spend': 'pred_tv_contribution',
        'trend': 'pred_trend',
        'season': 'pred_season'
    }
    
    validation_df = validation_df.rename(columns=rename_map)
    
    # Criar base revenue predita
    base_components = []
    for comp in ['pred_trend', 'pred_season', 'holiday', 'intercept']:
        if comp in validation_df.columns:
            base_components.append(comp)
    
    if base_components:
        validation_df['pred_base_revenue'] = validation_df[base_components].sum(axis=1)
    else:
        # Calcular como residual
        validation_df['pred_base_revenue'] = (
            validation_df['pred_revenue'] - 
            validation_df['pred_google_contribution'] -
            validation_df['pred_meta_contribution'] -
            validation_df['pred_tv_contribution']
        )
    
    # Guardar referência ao ground truth com spends para ROI
    validation_df = validation_df.merge(
        df_gt[['ds', 'google_spend', 'meta_spend', 'tv_spend']],
        on='ds',
        how='left',
        suffixes=('', '_spend_actual')
    )
    
    print(f"\n✅ Dados de validação preparados")
    print(f"   Período: {validation_df['ds'].min()} a {validation_df['ds'].max()}")
    print(f"   Observações: {len(validation_df)}")
    
    return validation_df


def calculate_metrics(y_true, y_pred, label=""):
    """Calcula métricas de regressão."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = np.array(y_true)[mask]
    y_pred_clean = np.array(y_pred)[mask]
    
    if len(y_true_clean) == 0:
        return {
            'label': label,
            'R2': np.nan,
            'RMSE': np.nan,
            'NRMSE': np.nan,
            'MAPE': np.nan,
            'Correlation': np.nan,
            'p_value': np.nan
        }
    
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))
    nrmse = rmse / np.mean(y_true_clean) * 100 if np.mean(y_true_clean) != 0 else np.nan
    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
    
    try:
        corr, p_value = pearsonr(y_true_clean, y_pred_clean)
    except:
        corr, p_value = np.nan, np.nan
    
    return {
        'label': label,
        'R2': r2,
        'RMSE': rmse,
        'NRMSE': nrmse,
        'MAPE': mape,
        'Correlation': corr,
        'p_value': p_value
    }


# ==============================================================================
# VALIDAÇÃO COMPLETA
# ==============================================================================

def run_validation_corrigida(pareto_result, solution_id, df_ground_truth, train_size=0.7):
    """Executa validação completa."""
    print("\n" + "="*80)
    print("VALIDAÇÃO ROBYN COM GROUND TRUTH (VERSÃO FINAL CORRIGIDA)")
    print("="*80)
    
    # ETAPA 1: Extrair
    print("\n[1/5] Extraindo decomposições...")
    decomp_df = extract_robyn_decompositions_corrigida(pareto_result, solution_id)
    
    if decomp_df is None:
        return None, None, None
    
    # ETAPA 2: Preparar
    print("\n[2/5] Preparando dados...")
    validation_df = prepare_validation_data_corrigida(decomp_df, df_ground_truth)
    
    # ETAPA 3: Dividir
    print("\n[3/5] Dividindo treino/teste...")
    n = len(validation_df)
    n_train = int(n * train_size)
    
    df_train = validation_df.iloc[:n_train].copy()
    df_test = validation_df.iloc[n_train:].copy()
    
    print(f"   Treino: {len(df_train)} obs ({len(df_train)/n*100:.1f}%)")
    print(f"   Teste:  {len(df_test)} obs ({len(df_test)/n*100:.1f}%)")
    
    # ETAPA 4: Calcular métricas
    print("\n[4/5] Calculando métricas...")
    results = {}
    
    # 4.1 Validação Global
    print("\n" + "-"*80)
    print("VALIDAÇÃO GLOBAL (REVENUE)")
    print("-"*80)
    
    metrics_train = calculate_metrics(df_train['revenue'], df_train['pred_revenue'], "Treino")
    metrics_test = calculate_metrics(df_test['revenue'], df_test['pred_revenue'], "Teste")
    
    overfitting = (metrics_train['R2'] - metrics_test['R2']) / metrics_train['R2'] if metrics_train['R2'] > 0 else np.nan
    
    print(f"\n📊 TREINO:")
    print(f"   R² = {metrics_train['R2']:.3f}")
    print(f"   NRMSE = {metrics_train['NRMSE']:.2f}%")
    print(f"   MAPE = {metrics_train['MAPE']:.2f}%")
    
    print(f"\n📊 TESTE:")
    print(f"   R² = {metrics_test['R2']:.3f}")
    print(f"   NRMSE = {metrics_test['NRMSE']:.2f}%")
    print(f"   MAPE = {metrics_test['MAPE']:.2f}%")
    
    print(f"\n⚖️ OVERFITTING SCORE = {overfitting:.3f}")
    if overfitting < 0.1:
        print("   ✅ Baixo overfitting")
    elif overfitting < 0.3:
        print("   ⚠️ Overfitting moderado")
    else:
        print("   ❌ Overfitting SEVERO")
    
    results['global'] = {
        'train': metrics_train,
        'test': metrics_test,
        'overfitting': overfitting
    }
    
    # 4.2 Por Componente
    print("\n" + "-"*80)
    print("VALIDAÇÃO POR COMPONENTE")
    print("-"*80)
    
    components = {
        'Base': ('pred_base_revenue', 'true_base_revenue'),
        'Google': ('pred_google_contribution', 'true_google_contribution'),
        'Meta': ('pred_meta_contribution', 'true_meta_contribution'),
        'TV': ('pred_tv_contribution', 'true_tv_contribution')
    }
    
    results['components'] = {}
    
    for comp_name, (pred_col, true_col) in components.items():
        if pred_col in validation_df.columns and true_col in validation_df.columns:
            print(f"\n{comp_name}:")
            
            m_train = calculate_metrics(df_train[true_col], df_train[pred_col], f"{comp_name}_Train")
            m_test = calculate_metrics(df_test[true_col], df_test[pred_col], f"{comp_name}_Test")
            
            print(f"   TREINO: MAPE={m_train['MAPE']:.2f}%, Corr={m_train['Correlation']:.3f}")
            print(f"   TESTE:  MAPE={m_test['MAPE']:.2f}%, Corr={m_test['Correlation']:.3f}")
            
            results['components'][comp_name] = {'train': m_train, 'test': m_test}
        else:
            print(f"\n{comp_name}: ⚠️ Colunas não encontradas")
    
    # 4.3 ROI
    print("\n" + "-"*80)
    print("VALIDAÇÃO DE ROI (TESTE)")
    print("-"*80)
    
    results['roi'] = {}
    
    roi_channels = {
        'Google': ('google_spend', 'pred_google_contribution', 'true_google_contribution'),
        'Meta': ('meta_spend', 'pred_meta_contribution', 'true_meta_contribution'),
        'TV': ('tv_spend', 'pred_tv_contribution', 'true_tv_contribution')
    }
    
    for channel, (spend_col, pred_col, true_col) in roi_channels.items():
        if all(col in df_test.columns for col in [spend_col, pred_col, true_col]):
            mask = df_test[spend_col] > 0
            
            if mask.sum() > 0:
                roi_true = df_test.loc[mask, true_col].sum() / df_test.loc[mask, spend_col].sum()
                roi_pred = df_test.loc[mask, pred_col].sum() / df_test.loc[mask, spend_col].sum()
                roi_error = (roi_pred - roi_true) / roi_true * 100
                
                print(f"\n{channel}:")
                print(f"   ROI Verdadeiro = {roi_true:.2f}")
                print(f"   ROI Estimado   = {roi_pred:.2f}")
                print(f"   Erro           = {roi_error:.1f}%")
                
                results['roi'][channel] = {
                    'roi_true': roi_true,
                    'roi_pred': roi_pred,
                    'roi_error': roi_error
                }
        else:
            print(f"\n{channel}: ⚠️ Dados de spend não disponíveis")
    
    # ETAPA 5: Visualizações
    print("\n[5/5] Gerando visualizações...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Validação Robyn: Predições vs. Ground Truth', fontsize=16, fontweight='bold')
    
    comparisons = [
        ('revenue', 'pred_revenue', 'Revenue Total'),
        ('true_base_revenue', 'pred_base_revenue', 'Base Revenue'),
        ('true_google_contribution', 'pred_google_contribution', 'Google'),
        ('true_meta_contribution', 'pred_meta_contribution', 'Meta')
    ]
    
    for idx, (true_col, pred_col, title) in enumerate(comparisons):
        ax = axes[idx // 2, idx % 2]
        
        if true_col in validation_df.columns and pred_col in validation_df.columns:
            ax.scatter(df_train[true_col], df_train[pred_col], 
                      alpha=0.6, s=50, label='Treino', color='blue')
            ax.scatter(df_test[true_col], df_test[pred_col], 
                      alpha=0.6, s=50, label='Teste', color='red')
            
            min_val = validation_df[true_col].min()
            max_val = validation_df[true_col].max()
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'k--', linewidth=2, label='Perfeito')
            
            ax.set_xlabel('Ground Truth', fontsize=11)
            ax.set_ylabel('Predição Robyn', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{title}\nDados não disponíveis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Predição')
    
    plt.tight_layout()
    
    print("\n" + "="*80)
    print("VALIDAÇÃO CONCLUÍDA!")
    print("="*80)
    
    return results, validation_df, fig


def print_report(results):
    """Imprime relatório formatado."""
    print("\n" + "="*80)
    print("RELATÓRIO DE VALIDAÇÃO")
    print("="*80)
    
    print("\n📊 TABELA 1: PERFORMANCE PREDITIVA GLOBAL")
    print("-" * 60)
    print(f"{'Métrica':<20} {'Treino':>15} {'Teste':>15}")
    print("-" * 60)
    
    g = results['global']
    print(f"{'R²':<20} {g['train']['R2']:>15.3f} {g['test']['R2']:>15.3f}")
    print(f"{'NRMSE (%)':<20} {g['train']['NRMSE']:>15.2f} {g['test']['NRMSE']:>15.2f}")
    print(f"{'MAPE (%)':<20} {g['train']['MAPE']:>15.2f} {g['test']['MAPE']:>15.2f}")
    print("-" * 60)
    print(f"{'Overfitting Score':<40} {g['overfitting']:>15.3f}")
    
    print("\n📊 TABELA 2: VALIDAÇÃO POR COMPONENTE (MAPE %)")
    print("-" * 60)
    print(f"{'Componente':<20} {'Treino':>15} {'Teste':>15}")
    print("-" * 60)
    
    for comp, metrics in results['components'].items():
        print(f"{comp:<20} {metrics['train']['MAPE']:>15.2f} {metrics['test']['MAPE']:>15.2f}")
    
    if results['roi']:
        print("\n📊 TABELA 3: VALIDAÇÃO DE ROI")
        print("-" * 70)
        print(f"{'Canal':<15} {'ROI True':>15} {'ROI Pred':>15} {'Erro (%)':>15}")
        print("-" * 70)
        
        for channel, roi in results['roi'].items():
            print(f"{channel:<15} {roi['roi_true']:>15.2f} {roi['roi_pred']:>15.2f} {roi['roi_error']:>15.1f}")
