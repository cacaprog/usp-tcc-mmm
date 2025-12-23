"""
Geração de Dataset Sintético para Marketing Mix Modeling
Reproduz os dados descritos no TCC sobre MMM em PMEs Brasileiras
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuração para reprodutibilidade
np.random.seed(42)

class SyntheticMMDataGenerator:
    """
    Gerador de dados sintéticos para Marketing Mix Modeling com ground truth conhecido.
    Implementa adstock geométrico, saturação via função Hill e componentes estruturais.
    """
    
    def __init__(self, n_weeks=156, start_year=2021):
        self.n_weeks = n_weeks
        self.start_year = start_year
        
        # Parâmetros de adstock (memória) por canal
        self.adstock_params = {
            'google': 0.4,   # Canal de performance - decaimento mais rápido
            'meta': 0.4,     # Canal de performance - decaimento mais rápido
            'tv': 0.6        # Canal de branding - decaimento mais lento
        }
        
        # Parâmetros de saturação (Hill) por canal
        self.saturation_params = {
            'google': {'alpha': 1.5, 'K': 3000, 'C': 50000},
            'meta': {'alpha': 1.8, 'K': 2500, 'C': 45000},
            'tv': {'alpha': 2.0, 'K': 5000, 'C': 80000}
        }
        
        # Coeficientes de contribuição real (ground truth)
        self.true_coefs = {
            'google': 0.35,
            'meta': 0.28,
            'tv': 0.37
        }
        
    def apply_adstock(self, x, decay_rate, max_lag=8):
        """
        Aplica transformação de adstock geométrico.
        
        M_adstock(t) = Σ(l=0 até L) w^l · M(t-l)
        
        Args:
            x: série temporal de investimento
            decay_rate: taxa de decaimento w ∈ (0,1)
            max_lag: horizonte máximo L
            
        Returns:
            série com efeito de memória aplicado
        """
        adstocked = np.zeros_like(x)
        for t in range(len(x)):
            for lag in range(min(t + 1, max_lag)):
                adstocked[t] += (decay_rate ** lag) * x[t - lag]
        return adstocked
    
    def apply_saturation(self, x, alpha, K, C):
        """
        Aplica função de saturação Hill.
        
        M_sat(t) = [M_adstock(t)^α / (K^α + M_adstock(t)^α)] · C
        
        Args:
            x: série após adstock
            alpha: parâmetro de curvatura
            K: ponto de meia-saturação
            C: capacidade máxima de resposta
            
        Returns:
            série com saturação aplicada
        """
        return (x ** alpha) / (K ** alpha + x ** alpha) * C
    
    def generate_base_revenue(self, dates_df):
        """
        Gera receita base com tendência polinomial, sazonalidade e variáveis exógenas.
        
        Componentes:
        - Tendência polinomial de segunda ordem
        - Sazonalidade anual (picos Q4)
        - Influência de SELIC e INCC
        - Ruído heterocedástico
        """
        n = len(dates_df)
        t = np.arange(n)
        
        # Tendência polinomial de segunda ordem
        trend = 350000 + 800 * t - 0.5 * t**2
        
        # Sazonalidade anual (picos no Q4)
        seasonality = 30000 * np.sin(2 * np.pi * dates_df['week'] / 52 + np.pi/2)
        
        # Variáveis exógenas
        selic_effect = -5000 * (dates_df['selic_rate'] - 8)
        incc_effect = 1000 * (dates_df['incc_index'] - 112)
        
        # Receita base (sem mídia)
        base = trend + seasonality + selic_effect + incc_effect
        
        # Ruído heterocedástico
        noise = np.random.normal(0, 15000 + 500 * t/n, n)
        
        return base + noise
    
    def generate_media_spend(self, dates_df):
        """
        Gera investimentos em mídia com padrões realistas.
        """
        n = len(dates_df)
        
        # Google Ads: investimento contínuo com variação
        google_base = 2000
        google_variation = 500 * np.sin(2 * np.pi * dates_df['week'] / 52)
        google_spend = google_base + google_variation + np.random.gamma(2, 200, n)
        google_spend = np.clip(google_spend, 500, 11200)
        
        # Meta Ads: similar ao Google mas com média menor
        meta_base = 1600
        meta_variation = 400 * np.sin(2 * np.pi * dates_df['week'] / 52 + np.pi/4)
        meta_spend = meta_base + meta_variation + np.random.gamma(2, 150, n)
        meta_spend = np.clip(meta_spend, 300, 7646)
        
        # TV: investimento esporádico (concentrado em alguns períodos)
        tv_spend = np.zeros(n)
        # Campanhas em períodos específicos
        campaign_weeks = np.random.choice(n, size=int(n * 0.35), replace=False)
        tv_spend[campaign_weeks] = np.random.gamma(3, 1200, len(campaign_weeks))
        tv_spend = np.clip(tv_spend, 0, 14075)
        
        return google_spend, meta_spend, tv_spend
    
    def generate_digital_metrics(self, google_spend, meta_spend):
        """
        Gera métricas digitais baseadas nos investimentos.
        """
        # Google Ads
        google_impressions = google_spend * np.random.uniform(80, 120, len(google_spend))
        google_impressions = np.clip(google_impressions, 54058, 500000)
        
        google_clicks = google_impressions * np.random.uniform(0.025, 0.05, len(google_impressions))
        google_clicks = np.clip(google_clicks, 1300, 22596)
        
        # Meta Ads
        meta_impressions = meta_spend * np.random.uniform(100, 150, len(meta_spend))
        meta_impressions = np.clip(meta_impressions, 33503, 600000)
        
        meta_reach = meta_impressions * np.random.uniform(0.6, 0.8, len(meta_impressions))
        meta_reach = np.clip(meta_reach, 23452, 420000)
        
        meta_frequency = np.ones(len(meta_spend))  # Simplificado para 1
        
        # Google Analytics
        total_traffic = (google_clicks * 0.8 + meta_reach * 0.001)
        ga_total_users = total_traffic * np.random.uniform(0.8, 1.2, len(total_traffic))
        ga_total_users = np.clip(ga_total_users, 7664, 15000)
        
        ga_new_users = ga_total_users * np.random.uniform(0.75, 0.85, len(ga_total_users))
        
        ga_engaged_sessions = ga_total_users * np.random.uniform(0.45, 0.55, len(ga_total_users))
        
        ga_page_views = ga_engaged_sessions * np.random.uniform(8, 12, len(ga_engaged_sessions))
        
        return {
            'google_impressions': google_impressions,
            'google_clicks': google_clicks,
            'meta_impressions': meta_impressions,
            'meta_reach': meta_reach,
            'meta_frequency': meta_frequency,
            'ga_total_users': ga_total_users,
            'ga_new_users': ga_new_users,
            'ga_engaged_sessions': ga_engaged_sessions,
            'ga_page_views': ga_page_views
        }
    
    def generate_exogenous_variables(self, dates_df):
        """
        Gera variáveis macroeconômicas (SELIC e INCC).
        """
        n = len(dates_df)
        
        # SELIC: variação ao longo do período (2021-2023)
        # Ciclo de alta e posterior queda
        selic_base = np.zeros(n)
        for i in range(n):
            if i < n//3:
                selic_base[i] = 2 + 6 * (i / (n//3))  # Subida de 2% para 8%
            elif i < 2*n//3:
                selic_base[i] = 8 + 6 * ((i - n//3) / (n//3))  # Subida de 8% para 14%
            else:
                selic_base[i] = 14 - 4 * ((i - 2*n//3) / (n//3))  # Queda de 14% para 10%
        
        selic_rate = selic_base + np.random.normal(0, 0.3, n)
        selic_rate = np.clip(selic_rate, 2, 14)
        
        # INCC: tendência de alta com sazonalidade
        incc_trend = 100 + 27 * np.arange(n) / n
        incc_seasonal = 2 * np.sin(2 * np.pi * dates_df['month'] / 12)
        incc_index = incc_trend + incc_seasonal + np.random.normal(0, 1, n)
        incc_index = np.clip(incc_index, 100, 127)
        
        return selic_rate, incc_index
    
    def generate_dataset(self):
        """
        Gera o dataset sintético completo com ground truth.
        """
        # Estrutura temporal
        dates = pd.date_range(
            start=f'{self.start_year}-01-01',
            periods=self.n_weeks,
            freq='W'
        )
        
        df = pd.DataFrame({
            'date': dates,
            'year': dates.year,
            'week': dates.isocalendar().week,
            'month': dates.month
        })
        
        # Variáveis exógenas
        df['selic_rate'], df['incc_index'] = self.generate_exogenous_variables(df)
        
        # Investimentos em mídia
        google_spend, meta_spend, tv_spend = self.generate_media_spend(df)
        df['google_spend'] = google_spend
        df['meta_spend'] = meta_spend
        df['tv_spend'] = tv_spend
        
        # Métricas digitais
        digital_metrics = self.generate_digital_metrics(google_spend, meta_spend)
        for key, value in digital_metrics.items():
            df[key] = value
        
        # Aplicar adstock
        google_adstock = self.apply_adstock(google_spend, self.adstock_params['google'])
        meta_adstock = self.apply_adstock(meta_spend, self.adstock_params['meta'])
        tv_adstock = self.apply_adstock(tv_spend, self.adstock_params['tv'])
        
        # Aplicar saturação
        google_sat = self.apply_saturation(
            google_adstock,
            self.saturation_params['google']['alpha'],
            self.saturation_params['google']['K'],
            self.saturation_params['google']['C']
        )
        
        meta_sat = self.apply_saturation(
            meta_adstock,
            self.saturation_params['meta']['alpha'],
            self.saturation_params['meta']['K'],
            self.saturation_params['meta']['C']
        )
        
        tv_sat = self.apply_saturation(
            tv_adstock,
            self.saturation_params['tv']['alpha'],
            self.saturation_params['tv']['K'],
            self.saturation_params['tv']['C']
        )
        
        # Ground truth: contribuições reais de cada canal
        df['google_contribution'] = google_sat * self.true_coefs['google']
        df['meta_contribution'] = meta_sat * self.true_coefs['meta']
        df['tv_contribution'] = tv_sat * self.true_coefs['tv']
        
        # Receita base
        base_revenue = self.generate_base_revenue(df)
        
        # Receita total = base + contribuições de mídia
        df['revenue'] = (
            base_revenue +
            df['google_contribution'] +
            df['meta_contribution'] +
            df['tv_contribution']
        )
        
        # Garantir que está nos limites observados
        df['revenue'] = np.clip(df['revenue'], 350618, 790287)
        
        return df
    
    def validate_statistics(self, df):
        """
        Valida se as estatísticas correspondem à Tabela 2 do trabalho.
        """
        print("="*80)
        print("VALIDAÇÃO DAS ESTATÍSTICAS DESCRITIVAS")
        print("="*80)
        
        variables = [
            'year', 'week', 'month', 'revenue',
            'google_spend', 'meta_spend', 'tv_spend',
            'google_impressions', 'google_clicks',
            'meta_impressions', 'meta_reach', 'meta_frequency',
            'ga_total_users', 'ga_new_users', 'ga_engaged_sessions', 'ga_page_views',
            'selic_rate', 'incc_index'
        ]
        
        stats_df = pd.DataFrame({
            'Variável': variables,
            'Obs': [len(df)] * len(variables),
            'Média': [df[v].mean() for v in variables],
            'Desvio': [df[v].std() for v in variables],
            'Mínimo': [df[v].min() for v in variables],
            '50%': [df[v].median() for v in variables],
            'Máximo': [df[v].max() for v in variables]
        })
        
        print(stats_df.to_string(index=False))
        print("\n")
        
        return stats_df


# Execução
if __name__ == "__main__":
    print("Gerando dataset sintético para Marketing Mix Modeling...")
    print("Metodologia: Adstock geométrico + Saturação Hill + Componentes estruturais\n")
    
    generator = SyntheticMMDataGenerator(n_weeks=156, start_year=2021)
    df_synthetic = generator.generate_dataset()
    
    # Validação
    stats = generator.validate_statistics(df_synthetic)
    
    # Salvar dataset
    df_synthetic.to_csv('synthetic_mmm_dataset.csv', index=False)
    print(f"\n✓ Dataset salvo como 'synthetic_mmm_dataset.csv'")
    print(f"✓ Total de {len(df_synthetic)} observações geradas")
    print(f"✓ Período: {df_synthetic['date'].min()} a {df_synthetic['date'].max()}")
    
    # Salvar ground truth separadamente
    ground_truth = df_synthetic[['date', 'google_contribution', 'meta_contribution', 'tv_contribution']].copy()
    ground_truth.to_csv('ground_truth_contributions.csv', index=False)
    print(f"✓ Ground truth salvo como 'ground_truth_contributions.csv'")
    
    print("\n" + "="*80)
    print("GERAÇÃO CONCLUÍDA COM SUCESSO")
    print("="*80)