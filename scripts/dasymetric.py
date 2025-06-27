import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from itertools import product
from tqdm import tqdm

def model_poprf(
    covar_arrays,
    area_raster,
    area_id_column,
    area_level,
    gdf_uc,
    covariate_keys,
    plot=True
):
    """
    Use Random Forest Regressor to learn optimal weights for covariates to minimize RMSE.

    Parameters:
    - covar_arrays: dict of normalized covariate arrays {cov_name: 2D np.array}
    - district_raster: 2D np.array with district int IDs
    - gdf_dist_uc: DataFrame with 'ADM2_PCODE' and 'm_population'
    - total_uc: Total population to disaggregate
    - covariate_keys: list of covariate keys to use (must be in covar_arrays)
    - plot: whether to plot feature importances

    Returns:
    - best_weights: dict of {covariate: relative importance}
    - rmse: float
    - result_df: DataFrame with dist_int, pop_predicted, m_population, abs_error
    """
    # Flatten all arrays
    flat_data = {cov: covar_arrays[cov].flatten() for cov in covariate_keys}
    area_flat = area_raster.flatten()

    # Keep only valid district pixels
    mask = area_flat > 0
    X = np.stack([flat_data[cov][mask] for cov in covariate_keys], axis=1)
    area_ids = area_flat[mask]

    # Aggregate X by district
    X_df = pd.DataFrame(X, columns=covariate_keys)
    X_df[area_id_column] = area_ids
    agg_X = X_df.groupby(area_id_column).mean()

    # Get target population per district
    pop_df = gdf_uc[[f'{area_level}_PCODE', f'{area_level}_TH', 'geometry', 'density']].copy()
    pop_df[area_id_column] = pop_df[f'{area_level}_PCODE'].astype(int)
    pop_df = pop_df.set_index(area_id_column)

    # Align and join
    data = agg_X.join(pop_df, how='inner')
    data['log_density'] = np.log1p(data['density'])
    y_linear = data['density'].values
    y_log = data['log_density'].values
    X_aligned = data[covariate_keys].values
    
    gdf_plot = gpd.GeoDataFrame(data.reset_index(), geometry='geometry', crs=gdf_uc.crs)
    
    if plot:
        # === SCATTER PLOTS ===
        plt.figure(figsize=(5 * len(covariate_keys), 4))
        for i, cov in enumerate(covariate_keys):
            plt.subplot(1, len(covariate_keys), i + 1)
            plt.scatter(X_aligned[:, i], data['density'].values, alpha=0.6, edgecolors='k', linewidths=0.3)
            plt.xlabel(cov)
            plt.ylabel('Population Density')
            plt.title(f'Density vs {cov}')
            plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot spatial distribution of population density
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        gdf_plot.plot(column='log_density', cmap='plasma', legend=True, ax=ax)
        ax.set_title('Log(Population Density) by District')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

        # Optional: Plot spatial distribution of each covariate
        for cov in covariate_keys:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            gdf_plot.plot(column=cov, cmap='plasma', legend=True, ax=ax)
            ax.set_title(f'{cov} (Aggregated) by District')
            ax.axis('off')
            plt.tight_layout()
            plt.show()

    # Fit Random Forest
    model_linear = RandomForestRegressor(random_state=42)
    model_linear.fit(X_aligned, y_linear)
    y_pred_linear = model_linear.predict(X_aligned)
    rmse_linear = np.sqrt(mean_squared_error(y_linear, y_pred_linear))
    y_mean = y_linear.mean()
    rrmse_linear = rmse_linear / y_linear.mean()
    
    importance_scores = model_linear.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': covariate_keys,
        'Importance': importance_scores
    }).sort_values(by='Importance', ascending=False)
    
    display(importance_df)

    return data, model_linear, rmse_linear, rrmse_linear

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pymc as pm
import arviz as az

def model_pop_bayesian(
    covar_arrays,
    area_raster,
    area_id_column,
    area_level,
    gdf_uc,
    covariate_keys,
    plot=True
):
    # Flatten covariate arrays
    flat_data = {cov: covar_arrays[cov].flatten() for cov in covariate_keys}
    area_flat = area_raster.flatten()

    # Valid pixel mask
    mask = area_flat > 0
    X = np.stack([flat_data[cov][mask] for cov in covariate_keys], axis=1)
    area_ids = area_flat[mask]

    # Aggregate covariates by area
    X_df = pd.DataFrame(X, columns=covariate_keys)
    X_df[area_id_column] = area_ids
    agg_X = X_df.groupby(area_id_column).mean()

    # Population data
    pop_df = gdf_uc[[f'{area_level}_PCODE', f'{area_level}_TH', 'geometry', 'density']].copy()
    pop_df[area_id_column] = pop_df[f'{area_level}_PCODE'].astype(int)
    pop_df = pop_df.set_index(area_id_column)

    # Merge covariates and population
    data = agg_X.join(pop_df, how='inner')
    data['log_density'] = np.log1p(data['density'])
    y_log = data['log_density'].values
    X_aligned = data[covariate_keys].values

    gdf_plot = gpd.GeoDataFrame(data.reset_index(), geometry='geometry', crs=gdf_uc.crs)

    if plot:
        for i, cov in enumerate(covariate_keys):
            plt.figure(figsize=(5, 4))
            plt.scatter(X_aligned[:, i], data['density'].values, alpha=0.6, edgecolors='k', linewidths=0.3)
            plt.xlabel(cov)
            plt.ylabel('Population Density')
            plt.title(f'Density vs {cov}')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        gdf_plot.plot(column='log_density', cmap='plasma', legend=True, ax=ax)
        ax.set_title('Log(Population Density) by District')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    # Bayesian linear regression with PyMC
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=1, shape=len(covariate_keys))
        sigma = pm.HalfNormal("sigma", sigma=1)
        mu = pm.math.dot(X_aligned, beta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_log)
        trace = pm.sample(1000, tune=1000, cores=2, return_inferencedata=True)

    # Posterior predictive
    posterior_beta = trace.posterior['beta'].mean(dim=["chain", "draw"]).values
    y_log_pred = X_aligned @ posterior_beta
    y_pred = np.expm1(y_log_pred)

    rmse = np.sqrt(mean_squared_error(data['density'].values, y_pred))
    rrmse = rmse / data['density'].mean()

    data['predicted_density'] = y_pred
    importance_df = pd.DataFrame({
        'Feature': covariate_keys,
        'PosteriorMeanBeta': posterior_beta
    }).sort_values(by='PosteriorMeanBeta', key=np.abs, ascending=False)

    return data, model, trace, rmse, rrmse, importance_df
