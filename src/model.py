from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from xgboost import XGBRegressor


def get_models():
    """
    Return all regression models.
    """

    models = {

        "Linear Regression":
            LinearRegression(),

        "Ridge Regression":
            Ridge(alpha=1.0),

        "Lasso Regression":
            Lasso(alpha=0.1),

        "ElasticNet":
            ElasticNet(alpha=0.1, l1_ratio=0.5),

        "Decision Tree":
            DecisionTreeRegressor(random_state=42),

        "Random Forest":
            RandomForestRegressor(
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            ),

        "XGBoost":
            XGBRegressor(
                random_state=42,
                objective="reg:squarederror",
                verbosity=0
            ),
    }

    return models