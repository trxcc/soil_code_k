try:
    from .auto_sklearn import AutoSklearnModel
    from .catboost import CatBoostModel
    from .cnn import CNNModel
    from .ft_transformer import FTTransformerModel
    from .lightgbm import LightGBMModel
    from .mlp import MLPModel
    from .random_forest import RandomForestModel
    from .resnet import ResNetModel
    from .xgboost import XGBoostModel
except:
    from .deep_forest import DeepForestModel

    # from .catboost import CatBoostModel
    # from .random_forest import RandomForestModel


# from .deep_forest import DeepForestModel
# from .catboost import CatBoostModel
# from .random_forest import RandomForestModel
# from .random_forest import RandomForestModel
# from .auto_sklearn import AutoSklearnModel
# from .ft_transformer import FTTransformerModel
# from .mlp import MLPModel
# from .resnet import ResNetModel
# from .xgboost import XGBoostModel
# from .catboost import CatBoostModel
# from .lightgbm import LightGBMModel
# from .cnn import CNNModel


def get_model(model_name, *args, **kwargs):
    model_name = model_name.lower()

    try:
        MODELS = {
            "randomforest": RandomForestModel,
            "autosklearn": AutoSklearnModel,
            "fttransformer": FTTransformerModel,
            "mlp": MLPModel,
            "resnet": ResNetModel,
            "xgboost": XGBoostModel,
            "lightgbm": LightGBMModel,
            "cnn": CNNModel,
            "catboost": CatBoostModel,
        }
    except:
        MODELS = {
            "deepforest": DeepForestModel,
            # 'catboost': CatBoostModel,
            # 'randomforest': RandomForestModel,
        }

    if model_name not in MODELS.keys():
        raise Exception(f"Model {model_name} not found.")

    return MODELS[model_name](*args, **kwargs)
